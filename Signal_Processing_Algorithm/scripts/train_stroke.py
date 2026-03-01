"""
train_stroke.py

Architecture: IMU → stroke parameters → rendered letter image

Instead of predicting 64×64 pixels (4096 values), the model predicts:
  - N_STROKES strokes, each as N_POINTS (row, col) waypoints + 1 validity flag
  - Total output: N_STROKES × (N_POINTS×2 + 1) = 124 values

The rendered output is stroke-like by construction — a predicted stroke is
always a connected curve, never a blob.

Pipeline:
  GT  = skeletonize(raw_photo) → label components → trace & sample paths
  Out = MLP head → sigmoid coords + validity logits
  Loss = SmoothL1 on coords (valid strokes only) + BCE on validity flags

Trains on ALL pages (no held-out validation split).
"""

import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import signal, ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT  = SCRIPT_DIR / '..' / '..' / 'Test_Data' / 'side_mount'
CSV_DIR    = DATA_ROOT / 'split_csvs'
IMG_DIR    = DATA_ROOT / 'split_images'
MODELS_DIR = SCRIPT_DIR / '..' / 'models'
MODELS_DIR.mkdir(exist_ok=True)

# ── IMU / window ───────────────────────────────────────────────────────────────
FS           = 104
HEAD_PAD_S   = 0.15
CORE_S       = 2.00
TAIL_PAD_S   = 0.15
TOTAL_S      = HEAD_PAD_S + CORE_S + TAIL_PAD_S
N_TIMESTEPS  = int(round(TOTAL_S * FS))          # 240
FEATURE_COLS = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]',
                'gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']

# ── Stroke representation ──────────────────────────────────────────────────────
IMAGE_SIZE   = 64
N_STROKES    = 4      # maximum strokes per letter
N_POINTS     = 15     # waypoints sampled along each stroke
RENDER_WIDTH = 2      # pixel line width when drawing predicted strokes

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS       = 120
BATCH_SIZE   = 16
HIDDEN_DIM   = 256
LR           = 3e-4
WARMUP       = 10
PATIENCE     = 60     # stop if training loss doesn't improve (no val split)
VALID_WEIGHT = 5.0    # weight on validity BCE vs coordinate SmoothL1
SNAPSHOT_EVERY = 10

# ── Augmentation (IMU only — stroke targets stay unchanged) ───────────────────
AUGMENT_FACTOR  = 6
AUG_NOISE_STD   = 0.04
AUG_SCALE_RANGE = (0.85, 1.15)
AUG_WARP_RANGE  = (0.80, 1.20)
AUG_DRIFT_PROB  = 0.25
AUG_DRIFT_MAX   = 0.10
AUG_DROP_PROB   = 0.25
AUG_DROP_LEN    = (3, 10)


# ─────────────────────────────────────────────────────────────────────────────
# Stroke extraction from GT images
# ─────────────────────────────────────────────────────────────────────────────

def _order_points(pts):
    """
    Greedily order a set of (row, col) skeleton pixels into a connected path.
    Starts from the topmost-then-leftmost pixel and always moves to the
    nearest unvisited neighbour.
    """
    if len(pts) <= 2:
        return pts

    start = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
    used = np.zeros(len(pts), dtype=bool)
    used[start] = True
    ordered = [start]
    tree = cKDTree(pts)

    while len(ordered) < len(pts):
        last = pts[ordered[-1]]
        k = min(20, len(pts))
        _, idxs = tree.query(last, k=k)
        moved = False
        for idx in idxs:
            idx = int(idx)
            if not used[idx]:
                ordered.append(idx)
                used[idx] = True
                moved = True
                break
        if not moved:                          # shouldn't happen, safety fallback
            rem = np.where(~used)[0]
            if len(rem):
                ordered.append(int(rem[0]))
                used[rem[0]] = True

    return pts[np.array(ordered)]


def extract_strokes(img_arr):
    """
    Convert a raw letter image into stroke waypoints.

    img_arr : (H, W) float32, stroke=1 bg=0
    Returns
      strokes : (N_STROKES, N_POINTS, 2)  float32
                row and col coordinates each normalised to [0, 1]
      valid   : (N_STROKES,)              bool
                True for strokes that actually exist in this letter
    """
    strokes = np.zeros((N_STROKES, N_POINTS, 2), dtype=np.float32)
    valid   = np.zeros(N_STROKES, dtype=bool)

    # Strokes are dark ink on white paper; after 1-arr inversion they are ~0.7-1.0
    binary = (img_arr > 0.3).astype(bool)
    if not binary.any():
        return strokes, valid

    # 2px dilation closes tiny gaps from ink variation without destroying curves.
    # (More dilation turns curved letters like S/a into blobs → skeleton collapses
    # to a vertical line and loses the shape entirely.)
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = binary.copy()
    for _ in range(2):
        dilated = ndimage.binary_dilation(dilated, structure=struct)

    skel = skeletonize(dilated)

    # MUST use 8-connectivity here.  Skeleton lines often step diagonally;
    # the default 4-connectivity treats those diagonal steps as disconnections
    # and splits one stroke into many tiny fragments.
    labeled, n_comp = ndimage.label(skel, structure=struct)

    components = []
    for i in range(1, n_comp + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 3:
            continue
        pts = np.column_stack([ys, xs]).astype(np.float32)
        components.append(_order_points(pts))

    # Longest strokes first so the most informative ones always fill slots 0..k
    components.sort(key=lambda x: -len(x))
    components = components[:N_STROKES]

    H, W = img_arr.shape
    for i, pts in enumerate(components):
        idx     = np.linspace(0, len(pts) - 1, N_POINTS).astype(int)
        sampled = pts[idx].astype(np.float32)
        sampled[:, 0] /= H    # row → [0, 1]
        sampled[:, 1] /= W    # col → [0, 1]
        strokes[i] = sampled
        valid[i]   = True

    return strokes, valid


# ─────────────────────────────────────────────────────────────────────────────
# Stroke renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_strokes(strokes, valid, size=IMAGE_SIZE, width=RENDER_WIDTH):
    """
    strokes : (N_STROKES, N_POINTS, 2)  normalised (row, col) in [0, 1]
    valid   : (N_STROKES,)              bool
    Returns : (H, W) float32 in [0, 1]
    """
    img  = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)

    for s in range(len(valid)):
        if not valid[s]:
            continue
        pts = strokes[s]   # (P, 2): row, col
        # PIL uses (x, y) = (col, row) in pixel units
        pix = [(float(p[1] * size), float(p[0] * size)) for p in pts]
        if len(pix) >= 2:
            draw.line(pix, fill=255, width=width)

    return np.array(img, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (IMU + images, reuses train_side_mount.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    df       = pd.read_csv(path, skiprows=1)
    data     = df[FEATURE_COLS].values.astype(np.float32)
    N        = len(data)
    b, a     = signal.butter(3, 0.5, 'high', fs=FS)
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i]) if N > 9 else data[:, i]

    magnitude = np.sqrt((filtered[:, :3] ** 2).sum(axis=1))
    dt        = 1.0 / FS
    acc_xy    = filtered[:, :2]
    vel_xy    = np.cumsum(acc_xy * dt, axis=0)
    b2, a2    = signal.butter(2, 0.3, 'high', fs=FS)
    for i in range(2):
        vel_xy[:, i] = signal.filtfilt(b2, a2, vel_xy[:, i]) if N > 9 else vel_xy[:, i]
    pos_xy = np.cumsum(vel_xy * dt, axis=0)
    for i in range(2):
        pos_xy[:, i] = signal.filtfilt(b2, a2, pos_xy[:, i]) if N > 9 else pos_xy[:, i]

    features  = np.hstack([filtered, magnitude[:, None], vel_xy, pos_xy])
    n_feat    = features.shape[1]
    start_idx = N - N_TIMESTEPS
    if start_idx >= 0:
        seg = features[start_idx:N]
    else:
        seg = np.vstack([np.zeros((-start_idx, n_feat), dtype=np.float32), features[:N]])
    return seg.astype(np.float32)


def load_image(path):
    img = Image.open(path).convert('L')
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr          # stroke=1, bg=0


def _build_img_dir_map():
    mapping = {}
    for d in IMG_DIR.iterdir():
        if d.is_dir():
            mapping[re.sub(r'_\d+$', '', d.name)] = d
    return mapping


def load_dataset():
    """
    Returns
      X        : (N, T, F)
      Y_strokes: (N, N_STROKES, N_POINTS, 2)  float32
      Y_valid  : (N, N_STROKES)               bool
      Y_raw    : (N, H, W)                    float32  (for visualisation only)
      labels   : list[str]
    """
    X, Y_strokes, Y_valid, Y_raw, labels = [], [], [], [], []
    missing = skipped = errors = 0
    img_dir_map = _build_img_dir_map()

    for csv_page_dir in sorted(CSV_DIR.iterdir()):
        if not csv_page_dir.is_dir():
            continue
        stem         = csv_page_dir.name
        img_page_dir = img_dir_map.get(stem)
        if img_page_dir is None:
            print(f'  WARNING: no image folder for {stem}, skipping')
            continue

        for csv_path in sorted(csv_page_dir.glob('box_*.csv')):
            box      = csv_path.stem
            img_path = img_page_dir / f'{box}.png'
            if not img_path.exists():
                missing += 1
                continue
            try:
                img_arr         = load_image(img_path)
                strokes, valid  = extract_strokes(img_arr)
                if not valid.any():
                    skipped += 1
                    continue
                X.append(load_csv(csv_path))
                Y_strokes.append(strokes)
                Y_valid.append(valid)
                Y_raw.append(img_arr)
                labels.append(f'{stem}/{box}')
            except Exception as e:
                errors += 1
                print(f'  ERROR {stem}/{box}: {e}')

    if missing: print(f'  Skipped {missing} pairs (no matching image)')
    if skipped: print(f'  Skipped {skipped} samples (no strokes found)')
    if errors:  print(f'  {errors} load errors')

    return (np.array(X,         dtype=np.float32),
            np.array(Y_strokes, dtype=np.float32),
            np.array(Y_valid,   dtype=bool),
            np.array(Y_raw,     dtype=np.float32),
            labels)


# ─────────────────────────────────────────────────────────────────────────────
# IMU augmentation  (stroke targets unchanged — the letter doesn't change)
# ─────────────────────────────────────────────────────────────────────────────

def _drift(imu, rng):
    imu     = imu.copy()
    n_segs  = rng.integers(2, 6)
    seg_len = len(imu) // n_segs
    for s in range(n_segs):
        st = s * seg_len
        en = min(st + seg_len, len(imu))
        dr = rng.uniform(-AUG_DRIFT_MAX, AUG_DRIFT_MAX, imu.shape[1]).astype(np.float32)
        imu[st:en] += np.linspace(0, 1, en - st)[:, None].astype(np.float32) * dr
    return imu


def _dropout(imu, rng):
    imu = imu.copy()
    for _ in range(rng.integers(1, 4)):
        dl = int(rng.integers(*AUG_DROP_LEN))
        st = int(rng.integers(1, max(2, len(imu) - dl)))
        imu[st:min(st + dl, len(imu))] = imu[st - 1]
    return imu


def _augment_imu(imu, rng):
    imu  = imu.copy() * np.float32(rng.uniform(*AUG_SCALE_RANGE))
    imu += rng.normal(0, AUG_NOISE_STD, imu.shape).astype(np.float32)
    if rng.random() < AUG_DRIFT_PROB:
        imu = _drift(imu, rng)
    if rng.random() < AUG_DROP_PROB:
        imu = _dropout(imu, rng)
    # Time-warp
    warp    = rng.uniform(*AUG_WARP_RANGE)
    new_len = max(int(len(imu) * warp), 2)
    xs_in   = np.linspace(0, 1, len(imu))
    xs_new  = np.linspace(0, 1, new_len)
    warped  = np.stack([np.interp(xs_new, xs_in, imu[:, c])
                        for c in range(imu.shape[1])], 1)
    xs_out  = np.linspace(0, 1, N_TIMESTEPS)
    return np.stack([np.interp(xs_out, np.linspace(0, 1, new_len), warped[:, c])
                     for c in range(warped.shape[1])], 1).astype(np.float32)


def augment_dataset(X, Y_strokes_flat, Y_valid_float, seed=0):
    """
    X              : (N, T, F)
    Y_strokes_flat : (N, N_STROKES, N_POINTS*2)
    Y_valid_float  : (N, N_STROKES)  float
    Stroke targets are copied unchanged; only IMU is augmented.
    """
    rng = np.random.default_rng(seed)
    Xs, Ys, Yv = [X], [Y_strokes_flat], [Y_valid_float]
    for k in range(AUGMENT_FACTOR):
        Xc = np.stack([_augment_imu(X[i], rng) for i in range(len(X))])
        Xs.append(Xc)
        Ys.append(Y_strokes_flat.copy())
        Yv.append(Y_valid_float.copy())
        print(f'  Augmentation pass {k + 1}/{AUGMENT_FACTOR} done')
    return (np.concatenate(Xs, 0),
            np.concatenate(Ys, 0),
            np.concatenate(Yv, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    def __init__(self, ch, ks=3, dil=1):
        super().__init__()
        pad     = dil * (ks - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch), nn.GELU(),
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch),
        )
    def forward(self, x):
        return F.gelu(self.net(x) + x)


class IMUToStrokes(nn.Module):
    """
    IMU sequence → stroke parameters.

    Encoder : same dilated 1D-ResBlock stack as v19 pixel model
    Head    : two-layer MLP → N_STROKES × (N_POINTS*2 + 1)
                                          └─ coords ─┘ └validity┘
    """
    def __init__(self, n_feat, hidden=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.enc  = nn.Sequential(
            ResBlock1D(hidden, dil=1),
            ResBlock1D(hidden, dil=2),
            ResBlock1D(hidden, dil=4),
            ResBlock1D(hidden, dil=8),
            ResBlock1D(hidden, dil=1),
            ResBlock1D(hidden, dil=2),
        )
        self.drop = nn.Dropout(0.3)
        out_dim   = N_STROKES * (N_POINTS * 2 + 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        B  = x.shape[0]
        h  = self.proj(x).permute(0, 2, 1)                    # (B, hidden, T)
        h  = self.enc(h)
        h  = self.drop(h).mean(dim=2)                          # (B, hidden)
        out = self.head(h).view(B, N_STROKES, N_POINTS * 2 + 1)
        coords   = torch.sigmoid(out[:, :, :N_POINTS * 2])    # (B, S, P*2) ∈ [0,1]
        validity = out[:, :, -1]                               # (B, S) raw logits
        return coords, validity


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def stroke_loss(pred_coords, pred_valid_logits, gt_coords, gt_valid_float):
    """
    pred_coords       : (B, S, P*2)  sigmoid output ∈ [0,1]
    pred_valid_logits : (B, S)       raw logits
    gt_coords         : (B, S, P*2)  ground-truth normalised coords
    gt_valid_float    : (B, S)       float 0/1
    """
    valid_loss = F.binary_cross_entropy_with_logits(pred_valid_logits, gt_valid_float)
    mask = gt_valid_float.bool().unsqueeze(-1).expand_as(pred_coords)
    if mask.any():
        coord_loss = F.smooth_l1_loss(pred_coords[mask], gt_coords[mask])
    else:
        coord_loss = pred_coords.sum() * 0.0     # zero with grad
    total = coord_loss + VALID_WEIGHT * valid_loss
    return total, coord_loss.item(), valid_loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_snapshot(model, X_snap, Y_raw_snap, Y_strokes_snap, Y_valid_snap,
                  lbl_snap, device, epoch, tr_loss):
    model.eval()
    n_show = min(6, len(X_snap))
    with torch.no_grad():
        X_t              = torch.from_numpy(X_snap[:n_show]).float().to(device)
        pred_coords_t, pred_valid_t = model(X_t)
        pred_coords = pred_coords_t.cpu().numpy()           # (n, S, P*2)
        pred_valid  = (torch.sigmoid(pred_valid_t) > 0.5).cpu().numpy()   # (n, S) bool

    fig, axes = plt.subplots(n_show, 3, figsize=(9, 2.5 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    axes[0, 0].set_title('GT photo',     fontsize=8)
    axes[0, 1].set_title('GT strokes',   fontsize=8)
    axes[0, 2].set_title('Pred strokes', fontsize=8)
    fig.suptitle(f'Epoch {epoch + 1}  |  train loss={tr_loss:.4f}', fontsize=10)

    for pi in range(n_show):
        gt_s  = Y_strokes_snap[pi]                           # (S, P, 2)
        pr_s  = pred_coords[pi].reshape(N_STROKES, N_POINTS, 2)
        gt_r  = render_strokes(gt_s,  Y_valid_snap[pi])
        pr_r  = render_strokes(pr_s,  pred_valid[pi])

        axes[pi, 0].imshow(Y_raw_snap[pi], cmap='gray', vmin=0, vmax=1)
        axes[pi, 1].imshow(gt_r,           cmap='gray', vmin=0, vmax=1)
        axes[pi, 2].imshow(pr_r,           cmap='gray', vmin=0, vmax=1)
        axes[pi, 0].set_ylabel(lbl_snap[pi].split('/')[-1], fontsize=5,
                               rotation=0, labelpad=45)
        for c in range(3):
            axes[pi, c].set_xticks([])
            axes[pi, c].set_yticks([])

    plt.tight_layout()
    p = MODELS_DIR / f'stroke_snapshot_epoch_{epoch + 1:03d}.png'
    plt.savefig(p, dpi=100)
    plt.close()
    print(f'  Snapshot: {p.name}')
    model.train()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load ──────────────────────────────────────────────────────────────────
    print('\nLoading dataset...')
    X, Y_strokes, Y_valid, Y_raw, labels = load_dataset()
    n_samples = len(X)
    n_stroke_pixels = int(Y_valid.sum())
    print(f'  Loaded {n_samples} samples  '
          f'({n_stroke_pixels} valid strokes total, '
          f'avg {n_stroke_pixels / n_samples:.1f} per letter)')

    # ── Scale IMU ─────────────────────────────────────────────────────────────
    scaler  = StandardScaler()
    n, t, f = X.shape
    X_sc    = scaler.fit_transform(X.reshape(-1, f)).reshape(n, t, f)
    with open(MODELS_DIR / 'scaler_stroke.pkl', 'wb') as fh:
        pickle.dump(scaler, fh)
    print(f'  Scaler saved.')

    # ── Snapshot set: first N samples (also in training — that's fine for demo) ─
    SNAP_N      = min(20, n_samples // 4)
    X_snap      = X_sc[:SNAP_N]
    Y_raw_snap  = Y_raw[:SNAP_N]
    Y_str_snap  = Y_strokes[:SNAP_N]       # (SNAP_N, S, P, 2)
    Y_val_snap  = Y_valid[:SNAP_N]         # (SNAP_N, S) bool
    lbl_snap    = labels[:SNAP_N]

    # ── Flatten strokes for DataLoader ────────────────────────────────────────
    # (N, S, P, 2) → (N, S, P*2)
    Y_strokes_flat = Y_strokes.reshape(n_samples, N_STROKES, N_POINTS * 2)
    Y_valid_float  = Y_valid.astype(np.float32)

    # ── Augment ───────────────────────────────────────────────────────────────
    print(f'\nAugmenting training set (×{AUGMENT_FACTOR})...')
    X_aug, Ys_aug, Yv_aug = augment_dataset(X_sc, Y_strokes_flat, Y_valid_float, seed=42)
    print(f'  Augmented size: {len(X_aug)}')

    # ── DataLoader ────────────────────────────────────────────────────────────
    ds     = TensorDataset(torch.from_numpy(X_aug).float(),
                           torch.from_numpy(Ys_aug).float(),
                           torch.from_numpy(Yv_aug).float())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=True, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_feat = X_aug.shape[2]
    model  = IMUToStrokes(n_feat).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nIMUToStrokes  params={n_params:,}  '
          f'output={N_STROKES}×({N_POINTS}×2+1)={N_STROKES*(N_POINTS*2+1)} values')

    opt = torch.optim.AdamW(model.parameters(), lr=LR / WARMUP, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6)

    best_loss = float('inf')
    pat_count = 0
    model_pt  = str(MODELS_DIR / 'best_stroke_model.pt')
    history   = {'loss': [], 'coord': [], 'valid': []}

    print(f'\nTraining for up to {EPOCHS} epochs...\n')

    for epoch in range(EPOCHS):
        # Linear LR warmup
        if epoch < WARMUP:
            for pg in opt.param_groups:
                pg['lr'] = LR * (epoch + 1) / WARMUP

        model.train()
        tr_total = tr_coord = tr_valid_l = tr_n = 0

        for x_b, ys_b, yv_b in loader:
            x_b  = x_b.to(device)
            ys_b = ys_b.to(device)
            yv_b = yv_b.to(device)
            opt.zero_grad()
            pred_coords, pred_valid = model(x_b)
            loss, cl, vl = stroke_loss(pred_coords, pred_valid, ys_b, yv_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs           = x_b.size(0)
            tr_total    += loss.item() * bs
            tr_coord    += cl * bs
            tr_valid_l  += vl * bs
            tr_n        += bs

        tr_total   /= tr_n
        tr_coord   /= tr_n
        tr_valid_l /= tr_n

        history['loss'].append(tr_total)
        history['coord'].append(tr_coord)
        history['valid'].append(tr_valid_l)

        if epoch >= WARMUP:
            sch.step(epoch - WARMUP)

        lr = opt.param_groups[0]['lr']
        print(f'Epoch {epoch + 1:3d}/{EPOCHS}  '
              f'loss={tr_total:.4f}  coord={tr_coord:.4f}  '
              f'valid={tr_valid_l:.4f}  lr={lr:.2e}')

        # Save on training loss improvement (training on all data → no val loss)
        if epoch >= WARMUP:
            if tr_total < best_loss:
                best_loss = tr_total
                pat_count = 0
                torch.save(model.state_dict(), model_pt)
            else:
                pat_count += 1
                if pat_count >= PATIENCE:
                    print(f'\nEarly stopping at epoch {epoch + 1}')
                    break

        if (epoch + 1) % SNAPSHOT_EVERY == 0 or epoch == 0:
            save_snapshot(model, X_snap, Y_raw_snap, Y_str_snap, Y_val_snap,
                          lbl_snap, device, epoch, tr_total)

    # ── Load best checkpoint ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_pt, weights_only=True))
    model.eval()

    # ── Training history plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, title in zip(axes,
                               ['loss', 'coord', 'valid'],
                               ['Total Loss', 'Coord SmoothL1', 'Validity BCE']):
        ax.plot(history[key])
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'training_history_stroke.png', dpi=150)
    plt.close()

    # ── Final prediction grid ──────────────────────────────────────────────────
    n_show = min(10, SNAP_N)
    with torch.no_grad():
        X_t              = torch.from_numpy(X_snap[:n_show]).float().to(device)
        pred_coords_t, pred_valid_t = model(X_t)
        pred_c = pred_coords_t.cpu().numpy()
        pred_v = (torch.sigmoid(pred_valid_t) > 0.5).cpu().numpy()

    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    axes[0, 0].set_title('GT photo',   fontsize=9)
    axes[0, 1].set_title('GT strokes', fontsize=9)
    axes[0, 2].set_title('Predicted',  fontsize=9)

    for pi in range(n_show):
        gt_s = Y_str_snap[pi]                                # (S, P, 2)
        pr_s = pred_c[pi].reshape(N_STROKES, N_POINTS, 2)
        gt_r = render_strokes(gt_s, Y_val_snap[pi])
        pr_r = render_strokes(pr_s, pred_v[pi])

        axes[pi, 0].imshow(Y_raw_snap[pi], cmap='gray', vmin=0, vmax=1)
        axes[pi, 1].imshow(gt_r,           cmap='gray', vmin=0, vmax=1)
        axes[pi, 2].imshow(pr_r,           cmap='gray', vmin=0, vmax=1)
        axes[pi, 0].set_ylabel(lbl_snap[pi].split('/')[-1], fontsize=6,
                               rotation=0, labelpad=55)
        for c in range(3):
            axes[pi, c].set_xticks([])
            axes[pi, c].set_yticks([])

    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'predictions_stroke.png', dpi=150)
    plt.close()
    print('Final predictions saved.')

    best_ep = int(np.argmin(history['loss']))
    print(f'\nBest train loss: {history["loss"][best_ep]:.4f} at epoch {best_ep + 1}')
    print('=' * 60)
    print(f'Done.  Model → {model_pt}')


if __name__ == '__main__':
    main()
