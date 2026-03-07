"""
train_side_mount.py

Architecture v20: IMU → 64×64 rendered-stroke image

GT pipeline (from train_stroke.py):
  binary = threshold(raw_photo)
  dilated = binary_dilation(binary, 2px)   # close gaps without destroying curves
  skel = skeletonize(dilated)              # 1px medial axis
  components = label(skel, 8-connectivity) # keeps diagonal steps connected
  GT_target = render_strokes(extract_components)  # clean thin lines, ~2px wide

Model is trained against these clean rendered stroke images.
Trains on ALL pages (no held-out val split — the v19 val split caused early
stopping to fire when the model still predicted blobs).
"""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR / ".." / ".." / "Test_Data" / "side_mount"
CSV_DIR = DATA_ROOT / "split_csvs"
IMG_DIR = DATA_ROOT / "split_images"
MODELS_DIR = SCRIPT_DIR / ".." / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Sampling / window ─────────────────────────────────────────────────────────
FS = 104
HEAD_PAD_S = 0.15
CORE_S = 2.00
TAIL_PAD_S = 0.15
TOTAL_S = HEAD_PAD_S + CORE_S + TAIL_PAD_S

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE = 64
N_TIMESTEPS = int(round(TOTAL_S * FS))  # 240
FEATURE_COLS = [
    "acc_x[mg]",
    "acc_y[mg]",
    "acc_z[mg]",
    "gyro_x[mdps]",
    "gyro_y[mdps]",
    "gyro_z[mdps]",
]

EPOCHS = 120
BATCH_SIZE = 16
HIDDEN_DIM = 256

# ── Loss settings ──────────────────────────────────────────────────────────────
# GT is rendered from clean extracted strokes (~2px wide).
# Dice loss directly penalises blob outputs — a blob against a thin-stroke GT
# gets Dice ≈ 0, forcing the model to be sparse and precise.
BCE_POS_WEIGHT = 2.0  # reduced from 6; Dice handles imbalance
DICE_WEIGHT = 0.8  # fraction of loss that is Dice; remainder is BCE
RENDER_WIDTH = 2

# Snapshot threshold
SNAP_THRESHOLD = 0.4

# Stroke extraction constants (mirrors train_stroke.py)
N_STROKES = 4
N_POINTS = 15

# ── Augmentation ───────────────────────────────────────────────────────────────
AUGMENT_FACTOR = 6
AUG_NOISE_STD = 0.04
AUG_SCALE_RANGE = (0.85, 1.15)
AUG_WARP_RANGE = (0.80, 1.20)
AUG_IMG_SHIFT = 3
AUG_IMG_ROT_DEG = 8
AUG_ELASTIC_ALPHA = 6.0
AUG_DRIFT_PROB = 0.25
AUG_DRIFT_MAX = 0.1
AUG_DROP_PROB = 0.25
AUG_DROP_LEN = (3, 10)


# ── Stroke GT helpers (same pipeline as train_stroke.py) ──────────────────────


def _order_points(pts):
    """Greedily order skeleton pixels into a connected path."""
    if len(pts) <= 2:
        return pts
    start = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
    used = np.zeros(len(pts), dtype=bool)
    used[start] = True
    ordered = [start]
    tree = cKDTree(pts)
    while len(ordered) < len(pts):
        last = pts[ordered[-1]]
        _, idxs = tree.query(last, k=min(20, len(pts)))
        moved = False
        for idx in idxs:
            idx = int(idx)
            if not used[idx]:
                ordered.append(idx)
                used[idx] = True
                moved = True
                break
        if not moved:
            rem = np.where(~used)[0]
            if len(rem):
                ordered.append(int(rem[0]))
                used[rem[0]] = True
    return pts[np.array(ordered)]


def _extract_strokes(img_arr):
    """Extract up to N_STROKES ordered paths from a letter image."""
    strokes = np.zeros((N_STROKES, N_POINTS, 2), dtype=np.float32)
    valid = np.zeros(N_STROKES, dtype=bool)
    binary = (img_arr > 0.3).astype(bool)
    if not binary.any():
        return strokes, valid
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = binary.copy()
    for _ in range(2):  # 2px: close gaps, preserve curves
        dilated = ndimage.binary_dilation(dilated, structure=struct)
    skel = skeletonize(dilated)
    labeled, n_comp = ndimage.label(skel, structure=struct)  # 8-connectivity
    components = []
    for i in range(1, n_comp + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 3:
            continue
        pts = np.column_stack([ys, xs]).astype(np.float32)
        components.append(_order_points(pts))
    components.sort(key=lambda x: -len(x))
    components = components[:N_STROKES]
    H, W = img_arr.shape
    for i, pts in enumerate(components):
        idx = np.linspace(0, len(pts) - 1, N_POINTS).astype(int)
        s = pts[idx].astype(np.float32)
        s[:, 0] /= H
        s[:, 1] /= W
        strokes[i] = s
        valid[i] = True
    return strokes, valid


def make_stroke_target(img_arr):
    """
    Convert a raw letter photo into a clean rendered-stroke target image.
    Uses the same extract → render pipeline as train_stroke.py.
    Returns float32 (H, W) in [0, 1].
    """
    strokes, valid = _extract_strokes(img_arr)
    if not valid.any():
        return np.zeros_like(img_arr)
    size = img_arr.shape[0]
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for s in range(N_STROKES):
        if not valid[s]:
            continue
        pts = strokes[s]  # (P, 2): row, col
        pix = [(float(p[1] * size), float(p[0] * size)) for p in pts]
        if len(pix) >= 2:
            draw.line(pix, fill=255, width=RENDER_WIDTH)
    return np.array(canvas, dtype=np.float32) / 255.0


# ── Data loading ───────────────────────────────────────────────────────────────


def load_csv(path):
    df = pd.read_csv(path, skiprows=1)
    data = df[FEATURE_COLS].values.astype(np.float32)
    N = len(data)

    b, a = signal.butter(3, 0.5, "high", fs=FS)  # gravity removal
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i]) if N > 9 else data[:, i]

    magnitude = np.sqrt((filtered[:, :3] ** 2).sum(axis=1))
    dt = 1.0 / FS
    acc_xy = filtered[:, :2]
    vel_xy = np.cumsum(acc_xy * dt, axis=0)
    b_hp, a_hp = signal.butter(2, 0.3, "high", fs=FS)  # remove noice from integration
    for i in range(2):
        vel_xy[:, i] = (
            signal.filtfilt(b_hp, a_hp, vel_xy[:, i]) if N > 9 else vel_xy[:, i]
        )
    pos_xy = np.cumsum(vel_xy * dt, axis=0)
    for i in range(2):
        pos_xy[:, i] = (
            signal.filtfilt(b_hp, a_hp, pos_xy[:, i]) if N > 9 else pos_xy[:, i]
        )

    features = np.hstack([filtered, magnitude[:, None], vel_xy, pos_xy])
    n_feat = features.shape[1]
    start_idx = N - N_TIMESTEPS
    if start_idx >= 0:
        seg = features[start_idx:N]
    else:
        seg = np.vstack(
            [np.zeros((-start_idx, n_feat), dtype=np.float32), features[:N]]
        )
    return seg.astype(np.float32)


def load_image(path):
    img = Image.open(path).convert("L")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr  # stroke=1, bg=0


def build_img_dir_map():
    mapping = {}
    for d in IMG_DIR.iterdir():
        if d.is_dir():
            mapping[re.sub(r"_\d+$", "", d.name)] = d
    return mapping


def _page_num(stem):
    m = re.search(r"page-(\d+)", stem)
    return int(m.group(1)) if m else -1


def load_dataset():
    """
    Returns:
      X        : (N, T, F)   IMU sequences
      y_target : (N, H, W)   rendered stroke GT (for training loss)
      y_raw    : (N, H, W)   raw photos (for display only)
      labels   : list of str
    """
    X, y_target, y_raw, labels = [], [], [], []
    missing = skipped = 0
    img_dir_map = build_img_dir_map()

    for csv_page_dir in sorted(CSV_DIR.iterdir()):
        if not csv_page_dir.is_dir():
            continue
        stem = csv_page_dir.name
        img_page_dir = img_dir_map.get(stem)
        if img_page_dir is None:
            print(f"  WARNING: no image folder for {stem}, skipping")
            continue

        for csv_path in sorted(csv_page_dir.glob("box_*.csv")):
            box = csv_path.stem
            img_path = img_page_dir / f"{box}.png"
            if not img_path.exists():
                missing += 1
                continue
            try:
                img_arr = load_image(img_path)
                target_arr = make_stroke_target(img_arr)
                if target_arr.sum() < 5:
                    skipped += 1
                    continue
                X.append(load_csv(csv_path))
                y_target.append(target_arr)
                y_raw.append(img_arr)
                labels.append(f"{stem}/{box}")
            except Exception as e:
                print(f"  ERROR {stem}/{box}: {e}")

    if missing:
        print(f"  Skipped {missing} CSV/PNG pairs (no image)")
    if skipped:
        print(f"  Skipped {skipped} samples (empty stroke target)")

    return (
        np.array(X, dtype=np.float32),
        np.array(y_target, dtype=np.float32),
        np.array(y_raw, dtype=np.float32),
        labels,
    )


# ── Augmentation ───────────────────────────────────────────────────────────────


def elastic_deform(image, alpha, sigma=4.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    shape = image.shape
    dx = ndimage.gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    dy = ndimage.gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    gx, gy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = [np.clip(gy + dy, 0, shape[0] - 1), np.clip(gx + dx, 0, shape[1] - 1)]
    return ndimage.map_coordinates(image, coords, order=1, mode="reflect")


def drift_augment(imu, rng):
    imu = imu.copy()
    n_segs = rng.integers(2, 6)
    seg_len = len(imu) // n_segs
    for s in range(n_segs):
        st = s * seg_len
        en = min(st + seg_len, len(imu))
        dr = rng.uniform(-AUG_DRIFT_MAX, AUG_DRIFT_MAX, imu.shape[1]).astype(np.float32)
        imu[st:en] += np.linspace(0, 1, en - st)[:, None].astype(np.float32) * dr
    return imu


def dropout_augment(imu, rng):
    imu = imu.copy()
    for _ in range(rng.integers(1, 4)):
        dl = int(rng.integers(*AUG_DROP_LEN))
        st = int(rng.integers(1, max(2, len(imu) - dl)))
        imu[st : min(st + dl, len(imu))] = imu[st - 1]
    return imu


def augment_sample(imu, img_raw, img_target, rng):
    # ── IMU augmentation ──
    imu = imu.copy() * rng.uniform(*AUG_SCALE_RANGE)
    imu += rng.normal(0, AUG_NOISE_STD, imu.shape).astype(np.float32)
    if rng.random() < AUG_DRIFT_PROB:
        imu = drift_augment(imu, rng)
    if rng.random() < AUG_DROP_PROB:
        imu = dropout_augment(imu, rng)
    warp = rng.uniform(*AUG_WARP_RANGE)
    new_len = max(int(len(imu) * warp), 2)
    xs_o = np.linspace(0, 1, len(imu))
    xs_n = np.linspace(0, 1, new_len)
    warped = np.stack(
        [np.interp(xs_n, xs_o, imu[:, c]) for c in range(imu.shape[1])], 1
    ).astype(np.float32)
    xs_out = np.linspace(0, 1, N_TIMESTEPS)
    imu = np.stack(
        [
            np.interp(xs_out, np.linspace(0, 1, new_len), warped[:, c])
            for c in range(warped.shape[1])
        ],
        1,
    ).astype(np.float32)

    # ── Image augmentation — transform raw, then re-derive stroke target ──
    dy = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    dx = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    angle = rng.uniform(-AUG_IMG_ROT_DEG, AUG_IMG_ROT_DEG)
    do_elastic = rng.random() < 0.5

    def transform_img(img):
        img = np.roll(np.roll(img.copy(), dy, 0), dx, 1)
        if dy > 0:
            img[:dy, :] = 0
        elif dy < 0:
            img[dy:, :] = 0
        if dx > 0:
            img[:, :dx] = 0
        elif dx < 0:
            img[:, dx:] = 0
        img = ndimage.rotate(img, angle, reshape=False, mode="constant", cval=0.0)
        if do_elastic:
            img = elastic_deform(img, AUG_ELASTIC_ALPHA, rng=rng)
        return np.clip(img, 0, 1).astype(np.float32)

    aug_raw = transform_img(img_raw)
    aug_target = make_stroke_target(aug_raw)  # re-derive from augmented photo

    return imu, aug_raw, aug_target


def augment_dataset(X_tr, yr_tr, yt_tr, seed=0):
    rng = np.random.default_rng(seed)
    Xs, Yrs, Yts = [X_tr], [yr_tr], [yt_tr]
    for k in range(AUGMENT_FACTOR):
        Xc = np.zeros_like(X_tr)
        Yrc = np.zeros_like(yr_tr)
        Ytc = np.zeros_like(yt_tr)
        for i in range(len(X_tr)):
            Xc[i], Yrc[i], Ytc[i] = augment_sample(X_tr[i], yr_tr[i], yt_tr[i], rng)
        Xs.append(Xc)
        Yrs.append(Yrc)
        Yts.append(Ytc)
        print(f"  Augmentation pass {k + 1}/{AUGMENT_FACTOR} done")
    return (np.concatenate(Xs, 0), np.concatenate(Yrs, 0), np.concatenate(Yts, 0))


# ── Model ──────────────────────────────────────────────────────────────────────


class ResBlock1D(nn.Module):
    def __init__(self, ch, ks=3, dil=1):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch),
        )

    def forward(self, x):
        return F.gelu(self.net(x) + x)


class IMUToImage(nn.Module):
    """
    v19: IMU → 64×64 logit map trained against skeleton GT.
    Encoder: 1D ResBlocks → global avg-pool → bottleneck
    Decoder: 4×4 seed → transposed-conv × 4 → 64×64
    """

    def __init__(self, n_feat, hidden=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.enc = nn.Sequential(
            ResBlock1D(hidden, dil=1),
            ResBlock1D(hidden, dil=2),
            ResBlock1D(hidden, dil=4),
            ResBlock1D(hidden, dil=8),
            ResBlock1D(hidden, dil=1),
            ResBlock1D(hidden, dil=2),
        )
        self.drop = nn.Dropout(0.3)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 256 * 4 * 4),
        )
        # 4×4 → 8 → 16 → 32 → 64
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        B = x.shape[0]
        h = self.proj(x).permute(0, 2, 1)
        h = self.enc(h)
        h = self.drop(h)
        h = h.mean(dim=2)
        seed = self.bottleneck(h).view(B, 256, 4, 4)
        logits = self.dec(seed).squeeze(1)  # (B, 64, 64)
        return logits


# ── Loss ───────────────────────────────────────────────────────────────────────


def dice_loss(pred_sigmoid, gt, eps=1e-6):
    """Sørensen–Dice loss averaged over the batch.

    A blob of all-ones vs 3% positive GT gets Dice ≈ 0.94 (terrible),
    while a precise thin-stroke prediction gets Dice ≈ 0. This forces
    the model to be sparse rather than carpet-predicting strokes.
    """
    inter = (pred_sigmoid * gt).sum(dim=(-2, -1))
    card = pred_sigmoid.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (card + eps)).mean()


def image_loss(logits, gt, pos_weight):
    """Combined Dice + weighted BCE.

    Dice (80%): penalises over-prediction / blobs.
    BCE (20%): keeps gradient signal when prediction is near-zero.
    """
    pred_sig = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight)
    dice = dice_loss(pred_sig, gt)
    return DICE_WEIGHT * dice + (1.0 - DICE_WEIGHT) * bce


# ── Snapshot ──────────────────────────────────────────────────────────────────

SNAPSHOT_EVERY = 10


def save_snapshot(model, X_snap, yr_snap, yt_snap, lbl_snap, device, epoch, tr_loss):
    model.eval()
    n_show = min(6, len(X_snap))
    with torch.no_grad():
        X_t = torch.from_numpy(X_snap[:n_show]).float().to(device)
        logits = model(X_t)
        pred = torch.sigmoid(logits).cpu().numpy()  # raw probabilities

    # 4 columns: GT photo | GT strokes | raw prob (sigmoid) | hard threshold
    fig, axes = plt.subplots(n_show, 4, figsize=(12, 2.5 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    axes[0, 0].set_title("GT (raw photo)", fontsize=7)
    axes[0, 1].set_title("GT (stroke target)", fontsize=7)
    axes[0, 2].set_title("Pred (raw sigmoid)", fontsize=7)
    axes[0, 3].set_title(f"Pred >{SNAP_THRESHOLD}", fontsize=7)
    fig.suptitle(f"Epoch {epoch + 1}  |  train loss={tr_loss:.4f}", fontsize=10)

    for pi in range(n_show):
        p_soft = pred[pi]  # [0,1] continuous
        p_hard = (p_soft >= SNAP_THRESHOLD).astype(np.float32)  # binary
        axes[pi, 0].imshow(yr_snap[pi], cmap="gray", vmin=0, vmax=1)
        axes[pi, 1].imshow(yt_snap[pi], cmap="gray", vmin=0, vmax=1)
        axes[pi, 2].imshow(p_soft, cmap="hot", vmin=0, vmax=1)  # heatmap
        axes[pi, 3].imshow(p_hard, cmap="gray", vmin=0, vmax=1)
        axes[pi, 0].set_ylabel(
            lbl_snap[pi].split("/")[-1], fontsize=5, rotation=0, labelpad=45
        )
        for c in range(4):
            axes[pi, c].set_xticks([])
            axes[pi, c].set_yticks([])

    plt.tight_layout()
    p = MODELS_DIR / f"snapshot_epoch_{epoch + 1:03d}.png"
    plt.savefig(p, dpi=100)
    plt.close()
    print(f"  Snapshot saved: {p.name}")
    model.train()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Load existing best_side_mount_model.pt and continue training",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading dataset...")
    X, y_target, y_raw, labels = load_dataset()
    print(f"  Loaded {len(X)} samples  (target density: {y_target.mean():.4f})")

    # ── Scale IMU ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    n, t, f = X.shape
    X_sc = scaler.fit_transform(X.reshape(-1, f)).reshape(n, t, f)
    with open(MODELS_DIR / "scaler_side_mount.pkl", "wb") as f_out:
        pickle.dump(scaler, f_out)

    # ── Snapshot set: first N samples (visualisation only, also in training) ──
    SNAP_N = min(20, n // 4)
    X_snap = X_sc[:SNAP_N]
    yr_snap = y_raw[:SNAP_N]
    yt_snap = y_target[:SNAP_N]
    lbl_snap = labels[:SNAP_N]

    # ── Augment ───────────────────────────────────────────────────────────────
    print(f"Augmenting training set (×{AUGMENT_FACTOR})...")
    X_sc, y_raw, y_target = augment_dataset(X_sc, y_raw, y_target, seed=42)
    print(f"  Augmented size: {len(X_sc)}")

    pos_weight_t = torch.tensor(BCE_POS_WEIGHT, device=device)
    print(f"  pos_weight={BCE_POS_WEIGHT}  render_width={RENDER_WIDTH}px")

    tr_ds = TensorDataset(
        torch.from_numpy(X_sc).float(), torch.from_numpy(y_target).float()
    )
    tr_ld = DataLoader(
        tr_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0
    )

    n_feat = X_sc.shape[2]
    model = IMUToImage(n_feat).to(device)
    print(
        f"\nModel: IMUToImage (v20)  "
        f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    model_pt = str(MODELS_DIR / "best_side_mount_model.pt")

    if args.resume:
        if Path(model_pt).exists():
            model.load_state_dict(
                torch.load(model_pt, weights_only=True, map_location=device)
            )
            print(f"  Resumed from {model_pt}")
        else:
            print(
                f"  WARNING: --resume passed but no saved model found at {model_pt}; starting fresh"
            )

    TARGET_LR = 3e-4
    WARMUP = 10
    PATIENCE = 50
    opt = torch.optim.AdamW(
        model.parameters(), lr=TARGET_LR / WARMUP, weight_decay=1e-4
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_loss = float("inf")
    pat_count = 0
    history = {"loss": []}

    print(f"\nTraining for {EPOCHS} epochs...\n")
    for epoch in range(EPOCHS):
        if epoch < WARMUP:
            for pg in opt.param_groups:
                pg["lr"] = TARGET_LR * (epoch + 1) / WARMUP

        model.train()
        tr_sum = tr_n = 0
        for x_b, yt_b in tr_ld:
            x_b = x_b.to(device)
            yt_b = yt_b.to(device)
            opt.zero_grad()
            loss = image_loss(model(x_b), yt_b, pos_weight_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item() * x_b.size(0)
            tr_n += x_b.size(0)
        tr_loss = tr_sum / tr_n
        history["loss"].append(tr_loss)

        if epoch >= WARMUP:
            sch.step(epoch - WARMUP)
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:3d}/{EPOCHS}  loss={tr_loss:.4f}  lr={lr:.2e}")

        # Save on training loss (all data is training — no val split)
        if epoch >= WARMUP:
            if tr_loss < best_loss:
                best_loss = tr_loss
                pat_count = 0
                torch.save(model.state_dict(), model_pt)
            else:
                pat_count += 1
                if pat_count >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        if (epoch + 1) % SNAPSHOT_EVERY == 0 or epoch == 0:
            save_snapshot(
                model, X_snap, yr_snap, yt_snap, lbl_snap, device, epoch, tr_loss
            )

    model.load_state_dict(torch.load(model_pt, weights_only=True))
    model.eval()

    # ── Training history ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="Train BCE")
    ax.set_title("BCE Loss (rendered stroke GT)")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "training_history_side_mount.png", dpi=150)
    plt.close()

    # ── Final predictions on snapshot set ─────────────────────────────────────
    n_show = min(10, SNAP_N)
    with torch.no_grad():
        logits_f = model(torch.from_numpy(X_snap[:n_show]).float().to(device))
        pred_f = torch.sigmoid(logits_f).cpu().numpy()

    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    axes[0, 0].set_title("GT (raw)", fontsize=9)
    axes[0, 1].set_title("GT (stroke)", fontsize=9)
    axes[0, 2].set_title("Predicted", fontsize=9)

    for pi in range(n_show):
        p_hard = (pred_f[pi] >= SNAP_THRESHOLD).astype(np.float32)
        axes[pi, 0].imshow(yr_snap[pi], cmap="gray", vmin=0, vmax=1)
        axes[pi, 1].imshow(yt_snap[pi], cmap="gray", vmin=0, vmax=1)
        axes[pi, 2].imshow(p_hard, cmap="gray", vmin=0, vmax=1)
        axes[pi, 0].set_ylabel(
            lbl_snap[pi].split("/")[-1], fontsize=6, rotation=0, labelpad=55
        )
        for c in range(3):
            axes[pi, c].set_xticks([])
            axes[pi, c].set_yticks([])

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "predictions_side_mount.png", dpi=150)
    plt.close()
    print("Predictions saved.")

    best_ep = int(np.argmin(history["loss"]))
    print(f"\nBest train BCE: {history['loss'][best_ep]:.4f} at epoch {best_ep + 1}")
    print("=" * 60)
    print(f"Done. Model: {model_pt}")


if __name__ == "__main__":
    main()
