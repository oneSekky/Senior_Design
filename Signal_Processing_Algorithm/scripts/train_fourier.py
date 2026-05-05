"""
train_fourier.py
================
Trains a Fourier-descriptor-based letter image generation model.

Complementary to (NOT a replacement for) best_side_mount_model.pt.

Input representation — no letter labels needed:
  HP-filter acc_x, acc_y, acc_z from the box CSV
  Form complex signal z(t) = ax(t) + i·ay(t)
  FFT → apply P(ω) = A(ω) / −ω²  (zero-phase position-equivalent)
    * This is mathematically identical to zero-phase double integration
      (filtfilt+cumsum), but done in one step — no causal drift accumulation.
    * DC and near-DC (<0.5 Hz) are zeroed to prevent blow-up from 1/ω².
  Keep first N_DESC=32 complex coefficients → 64 real features (re, im)
  Also keep first N_DESC magnitude coefficients of acc_z → 32 more features
  Total: 96 features per stroke

Same paired (IMU CSV, 64×64 PNG) data as the existing model.
GPU: CUDA + mixed-precision (AMP) — ~10x faster than CPU.

Outputs:
  models/best_fourier_model.pt
  models/scaler_fourier.pkl
  models/fourier_training_history.png
  models/fourier_predictions.png
"""

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
from PIL import Image
from scipy import ndimage
from scipy import signal as sp_signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT  = SCRIPT_DIR / ".." / ".." / "Test_Data" / "side_mount"
CSV_DIR    = DATA_ROOT / "split_csvs"
IMG_DIR    = DATA_ROOT / "split_images"
MODELS_DIR = SCRIPT_DIR / ".." / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
FS          = 104
IMAGE_SIZE  = 64
N_DESC      = 32      # Fourier coefficients kept (positive freqs, excl. DC)
              # → 64 re/im from acc_xy + 32 mag from acc_z = 96 total features
HIDDEN      = 96
EPOCHS      = 150
BATCH_SIZE  = 32
LR          = 3e-4
TEST_SIZE   = 0.15

AUGMENT_FACTOR   = 12
AUG_NOISE_STD    = 0.04
AUG_SCALE_RANGE  = (0.85, 1.15)
AUG_IMG_SHIFT    = 3
AUG_IMG_ROT_DEG  = 8
AUG_ELASTIC_A    = 6.0

DICE_WEIGHT      = 0.7
BCE_POS_WEIGHT   = 5.0

ACTIVITY_THR  = 40.0
SMOOTH_WIN    = 10
MIN_ACTIVE    = 20
MERGE_GAP     = 15

FEATURE_COLS = [
    "acc_x[mg]", "acc_y[mg]", "acc_z[mg]",
    "gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]",
]


# ── Fourier descriptors ───────────────────────────────────────────────────────

def compute_descriptors(acc: np.ndarray, n_desc: int = N_DESC, fs: int = FS) -> np.ndarray:
    """
    Compute Fourier descriptors from a (N, 6) raw IMU segment.

    Returns (3*n_desc,) float32:
      [0         : n_desc]   real parts  of position-equivalent P(ω) for acc_x+i*acc_y
      [n_desc    : 2*n_desc] imag parts  of P(ω)
      [2*n_desc  : 3*n_desc] magnitudes of acc_z position spectrum

    Position-equivalent: P(ω) = A(ω) / −ω²
      Same as zero-phase double integration, but the whole stroke is processed
      at once so drift does not accumulate causally. DC and sub-0.5 Hz zeroed.
    """
    if len(acc) < 8:
        return np.zeros(3 * n_desc, dtype=np.float32)

    b, a = sp_signal.butter(3, 0.5, "high", fs=fs)

    def hp(x):
        return sp_signal.filtfilt(b, a, x.astype(np.float64))

    ax = hp(acc[:, 0])
    ay = hp(acc[:, 1])
    az = hp(acc[:, 2])

    N     = len(ax)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    omega = 2.0 * np.pi * freqs

    # 1/ω² weighting — zero below 0.5 Hz to avoid DC blow-up
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(freqs) > 0.5, -1.0 / (omega ** 2), 0.0)

    # acc_x + i*acc_y → position-equivalent complex spectrum
    Z_xy = np.fft.fft(ax + 1j * ay) * w
    Z_xy[0] = 0.0  # zero translation

    # acc_z → position-equivalent real spectrum (magnitude only — no orientation)
    Z_z = np.abs(np.fft.fft(az) * w)
    Z_z[0] = 0.0

    # Keep first n_desc positive-frequency bins (index 1..n_desc)
    c_xy = Z_xy[1 : n_desc + 1]
    c_z  = Z_z[1 : n_desc + 1]

    return np.concatenate(
        [c_xy.real, c_xy.imag, c_z]
    ).astype(np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

def _active_segment(data: np.ndarray) -> np.ndarray:
    """Return the active-writing sub-segment of a box CSV array."""
    b, a = sp_signal.butter(3, 0.5, "high", fs=FS)
    filt = np.column_stack(
        [sp_signal.filtfilt(b, a, data[:, i]) for i in range(data.shape[1])]
    )
    mag = np.sqrt((filt[:, :3] ** 2).sum(axis=1))
    sm  = np.convolve(mag, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="same")

    # Find runs above threshold
    above = sm > ACTIVITY_THR
    runs, cs, cl = [], 0, 0
    for i, v in enumerate(above):
        if v:
            if cl == 0: cs = i
            cl += 1
        else:
            if cl > 0: runs.append((cs, cs + cl))
            cl = 0
    if cl > 0:
        runs.append((cs, cs + cl))

    # Merge close runs
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])

    # Longest active run
    best_s = best_e = 0
    for s, e in merged:
        if e - s > best_e - best_s:
            best_s, best_e = s, e

    if best_e - best_s >= MIN_ACTIVE:
        return data[best_s:best_e]

    # Fallback: peak-centered 150-sample window
    peak = int(sm.argmax())
    s = max(0, peak - 75)
    e = min(len(data), s + 150)
    return data[s:e]


def load_csv(path: Path) -> np.ndarray:
    df   = pd.read_csv(path, skiprows=1)
    data = df[FEATURE_COLS].values.astype(np.float32)
    seg  = _active_segment(data)
    return compute_descriptors(seg)


def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr   # stroke=1, background=0


def _build_img_map() -> dict:
    m = {}
    for d in IMG_DIR.iterdir():
        if d.is_dir():
            m[re.sub(r"_\d+$", "", d.name)] = d
    return m


def load_dataset():
    X, y, labels = [], [], []
    img_map = _build_img_map()
    missing = 0

    for csv_dir in sorted(CSV_DIR.iterdir()):
        if not csv_dir.is_dir():
            continue
        stem     = csv_dir.name
        img_dir  = img_map.get(stem)
        if img_dir is None:
            print(f"  no image folder for {stem} — skipping")
            continue

        for csv_path in sorted(csv_dir.glob("box_*.csv")):
            box      = csv_path.stem
            img_path = img_dir / f"{box}.png"
            if not img_path.exists():
                missing += 1
                continue
            try:
                img = load_image(img_path)
                if img.sum() < 5:      # blank box
                    continue
                X.append(load_csv(csv_path))
                y.append(img)
                labels.append(f"{stem}/{box}")
            except Exception as exc:
                print(f"  ERROR {stem}/{box}: {exc}")

    if missing:
        print(f"  Skipped {missing} pairs (no matching image)")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), labels


# ── Augmentation ──────────────────────────────────────────────────────────────

def _elastic(img: np.ndarray, alpha: float, sigma: float = 4.0,
             rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    sh = img.shape
    dx = ndimage.gaussian_filter(rng.uniform(-1, 1, sh), sigma) * alpha
    dy = ndimage.gaussian_filter(rng.uniform(-1, 1, sh), sigma) * alpha
    gx, gy = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
    coords  = [np.clip(gy + dy, 0, sh[0] - 1), np.clip(gx + dx, 0, sh[1] - 1)]
    return ndimage.map_coordinates(img, coords, order=1, mode="reflect")


def augment_pair(desc: np.ndarray, img: np.ndarray,
                 rng: np.random.Generator) -> tuple:
    # Descriptor augmentation: amplitude scale + noise
    # (preserves shape; equivalent to writing with different pressure/speed)
    desc = desc.copy() * float(rng.uniform(*AUG_SCALE_RANGE))
    desc += rng.normal(0, AUG_NOISE_STD, desc.shape).astype(np.float32)

    # Image augmentation: shift, rotate, elastic deform
    img  = img.copy()
    dy   = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    dx   = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    img  = np.roll(np.roll(img, dy, 0), dx, 1)
    if dy > 0:  img[:dy,  :] = 0
    elif dy < 0: img[dy:,  :] = 0
    if dx > 0:  img[:,  :dx] = 0
    elif dx < 0: img[:, dx:] = 0
    img = ndimage.rotate(img, rng.uniform(-AUG_IMG_ROT_DEG, AUG_IMG_ROT_DEG),
                         reshape=False, mode="constant", cval=0.0)
    if rng.random() < 0.5:
        img = _elastic(img, AUG_ELASTIC_A, rng=rng)
    return desc, np.clip(img, 0.0, 1.0).astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class StrokeDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = False, aug_factor: int = AUGMENT_FACTOR):
        self._X      = X
        self._y      = y
        self._aug    = augment
        self._factor = aug_factor if augment else 1
        self._n      = len(X)
        self._rng    = np.random.default_rng(0)

    def __len__(self):
        return self._n * self._factor

    def __getitem__(self, idx):
        real = idx % self._n
        desc, img = self._X[real].copy(), self._y[real].copy()
        if self._aug and idx >= self._n:
            desc, img = augment_pair(desc, img, self._rng)
        return torch.from_numpy(desc), torch.from_numpy(img)


# ── Model ─────────────────────────────────────────────────────────────────────

class FourierToImage(nn.Module):
    """
    Small MLP encoder over Fourier descriptors → ConvTranspose decoder → 64×64.

    Input:  (B, 3*N_DESC)   — real, imag from acc_xy position + mag from acc_z
    Output: (B, 64, 64)     — stroke logit map (apply sigmoid for probability)
    """

    def __init__(self, in_dim: int, hidden: int = HIDDEN):
        super().__init__()
        # MLP encoder → 8×8 spatial seed at 16 channels
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),  nn.GELU(), nn.LayerNorm(hidden),
            nn.Dropout(0.4),
            nn.Linear(hidden, 16 * 8 * 8),  # 16ch @ 8×8
        )
        # Upsample+Conv decoder — no checkerboard artifacts: 8→16→32→64
        def up_block(c_in, c_out, groups):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.GroupNorm(groups, c_out),
                nn.GELU(),
            )
        self.decoder = nn.Sequential(
            up_block(16, 16, 4),   # 8  → 16
            up_block(16,  8, 4),   # 16 → 32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(8, 1, 3, padding=1),   # 32 → 64 (logits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).view(x.shape[0], 16, 8, 8)
        return self.decoder(h).squeeze(1)   # (B, 64, 64)


# ── Loss ──────────────────────────────────────────────────────────────────────

def _dice_loss(pred_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # smooth=1.0 prevents float16 underflow in AMP
    p     = torch.sigmoid(pred_logit.float())   # fp32 for stability
    t     = target.float()
    inter = (p * t).sum(dim=(-2, -1))
    union = p.sum(dim=(-2, -1)) + t.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + 1.0) / (union + 1.0)).mean()


_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(-2, -1).contiguous()


def _edge_loss(pred_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on Sobel gradient maps — forces thin sharp strokes."""
    p  = torch.sigmoid(pred_logit.float()).unsqueeze(1)
    t  = target.float().unsqueeze(1)
    kx = _SOBEL_X.to(p.device)
    ky = _SOBEL_Y.to(p.device)
    return (F.l1_loss(F.conv2d(p, kx, padding=1), F.conv2d(t, kx, padding=1)) +
            F.l1_loss(F.conv2d(p, ky, padding=1), F.conv2d(t, ky, padding=1))) * 0.5


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c = torch.clamp(pred.float(), -20.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(
        pred_c, target.float(),
        pos_weight=torch.tensor(BCE_POS_WEIGHT, device=pred.device),
    )
    return (1.0 - DICE_WEIGHT) * bce + DICE_WEIGHT * _dice_loss(pred, target)


# ── Snapshot ──────────────────────────────────────────────────────────────────

def _save_snapshot(model: nn.Module, X_snap: np.ndarray, y_snap: np.ndarray,
                   epoch: int, device: torch.device, use_amp: bool,
                   filename: str = None) -> None:
    """Save a side-by-side ground-truth / prediction grid for fixed val samples."""
    model.eval()
    n = len(X_snap)
    with torch.no_grad():
        xb = torch.from_numpy(X_snap).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = torch.sigmoid(model(xb)).cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    fig.patch.set_facecolor("#111")
    for i in range(n):
        axes[0, i].imshow(y_snap[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(preds[i],  cmap="gray", vmin=0, vmax=1)
        for r in range(2):
            axes[r, i].axis("off")
    axes[0, 0].set_title("GT",   color="#aaa", fontsize=7, loc="left")
    axes[1, 0].set_title("Pred", color="#aaa", fontsize=7, loc="left")
    fig.suptitle(f"Epoch {epoch:03d}", color="white", fontsize=9)
    plt.tight_layout()
    out = filename if filename else f"fourier_snapshot_epoch_{epoch:03d}.png"
    plt.savefig(MODELS_DIR / out, dpi=100, facecolor="#111")
    plt.close()


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"\nDevice: {device}")
    if use_amp:
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU  : {props.name}")
        print(f"  VRAM : {props.total_memory / 1e9:.1f} GB")
        print(f"  AMP  : enabled (mixed precision)")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    X, y, labels = load_dataset()
    print(f"  {len(X)} samples   descriptors {X.shape}   images {y.shape}")
    if len(X) < 10:
        print("ERROR: too few samples — check CSV/image paths.")
        return

    # ── Normalise ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X      = scaler.fit_transform(X).astype(np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42,
    )
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

    train_dl = DataLoader(
        StrokeDataset(X_tr, y_tr, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=use_amp,
    )
    val_dl = DataLoader(
        StrokeDataset(X_val, y_val, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=use_amp,
    )

    # Fixed samples used for per-epoch snapshots
    n_snap  = min(8, len(X_val))
    X_snap  = X_val[:n_snap]
    y_snap  = y_val[:n_snap]

    # ── Model ─────────────────────────────────────────────────────────────────
    in_dim = X.shape[1]   # 3 * N_DESC = 96
    model  = FourierToImage(in_dim=in_dim, hidden=HIDDEN).to(device)
    n_par  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: FourierToImage   params={n_par:,}   in_dim={in_dim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val = float("inf")
    tr_hist, val_hist = [], []

    print(f"\n{'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Best':>8}  {'LR':>10}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_dl.dataset)

        # Validate
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    v_loss += loss_fn(model(xb), yb).item() * len(xb)
        v_loss /= len(val_dl.dataset)

        tr_hist.append(tr_loss)
        val_hist.append(v_loss)
        scheduler.step()

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(),
                       MODELS_DIR / "best_loss_fourier_model.pt")
            _save_snapshot(model, X_snap, y_snap, epoch, device, use_amp,
                           filename="fourier_snapshot_best_loss.png")
            model.train()

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"{epoch:>6}  {tr_loss:>8.4f}  {v_loss:>8.4f}  "
                  f"{best_val:>8.4f}  {lr_now:>10.2e}")
            _save_snapshot(model, X_snap, y_snap, epoch, device, use_amp)
            model.train()

    # ── Save final-epoch model (visually best) and scaler ────────────────────
    torch.save(model.state_dict(), MODELS_DIR / "best_fourier_model.pt")
    _save_snapshot(model, X_snap, y_snap, EPOCHS, device, use_amp,
                   filename="fourier_snapshot_best.png")

    with open(MODELS_DIR / "scaler_fourier.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved  best_fourier_model.pt  (final epoch — best visual quality)")
    print(f"Saved  best_loss_fourier_model.pt  (min val loss = {best_val:.4f})")
    print(f"Saved  scaler_fourier.pkl")

    # ── Training curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")
    ax.plot(tr_hist,  color="#44aaff", lw=1.5, label="train")
    ax.plot(val_hist, color="#ff8844", lw=1.5, label="val")
    ax.axhline(best_val, color="#55ff55", lw=0.8, ls="--",
               label=f"best val = {best_val:.4f}")
    ax.set_xlabel("Epoch", color="#aaa", fontsize=10)
    ax.set_ylabel("Dice + BCE loss", color="#aaa", fontsize=10)
    ax.set_title("FourierToImage — Training History", color="#ddd", fontsize=11)
    ax.legend(facecolor="#222", labelcolor="#ccc", fontsize=9)
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "fourier_training_history.png", dpi=120, facecolor="#111")
    plt.close()

    # ── Prediction samples (final epoch weights) ──────────────────────────────
    model.eval()

    n_show = min(16, len(X_val))
    with torch.no_grad():
        xb  = torch.from_numpy(X_val[:n_show]).to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = torch.sigmoid(model(xb)).cpu().numpy()

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2, 4))
    fig.patch.set_facecolor("#111")
    for i in range(n_show):
        axes[0, i].imshow(y_val[i],  cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(preds[i], cmap="gray", vmin=0, vmax=1)
        for r in range(2):
            axes[r, i].axis("off")
    axes[0, 0].set_title("Ground truth", color="#ddd", fontsize=8, loc="left")
    axes[1, 0].set_title("Predicted",    color="#ddd", fontsize=8, loc="left")
    plt.suptitle(
        f"FourierToImage predictions  |  best_val={best_val:.4f}  "
        f"|  {n_par:,} params  |  {n_show} val samples",
        color="white", fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "fourier_predictions.png", dpi=120, facecolor="#111")
    plt.close()
    print("Saved  fourier_training_history.png  +  fourier_predictions.png")


if __name__ == "__main__":
    train()
