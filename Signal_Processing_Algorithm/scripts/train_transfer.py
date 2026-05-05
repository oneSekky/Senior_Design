"""
train_transfer.py
=================
Option-2 transfer learning: pre-trained OnHW IMUEncoder → new image decoder
trained on the side-mount (CSV + PNG) dataset.

Pipeline
  Input  : raw acc_x, acc_y, acc_z  (first 3 FEATURE_COLS, no filtering needed —
           the encoder was trained on raw OnHW acc and handles gravity itself)
  Encoder: IMUEncoder from train_onhw.py, loaded from encoder_onhw.pt
  Decoder: new lightweight ConvTranspose-free upsampling decoder (same style as
           train_fourier.py) trained from scratch
  Target : 64×64 stroke image

Two-phase training
  Phase 1 (EPOCHS_FROZEN)  — encoder frozen, decoder only
  Phase 2 (EPOCHS_FINETUNE) — all weights unfrozen, lower LR

Outputs (all in Signal_Processing_Algorithm/models/)
  best_transfer_model.pt        — best val-loss full model state dict
  scaler_transfer.pkl           — StandardScaler fitted on side-mount acc data
  transfer_training_history.png — loss curves for both phases
  transfer_snapshot_best.png    — GT vs prediction for fixed val samples
"""

import pickle
import re
import sys
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
from scipy import ndimage, signal as sp_signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_onhw import IMUEncoder  # reuse pre-trained encoder definition

# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = SCRIPT_DIR.parent / "models"
DATA_ROOT  = SCRIPT_DIR.parent.parent / "Test_Data" / "side_mount"
CSV_DIR    = DATA_ROOT / "split_csvs"
IMG_DIR    = DATA_ROOT / "split_images"

# ── Config ────────────────────────────────────────────────────────────────────

FS           = 104
IMAGE_SIZE   = 64
FEATURE_COLS = [
    "acc_x[mg]", "acc_y[mg]", "acc_z[mg]",
    "gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]",
]
ACC_COLS     = [0, 1, 2]   # indices into FEATURE_COLS for raw acc only

FEAT_DIM     = 256          # must match encoder_onhw.pt out_dim
HIDDEN       = 256          # must match encoder_onhw.pt hidden

EPOCHS_FROZEN   = 60        # phase 1: decoder only
EPOCHS_FINETUNE = 60        # phase 2: full fine-tune
BATCH_SIZE      = 32
LR_FROZEN       = 3e-4
LR_FINETUNE     = 5e-5
WEIGHT_DECAY    = 1e-4
TEST_SIZE       = 0.15

AUGMENT_FACTOR   = 8
AUG_NOISE_STD    = 0.04
AUG_SCALE_RANGE  = (0.85, 1.15)
AUG_IMG_SHIFT    = 3
AUG_IMG_ROT_DEG  = 8
AUG_ELASTIC_A    = 6.0

DICE_WEIGHT    = 0.7
BCE_POS_WEIGHT = 5.0

# Active-segment detection parameters
ACTIVITY_THR = 40.0
SMOOTH_WIN   = 10
MIN_ACTIVE   = 20
MERGE_GAP    = 15

N_SNAP = 8   # number of fixed val samples shown in snapshots


# ── Decoder + full model ──────────────────────────────────────────────────────

def _up_block(c_in: int, c_out: int, groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(groups, c_out),
        nn.GELU(),
    )


class ImageDecoder(nn.Module):
    """256-dim feature vector → (64, 64) logits."""

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.seed = nn.Linear(feat_dim, 16 * 8 * 8)
        self.dec  = nn.Sequential(
            _up_block(16, 16, 4),                               # 8  → 16
            _up_block(16,  8, 4),                               # 16 → 32
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(8, 1, 3, padding=1),                      # 32 → 64 (logits)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.dec(self.seed(feat).view(-1, 16, 8, 8)).squeeze(1)


class TransferModel(nn.Module):
    """Pre-trained IMUEncoder + new ImageDecoder."""

    def __init__(self, feat_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.encoder = IMUEncoder(in_ch=3, hidden=hidden, out_dim=feat_dim)
        self.decoder = ImageDecoder(feat_dim)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(self.encoder(x, lengths))

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(True)


# ── Data loading ──────────────────────────────────────────────────────────────

def _active_segment(data: np.ndarray) -> np.ndarray:
    b, a = sp_signal.butter(3, 0.5, "high", fs=FS)
    filt = np.column_stack(
        [sp_signal.filtfilt(b, a, data[:, i]) for i in range(data.shape[1])]
    )
    mag = np.sqrt((filt[:, :3] ** 2).sum(axis=1))
    sm  = np.convolve(mag, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="same")

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

    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])

    best_s = best_e = 0
    for s, e in merged:
        if e - s > best_e - best_s:
            best_s, best_e = s, e

    if best_e - best_s >= MIN_ACTIVE:
        return data[best_s:best_e]

    peak = int(sm.argmax())
    s = max(0, peak - 75)
    e = min(len(data), s + 150)
    return data[s:e]


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr   # stroke=1, background=0


def load_dataset():
    img_map = {}
    for d in IMG_DIR.iterdir():
        if d.is_dir():
            img_map[re.sub(r"_\d+$", "", d.name)] = d

    X_raw, Y, labels = [], [], []
    missing = 0

    for csv_dir in sorted(CSV_DIR.iterdir()):
        if not csv_dir.is_dir():
            continue
        img_dir = img_map.get(csv_dir.name)
        if img_dir is None:
            continue
        for csv_path in sorted(csv_dir.glob("box_*.csv")):
            img_path = img_dir / f"{csv_path.stem}.png"
            if not img_path.exists():
                missing += 1
                continue
            try:
                img = _load_image(img_path)
                if img.sum() < 5:
                    continue
                df  = pd.read_csv(csv_path, skiprows=1)
                raw = df[FEATURE_COLS].values.astype(np.float32)  # (T, 6)
                seg = _active_segment(raw)                         # (T', 6)
                acc = seg[:, ACC_COLS]                             # (T', 3)  raw acc
                if len(acc) < 5:
                    continue
                X_raw.append(acc)
                Y.append(img)
                labels.append(f"{csv_dir.name}/{csv_path.stem}")
            except Exception as exc:
                print(f"  ERROR {csv_dir.name}/{csv_path.stem}: {exc}")

    if missing:
        print(f"  Skipped {missing} pairs (no matching image)")
    return X_raw, np.array(Y, dtype=np.float32), labels


# ── Augmentation ──────────────────────────────────────────────────────────────

def _elastic(img: np.ndarray, alpha: float, rng: np.random.Generator) -> np.ndarray:
    sh = img.shape
    dx = ndimage.gaussian_filter(rng.uniform(-1, 1, sh), 4.0) * alpha
    dy = ndimage.gaussian_filter(rng.uniform(-1, 1, sh), 4.0) * alpha
    gx, gy = np.meshgrid(np.arange(sh[1]), np.arange(sh[0]))
    coords = [np.clip(gy + dy, 0, sh[0]-1), np.clip(gx + dx, 0, sh[1]-1)]
    return ndimage.map_coordinates(img, coords, order=1, mode="reflect")


def augment_pair(acc: np.ndarray, img: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # IMU: amplitude scale + noise
    acc = acc.copy() * float(rng.uniform(*AUG_SCALE_RANGE))
    acc += rng.normal(0, AUG_NOISE_STD, acc.shape).astype(np.float32)

    # Image: shift, rotate, elastic
    img = img.copy()
    dy  = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    dx  = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    angle = float(rng.uniform(-AUG_IMG_ROT_DEG, AUG_IMG_ROT_DEG))
    img = ndimage.rotate(img, angle, reshape=False, order=1, mode="constant", cval=0)
    img = _elastic(img, AUG_ELASTIC_A, rng)
    return acc, img.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TransferDataset(Dataset):
    def __init__(self, X_raw: list, Y: np.ndarray,
                 scaler: StandardScaler, augment: bool = False):
        # Build augmented pool at construction time
        rng = np.random.default_rng(42)
        self._data: list[tuple[np.ndarray, np.ndarray]] = []

        for acc, img in zip(X_raw, Y):
            acc_s = scaler.transform(acc).astype(np.float32)
            self._data.append((acc_s, img))
            if augment:
                for _ in range(AUGMENT_FACTOR):
                    a, i = augment_pair(acc, img, rng)
                    self._data.append((scaler.transform(a).astype(np.float32), i))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        acc, img = self._data[idx]
        return torch.from_numpy(acc), torch.from_numpy(img), len(acc)


def collate_fn(batch):
    xs, ys, lengths = zip(*batch)
    B, C = len(xs), xs[0].shape[1]
    max_len = max(lengths)
    padded = torch.zeros(B, max_len, C)
    for i, (x, l) in enumerate(zip(xs, lengths)):
        padded[i, :l] = x
    return (padded,
            torch.stack(ys),
            torch.tensor(lengths, dtype=torch.long))


# ── Loss ─────────────────────────────────────────────────────────────────────

def _dice_loss(pred_logits: torch.Tensor, target: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(pred_logits.float())
    num  = 2 * (prob * target.float()).sum(dim=(-2, -1)) + eps
    den  = prob.sum(dim=(-2, -1)) + target.float().sum(dim=(-2, -1)) + eps
    return (1 - num / den).mean()


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c = torch.clamp(pred.float(), -20.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(
        pred_c, target.float(),
        pos_weight=torch.tensor(BCE_POS_WEIGHT, device=pred.device),
    )
    return (1.0 - DICE_WEIGHT) * bce + DICE_WEIGHT * _dice_loss(pred, target)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, optimizer=None, amp_scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            if training:
                optimizer.zero_grad()
            with autocast("cuda"):
                pred = model(x, lengths)
                loss = loss_fn(pred, y)
            if training:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            total_loss += loss.item() * len(y)
            total      += len(y)

    return total_loss / total


# ── Snapshot ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_snapshot(model, X_snap, y_snap, device, filename: str) -> None:
    model.eval()
    n = len(X_snap)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    fig.patch.set_facecolor("#111")

    for i, (acc, gt) in enumerate(zip(X_snap, y_snap)):
        xt = torch.from_numpy(acc).unsqueeze(0).to(device)
        with autocast("cuda"):
            pred = torch.sigmoid(model(xt)).squeeze().cpu().numpy()
        for row, arr in enumerate([gt, pred]):
            axes[row, i].imshow(arr, cmap="gray", vmin=0, vmax=1)
            axes[row, i].axis("off")

    axes[0, 0].set_ylabel("GT",   color="#cccccc", fontsize=8, rotation=0, labelpad=30, va="center")
    axes[1, 0].set_ylabel("Pred", color="#44aaff", fontsize=8, rotation=0, labelpad=30, va="center")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / filename, dpi=100, facecolor="#111", bbox_inches="tight")
    plt.close()


def save_history(phase1_t, phase1_v, phase2_t, phase2_v) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#222")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    e1 = list(range(1, len(phase1_t) + 1))
    e2 = list(range(len(phase1_t) + 1, len(phase1_t) + len(phase2_t) + 1))
    ax.plot(e1, phase1_t, color="#44aaff",  label="train (frozen)")
    ax.plot(e1, phase1_v, color="#44aaff",  linestyle="--", label="val (frozen)")
    if phase2_t:
        ax.plot(e2, phase2_t, color="#55ff55", label="train (finetune)")
        ax.plot(e2, phase2_v, color="#55ff55", linestyle="--", label="val (finetune)")
    if e1 or e2:
        split_x = len(phase1_t) + 0.5
        ax.axvline(split_x, color="#ff8844", linewidth=1, linestyle=":")
        ax.text(split_x + 0.5, ax.get_ylim()[1] * 0.95,
                "unfreeze", color="#ff8844", fontsize=8)
    ax.set_xlabel("epoch", color="white")
    ax.set_ylabel("loss",  color="white")
    ax.set_title("Transfer Training History", color="white")
    ax.legend(facecolor="#333", labelcolor="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "transfer_training_history.png",
                dpi=100, facecolor="#111", bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU  : {props.name}")
        print(f"  VRAM : {props.total_memory / 1e9:.1f} GB")
        print(f"  AMP  : enabled")

    MODELS_DIR.mkdir(exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    X_raw, Y, labels = load_dataset()
    print(f"  {len(X_raw)} samples loaded")

    idx = list(range(len(X_raw)))
    tr_idx, va_idx = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=42, shuffle=True)
    X_tr  = [X_raw[i] for i in tr_idx]
    X_va  = [X_raw[i] for i in va_idx]
    Y_tr  = Y[tr_idx]
    Y_va  = Y[va_idx]
    print(f"  Train: {len(X_tr)}  Val: {len(X_va)}")

    # Fit scaler on train acc data only
    flat = np.vstack(X_tr)
    scaler = StandardScaler().fit(flat)
    with open(MODELS_DIR / "scaler_transfer.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train_ds = TransferDataset(X_tr, Y_tr, scaler, augment=True)
    val_ds   = TransferDataset(X_va, Y_va, scaler, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, persistent_workers=True)

    # Fixed val snapshots
    rng_snap  = np.random.default_rng(7)
    snap_idx  = rng_snap.choice(len(X_va), min(N_SNAP, len(X_va)), replace=False)
    X_snap    = [scaler.transform(X_va[i]).astype(np.float32) for i in snap_idx]
    y_snap    = [Y_va[i] for i in snap_idx]

    # ── Build model ───────────────────────────────────────────────────────────
    model = TransferModel(feat_dim=FEAT_DIM, hidden=HIDDEN).to(device)

    enc_path = MODELS_DIR / "encoder_onhw.pt"
    if enc_path.exists():
        model.encoder.load_state_dict(
            torch.load(enc_path, map_location=device, weights_only=True))
        print(f"\n  Loaded pre-trained encoder from {enc_path.name}")
    else:
        print("\n  WARNING: encoder_onhw.pt not found — encoder initialised randomly")

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"  Encoder params: {n_enc:,}  Decoder params: {n_dec:,}")

    amp_scaler  = GradScaler("cuda")
    best_val    = float("inf")
    p1_t, p1_v, p2_t, p2_v = [], [], [], []

    # ── Phase 1: frozen encoder ───────────────────────────────────────────────
    model.freeze_encoder()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_FROZEN, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_FROZEN, eta_min=1e-5)

    print(f"\n── Phase 1: frozen encoder ({EPOCHS_FROZEN} epochs) ──")
    print(f"  {'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Best':>8}  {'LR':>10}")
    print("  " + "-" * 50)

    for epoch in range(1, EPOCHS_FROZEN + 1):
        t_loss = run_epoch(model, train_loader, device, optimizer, amp_scaler)
        v_loss = run_epoch(model, val_loader,   device)
        scheduler.step()
        p1_t.append(t_loss); p1_v.append(v_loss)

        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            torch.save(model.state_dict(), MODELS_DIR / "best_transfer_model.pt")

        lr = scheduler.get_last_lr()[0]
        marker = "  *" if improved else ""
        print(f"  {epoch:>6}  {t_loss:>8.4f}  {v_loss:>8.4f}"
              f"  {best_val:>8.4f}  {lr:>10.2e}{marker}")

        if epoch % 10 == 0 or epoch == 1:
            save_snapshot(model, X_snap, y_snap, device,
                          f"transfer_snapshot_phase1_ep{epoch:03d}.png")
            save_history(p1_t, p1_v, p2_t, p2_v)

    # ── Phase 2: full fine-tune ───────────────────────────────────────────────
    model.load_state_dict(
        torch.load(MODELS_DIR / "best_transfer_model.pt",
                   map_location=device, weights_only=True))
    model.unfreeze_encoder()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_FINETUNE, eta_min=1e-6)

    print(f"\n── Phase 2: full fine-tune ({EPOCHS_FINETUNE} epochs) ──")
    print(f"  {'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Best':>8}  {'LR':>10}")
    print("  " + "-" * 50)

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        t_loss = run_epoch(model, train_loader, device, optimizer, amp_scaler)
        v_loss = run_epoch(model, val_loader,   device)
        scheduler.step()
        p2_t.append(t_loss); p2_v.append(v_loss)

        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            torch.save(model.state_dict(), MODELS_DIR / "best_transfer_model.pt")

        lr = scheduler.get_last_lr()[0]
        marker = "  *" if improved else ""
        print(f"  {epoch:>6}  {t_loss:>8.4f}  {v_loss:>8.4f}"
              f"  {best_val:>8.4f}  {lr:>10.2e}{marker}")

        if epoch % 10 == 0 or epoch == 1:
            save_snapshot(model, X_snap, y_snap, device,
                          f"transfer_snapshot_phase2_ep{epoch:03d}.png")
            save_history(p1_t, p1_v, p2_t, p2_v)

    # ── Final ─────────────────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(MODELS_DIR / "best_transfer_model.pt",
                   map_location=device, weights_only=True))
    save_snapshot(model, X_snap, y_snap, device, "transfer_snapshot_best.png")
    save_history(p1_t, p1_v, p2_t, p2_v)

    print(f"\nDone.  Best val loss: {best_val:.4f}")
    print(f"  Model  : {MODELS_DIR / 'best_transfer_model.pt'}")
    print(f"  Scaler : {MODELS_DIR / 'scaler_transfer.pkl'}")


if __name__ == "__main__":
    main()
