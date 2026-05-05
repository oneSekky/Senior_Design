"""
ensemble_visualize.py
=====================
Compares the Normal model, Fourier model, and their averaged ensemble
on a random selection of samples from the test data.

Outputs:
  Signal_Processing_Algorithm/models/ensemble_comparison.png

Rows (top→bottom): Ground Truth | Normal | Fourier | Ensemble (avg)
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
from scipy import signal as sp_signal

# Import Fourier model + preprocessing from train_fourier
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_fourier import (
    FourierToImage, compute_descriptors, _active_segment,
    FEATURE_COLS, FS, N_DESC, IMAGE_SIZE,
)

MODELS_DIR = SCRIPT_DIR.parent / "models"
DATA_ROOT  = SCRIPT_DIR.parent.parent / "Test_Data" / "side_mount"
CSV_DIR    = DATA_ROOT / "split_csvs"
IMG_DIR    = DATA_ROOT / "split_images"

N_SHOW = 10
SEED   = 7


# ── Normal model (mirrors inference.py / train_side_mount.py) ────────────────

class _ResBlock1D(nn.Module):
    def __init__(self, ch: int, ks: int = 3, dil: int = 1):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch), nn.GELU(),
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + x)


class _IMUToImage(nn.Module):
    def __init__(self, n_feat: int = 11, hidden: int = 256):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.enc  = nn.Sequential(
            _ResBlock1D(hidden, dil=1), _ResBlock1D(hidden, dil=2),
            _ResBlock1D(hidden, dil=4), _ResBlock1D(hidden, dil=8),
            _ResBlock1D(hidden, dil=1), _ResBlock1D(hidden, dil=2),
        )
        self.drop = nn.Dropout(0.3)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 256 * 4 * 4),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1),
            nn.GroupNorm(8,  64), nn.GELU(),
            nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=1),
            nn.GroupNorm(8,  32), nn.GELU(),
            nn.ConvTranspose2d( 32,   1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.proj(x).permute(0, 2, 1)
        h = self.enc(h)
        h = self.drop(h)
        h = h.mean(dim=2)
        return self.dec(self.bottleneck(h).view(B, 256, 4, 4)).squeeze(1)


# ── Normal model preprocessing ───────────────────────────────────────────────

_N_STEPS               = int(round(2.30 * FS))   # 239
_b_grav,  _a_grav      = sp_signal.butter(3, 0.5, "high", fs=FS)
_b_drift, _a_drift     = sp_signal.butter(2, 0.3, "high", fs=FS)


def _preprocess_normal(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    N    = len(data)
    filt = np.zeros_like(data)
    for i in range(6):
        filt[:, i] = sp_signal.filtfilt(_b_grav, _a_grav, data[:, i])
    mag = np.sqrt((filt[:, :3] ** 2).sum(axis=1))
    dt  = 1.0 / FS
    vel = np.cumsum(filt[:, :2] * dt, axis=0)
    if N > 9:
        for i in range(2):
            vel[:, i] = sp_signal.filtfilt(_b_drift, _a_drift, vel[:, i])
    pos = np.cumsum(vel * dt, axis=0)
    if N > 9:
        for i in range(2):
            pos[:, i] = sp_signal.filtfilt(_b_drift, _a_drift, pos[:, i])
    feat = np.hstack([filt, mag[:, None], vel, pos]).astype(np.float32)
    if N != _N_STEPS:
        xo = np.linspace(0, 1, N)
        xn = np.linspace(0, 1, _N_STEPS)
        feat = np.column_stack([np.interp(xn, xo, feat[:, i])
                                for i in range(feat.shape[1])])
    return feat.astype(np.float32)


# ── Load models & scalers ─────────────────────────────────────────────────────

def _load_models():
    fourier = FourierToImage(in_dim=3 * N_DESC)
    fourier.load_state_dict(
        torch.load(MODELS_DIR / "best_fourier_model.pt",
                   map_location="cpu", weights_only=True))
    fourier.eval()
    with open(MODELS_DIR / "scaler_fourier.pkl", "rb") as f:
        f_scaler = pickle.load(f)

    normal = _IMUToImage(n_feat=11, hidden=256)
    normal.load_state_dict(
        torch.load(MODELS_DIR / "best_side_mount_model.pt",
                   map_location="cpu", weights_only=False))
    normal.eval()
    with open(MODELS_DIR / "scaler_side_mount.pkl", "rb") as f:
        n_scaler = pickle.load(f)

    return fourier, f_scaler, normal, n_scaler


# ── Collect sample pairs ──────────────────────────────────────────────────────

def _collect_pairs():
    img_map = {}
    for d in IMG_DIR.iterdir():
        if d.is_dir():
            img_map[re.sub(r"_\d+$", "", d.name)] = d

    pairs = []
    for csv_dir in sorted(CSV_DIR.iterdir()):
        if not csv_dir.is_dir():
            continue
        img_dir = img_map.get(csv_dir.name)
        if img_dir is None:
            continue
        for csv_path in sorted(csv_dir.glob("box_*.csv")):
            img_path = img_dir / f"{csv_path.stem}.png"
            if img_path.exists():
                pairs.append((csv_path, img_path))
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading models...")
    fourier, f_scaler, normal, n_scaler = _load_models()

    print("Collecting samples...")
    all_pairs = _collect_pairs()
    rng = np.random.default_rng(SEED)
    chosen = [all_pairs[i] for i in rng.choice(len(all_pairs), N_SHOW, replace=False)]
    print(f"  {N_SHOW} samples selected from {len(all_pairs)} total")

    fig, axes = plt.subplots(4, N_SHOW, figsize=(N_SHOW * 2, 8))
    fig.patch.set_facecolor("#111")

    for i, (csv_path, img_path) in enumerate(chosen):
        # Ground truth
        img = Image.open(img_path).convert("L")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        gt = 1.0 - np.array(img, dtype=np.float32) / 255.0

        # Shared preprocessing: extract active segment
        df   = pd.read_csv(csv_path, skiprows=1)
        data = df[FEATURE_COLS].values.astype(np.float32)
        seg  = _active_segment(data)

        # Fourier prediction
        desc  = compute_descriptors(seg).reshape(1, -1)
        desc  = f_scaler.transform(desc).astype(np.float32)
        with torch.no_grad():
            f_pred = torch.sigmoid(fourier(torch.from_numpy(desc))).squeeze().numpy()

        # Normal prediction
        feat  = _preprocess_normal(seg)
        feat  = n_scaler.transform(feat).astype(np.float32)
        with torch.no_grad():
            n_pred = torch.sigmoid(normal(torch.from_numpy(feat).unsqueeze(0))).squeeze().numpy()

        # Ensemble: simple average
        ens = (f_pred + n_pred) / 2.0

        for row, arr in enumerate([gt, n_pred, f_pred, ens]):
            axes[row, i].imshow(arr, cmap="gray", vmin=0, vmax=1)
            axes[row, i].axis("off")

        # Column header: folder/box label
        label = f"{csv_path.parent.name}\n{csv_path.stem}"
        axes[0, i].set_title(label, color="#555", fontsize=5)

    row_labels  = ["Ground Truth", "Normal model", "Fourier model", "Ensemble (avg)"]
    row_colors  = ["#cccccc",      "#44aaff",       "#ff8844",       "#55ff55"]
    for row, (lbl, col) in enumerate(zip(row_labels, row_colors)):
        axes[row, 0].set_ylabel(lbl, color=col, fontsize=8,
                                rotation=0, labelpad=55, va="center")

    plt.suptitle("Model Ensemble Comparison", color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    out = MODELS_DIR / "ensemble_comparison.png"
    plt.savefig(out, dpi=120, facecolor="#111", bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
