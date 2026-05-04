"""
diagnose_app_vs_train.py

Compares model output for:
  (A) full CSV (training path)
  (B) what the stroke buffer actually captures from the replay stream

Run from App_UI/:  python diagnose_app_vs_train.py
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal
from scipy.signal import lfilter, lfilter_zi

_ROOT = Path(__file__).parent.parent.parent
_MODEL_PATH = _ROOT / "Signal_Processing_Algorithm" / "models" / "best_side_mount_model.pt"
_SCALER_PATH = _ROOT / "Signal_Processing_Algorithm" / "models" / "scaler_side_mount.pkl"
_CSV_DIR = _ROOT / "Test_Data" / "side_mount" / "split_csvs" / "box-page-15-2sec"

FS = 104
N_TIMESTEPS = int(round(2.30 * FS))  # 239
FEATURE_COLS = [
    "acc_x[mg]", "acc_y[mg]", "acc_z[mg]",
    "gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]",
]

# stroke buffer constants (must match stroke_buffer.py)
REST_SAMPLES     = 500
ACTIVITY_THRESHOLD = 40.0
SMOOTH_WINDOW    = 10
STROKE_END_SAMPLES = 31
MIN_STROKE_SAMPLES = 20


# ── Model (mirrors inference.py) ──────────────────────────────────────────────

class _ResBlock1D(nn.Module):
    def __init__(self, ch, ks=3, dil=1):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil), nn.GroupNorm(1, ch), nn.GELU(),
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil), nn.GroupNorm(1, ch),
        )
    def forward(self, x): return F.gelu(self.net(x) + x)

class _IMUToImage(nn.Module):
    def __init__(self, n_feat=11, hidden=256):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.enc = nn.Sequential(
            _ResBlock1D(hidden, dil=1), _ResBlock1D(hidden, dil=2),
            _ResBlock1D(hidden, dil=4), _ResBlock1D(hidden, dil=8),
            _ResBlock1D(hidden, dil=1), _ResBlock1D(hidden, dil=2),
        )
        self.drop = nn.Dropout(0.3)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 256 * 4 * 4),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.GroupNorm(8, 64),  nn.GELU(),
            nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1), nn.GroupNorm(8, 32),  nn.GELU(),
            nn.ConvTranspose2d(32,  1,   4, stride=2, padding=1),
        )
    def forward(self, x):
        B = x.shape[0]
        h = self.proj(x).permute(0, 2, 1)
        h = self.enc(h); h = self.drop(h); h = h.mean(dim=2)
        return self.dec(self.bottleneck(h).view(B, 256, 4, 4)).squeeze(1)


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def _preprocess(raw, scaler):
    """Exact inference.py _preprocess + scaler.transform."""
    data = raw.astype(np.float32)
    N = len(data)
    b, a = sp_signal.butter(3, 0.5, "high", fs=FS)
    filt = np.zeros_like(data)
    for i in range(data.shape[1]):
        filt[:, i] = sp_signal.filtfilt(b, a, data[:, i])
    mag = np.sqrt((filt[:, :3] ** 2).sum(axis=1))
    dt = 1.0 / FS
    vel = np.cumsum(filt[:, :2] * dt, axis=0)
    b2, a2 = sp_signal.butter(2, 0.3, "high", fs=FS)
    if N > 9:
        for i in range(2):
            vel[:, i] = sp_signal.filtfilt(b2, a2, vel[:, i])
    pos = np.cumsum(vel * dt, axis=0)
    if N > 9:
        for i in range(2):
            pos[:, i] = sp_signal.filtfilt(b2, a2, pos[:, i])
    features = np.hstack([filt, mag[:, None], vel, pos]).astype(np.float32)
    start = N - N_TIMESTEPS
    if start >= 0:
        seg = features[start:]
    else:
        seg = np.vstack([np.zeros((-start, 11), np.float32), features])
    return scaler.transform(seg).astype(np.float32)


def run_model(model, scaled):
    x = torch.from_numpy(scaled).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    return torch.sigmoid(logits).squeeze(0).numpy()


# ── Stroke buffer simulation ──────────────────────────────────────────────────

def simulate_stroke_buffer(box_raw):
    """
    Simulates StrokeBuffer fed with: 500 rest samples + box_raw + 500 rest.
    Returns list of (raw_array, label) tuples for each detected stroke.
    """
    rng = np.random.default_rng(42)

    def rest():
        return [rng.normal(0, 4), rng.normal(0, 4), rng.normal(1000, 8),
                rng.normal(0, 15), rng.normal(0, 15), rng.normal(0, 15)]

    stream = (
        [rest() for _ in range(REST_SAMPLES)]
        + box_raw.tolist()
        + [rest() for _ in range(REST_SAMPLES)]
    )

    b, a = sp_signal.butter(3, 0.5, "high", fs=FS)
    zi_1 = lfilter_zi(b, a)
    zi = np.tile(zi_1[:, None], (1, 3)).copy()

    mag_win, raw_buf = [], []
    inactive_count, is_active = 0, False
    captured = []

    for s in stream:
        s = np.asarray(s, dtype=np.float32)
        x3 = s[:3][np.newaxis, :]
        out = np.empty_like(x3)
        for ch in range(3):
            y, zi[:, ch] = lfilter(b, a, x3[:, ch], zi=zi[:, ch])
            out[0, ch] = y[0]
        mag = float(np.sqrt((out[0] ** 2).sum()))
        mag_win.append(mag)
        if len(mag_win) > SMOOTH_WINDOW:
            mag_win.pop(0)
        smoothed = float(np.mean(mag_win))

        if smoothed > ACTIVITY_THRESHOLD:
            is_active = True; inactive_count = 0; raw_buf.append(s)
        elif is_active:
            raw_buf.append(s); inactive_count += 1
            if inactive_count >= STROKE_END_SAMPLES:
                data = np.array(raw_buf[:-STROKE_END_SAMPLES], np.float32)
                if len(data) >= MIN_STROKE_SAMPLES:
                    captured.append(data)
                raw_buf = []; is_active = False; inactive_count = 0

    return captured


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model and scaler...")
    model = _IMUToImage()
    model.load_state_dict(torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=False))
    model.eval()
    with open(_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    box_files = sorted(_CSV_DIR.glob("box_*.csv"))[:12]
    n = len(box_files)

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    axes[0, 0].set_title("Full CSV\n(train path)", fontsize=8)
    axes[0, 1].set_title("Stroke buffer\n(app captures)", fontsize=8)
    axes[0, 2].set_title("Diff |full - buf|", fontsize=8)
    axes[0, 3].set_title("Samples\nCSV vs buffer", fontsize=8)

    thresh = 0.30
    for i, csv_path in enumerate(box_files):
        print(f"\n{csv_path.name}")

        df = pd.read_csv(csv_path, skiprows=1)
        raw = df[FEATURE_COLS].values.astype(np.float32)

        # Full CSV prediction (= what training does)
        scaled_full = _preprocess(raw, scaler)
        pred_full = run_model(model, scaled_full)

        # Stroke buffer prediction (= what app does)
        strokes = simulate_stroke_buffer(raw)
        print(f"  buffer fired {len(strokes)}x: {[len(s) for s in strokes]}")

        if strokes:
            buf_raw = strokes[0]
            scaled_buf = _preprocess(buf_raw, scaler)
            pred_buf = run_model(model, scaled_buf)
            diff = np.abs(pred_full - pred_buf)
            print(f"  CSV N={len(raw)}, buf N={len(buf_raw)}, max_diff={diff.max():.4f}")
            axes[i, 1].imshow(1 - (pred_buf > thresh), cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
            lbl = f"CSV: {len(raw)}\nbuf: {len(buf_raw)}"
            if len(strokes) > 1:
                lbl += f"\n({len(strokes)} strokes!)"
            axes[i, 3].text(0.5, 0.5, lbl, ha="center", va="center",
                            fontsize=9, transform=axes[i, 3].transAxes,
                            color="red" if len(strokes) > 1 else "black")
        else:
            print("  WARNING: nothing captured!")
            for c in (1, 2):
                axes[i, c].text(0.5, 0.5, "NOTHING", ha="center", va="center",
                                transform=axes[i, c].transAxes, color="red")
            axes[i, 3].text(0.5, 0.5, "0 captured", ha="center", va="center",
                            transform=axes[i, 3].transAxes, color="red", fontsize=9)

        axes[i, 0].imshow(1 - (pred_full > thresh), cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(csv_path.stem, fontsize=7, rotation=0, labelpad=55)
        for c in range(4):
            axes[i, c].set_xticks([]); axes[i, c].set_yticks([])

    plt.tight_layout()
    out = Path(__file__).parent / "output" / "diagnose_output.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
