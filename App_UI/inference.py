"""
inference.py — PyTorch IMUToImage model wrapper.

Preprocessing matches train_side_mount.py (HEAD/PyTorch branch) exactly:
  - Butterworth HP filter (gravity removal)
  - Magnitude + velocity + position features → 11 total
  - Window: last N_TIMESTEPS=239 samples, zero-padded at front if shorter
  - StandardScaler applied per-feature

This windowing is critical — the model was trained with the last 239 samples
of each 208-sample box, zero-padded at the front. Passing arbitrary lengths
produces degraded output.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal

_ROOT = Path(__file__).parent.parent
_MODEL_PATH = _ROOT / "Signal_Processing_Algorithm" / "models" / "best_side_mount_model.pt"
_SCALER_PATH = _ROOT / "Signal_Processing_Algorithm" / "models" / "scaler_side_mount.pkl"

FS = 104
# Match train_side_mount.py HEAD exactly:
# HEAD_PAD_S=0.15, CORE_S=2.00, TAIL_PAD_S=0.15 → TOTAL_S=2.30
N_TIMESTEPS = int(round(2.30 * FS))  # 239


# ── Model definition (mirrors train_side_mount.py HEAD) ──────────────────────

class _ResBlock1D(nn.Module):
    def __init__(self, ch: int, ks: int = 3, dil: int = 1):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.net(x) + x)


class _IMUToImage(nn.Module):
    def __init__(self, n_feat: int = 11, hidden: int = 256):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.enc = nn.Sequential(
            _ResBlock1D(hidden, dil=1), _ResBlock1D(hidden, dil=2),
            _ResBlock1D(hidden, dil=4), _ResBlock1D(hidden, dil=8),
            _ResBlock1D(hidden, dil=1), _ResBlock1D(hidden, dil=2),
        )
        self.drop = nn.Dropout(0.3)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 256 * 4 * 4),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.proj(x).permute(0, 2, 1)   # (B, hidden, T)
        h = self.enc(h)
        h = self.drop(h)
        h = h.mean(dim=2)                    # global avg-pool over time → (B, hidden)
        seed = self.bottleneck(h).view(B, 256, 4, 4)
        return self.dec(seed).squeeze(1)     # (B, 64, 64) logits


# ── Public API ───────────────────────────────────────────────────────────────

class InferenceEngine:
    def __init__(self):
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {_MODEL_PATH}")
        if not _SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler not found: {_SCALER_PATH}")

        self._model = _IMUToImage(n_feat=11, hidden=256)
        self._model.load_state_dict(
            torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=False)
        )
        self._model.eval()

        with open(_SCALER_PATH, "rb") as f:
            self._scaler = pickle.load(f)

        # Butterworth HP filters (match training exactly)
        self._b_grav, self._a_grav = sp_signal.butter(3, 0.5, "high", fs=FS)
        self._b_drift, self._a_drift = sp_signal.butter(2, 0.3, "high", fs=FS)

    def predict(self, raw_samples: np.ndarray) -> np.ndarray | None:
        """
        raw_samples : (N, 6) float32 — [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
                      in mg / mdps, as received from the IMU.
        Returns     : (64, 64) float32 in [0, 1] (sigmoid output), or None if too short.
        """
        if len(raw_samples) < 5:
            return None

        features = self._preprocess(raw_samples)           # always (239, 11)
        scaled = self._scaler.transform(features).astype(np.float32)
        x = torch.from_numpy(scaled).unsqueeze(0)          # (1, 239, 11)
        with torch.no_grad():
            logits = self._model(x)                        # (1, 64, 64)
        return torch.sigmoid(logits).squeeze(0).numpy()    # (64, 64)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Replicate train_side_mount.py load_csv() exactly:
          1. HP filter all 6 channels
          2. Compute magnitude, velocity, position → 11 features
          3. Window to N_TIMESTEPS=239: take last 239 samples,
             zero-pad at the FRONT if the stroke is shorter.
        """
        data = data.astype(np.float32)
        N = len(data)

        # Gravity removal
        filt = np.zeros_like(data)
        for i in range(6):
            filt[:, i] = sp_signal.filtfilt(self._b_grav, self._a_grav, data[:, i])

        # Magnitude (7th feature)
        mag = np.sqrt((filt[:, :3] ** 2).sum(axis=1))

        # Velocity: integrate acc_xy, remove drift
        dt = 1.0 / FS
        vel = np.cumsum(filt[:, :2] * dt, axis=0)
        if N > 9:
            for i in range(2):
                vel[:, i] = sp_signal.filtfilt(self._b_drift, self._a_drift, vel[:, i])

        # Position: integrate velocity, remove drift
        pos = np.cumsum(vel * dt, axis=0)
        if N > 9:
            for i in range(2):
                pos[:, i] = sp_signal.filtfilt(self._b_drift, self._a_drift, pos[:, i])

        features = np.hstack([filt, mag[:, None], vel, pos]).astype(np.float32)  # (N, 11)

        # Window exactly as training does: last N_TIMESTEPS, zero-pad front if shorter
        start_idx = N - N_TIMESTEPS
        if start_idx >= 0:
            return features[start_idx:]                                    # (239, 11)
        else:
            pad = np.zeros((-start_idx, 11), dtype=np.float32)
            return np.vstack([pad, features])                              # (239, 11)
