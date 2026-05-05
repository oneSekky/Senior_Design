"""
train_side_mount_classifier.py
===============================
Train a letter classifier on side-mount accelerometer data.

Architecture
  Same 1D-ResNet encoder as the image generation model → global-avg-pool →
  Dropout → Linear(256, n_classes).  Uses identical preprocessing to
  App_UI/inference.py so the trained model drops straight into the UI.

Data
  Test_Data/side_mount/labels.csv  — image_path, csv_path, label
  Only rows where csv_path is non-empty and label is a single A-Z letter
  are used.  Rows labelled SKIP are ignored.

Outputs (written to Signal_Processing_Algorithm/models/)
  best_side_mount_classifier.pt   — state-dict of the full classifier
  scaler_side_mount_cls.pkl       — StandardScaler fitted on training set
  label_map_side_mount.pkl        — {index: letter} and {letter: index}

Usage
  python train_side_mount_classifier.py
  python train_side_mount_classifier.py --epochs 120 --lr 3e-4
"""

import argparse
import csv
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as sp_signal
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
ROOT        = SCRIPT_DIR.parent.parent
DATA_ROOT   = ROOT / "Test_Data" / "side_mount"
LABELS_CSV  = DATA_ROOT / "labels.csv"
MODELS_DIR  = SCRIPT_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_OUT   = MODELS_DIR / "best_side_mount_classifier.pt"
SCALER_OUT  = MODELS_DIR / "scaler_side_mount_cls.pkl"
LABEL_OUT   = MODELS_DIR / "label_map_side_mount.pkl"

# ── Signal constants (must match App_UI/inference.py exactly) ─────────────────

FS           = 104
N_TIMESTEPS  = int(round(2.30 * FS))   # 239
N_FEAT       = 11

_b_grav,  _a_grav  = sp_signal.butter(3, 0.5, "high", fs=FS)
_b_drift, _a_drift = sp_signal.butter(2, 0.3, "high", fs=FS)


# ── Preprocessing (identical to InferenceEngine._preprocess) ──────────────────

def preprocess(data: np.ndarray) -> np.ndarray:
    """(N, 6) raw IMU → (239, 11) feature array."""
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

    features = np.hstack([filt, mag[:, None], vel, pos]).astype(np.float32)

    if N != N_TIMESTEPS:
        x_old    = np.linspace(0.0, 1.0, N)
        x_new    = np.linspace(0.0, 1.0, N_TIMESTEPS)
        features = np.column_stack([
            np.interp(x_new, x_old, features[:, i])
            for i in range(features.shape[1])
        ])
    return features.astype(np.float32)


def load_csv(path: Path) -> np.ndarray | None:
    """Load a split CSV → (N, 6) array of [acc_x, acc_y, acc_z, gx, gy, gz]."""
    rows = []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("time"):   # header row
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                rows.append([float(parts[i]) for i in range(1, 7)])
            except ValueError:
                continue
    if len(rows) < 5:
        return None
    return np.array(rows, dtype=np.float32)


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(239, 11) → (239, 11) with random perturbations."""
    x = x * float(rng.uniform(0.85, 1.15))
    x = x + rng.normal(0, 0.04, x.shape).astype(np.float32)
    factor  = float(rng.uniform(0.85, 1.15))
    new_len = max(5, int(N_TIMESTEPS * factor))
    t_old   = np.linspace(0.0, 1.0, N_TIMESTEPS)
    t_new   = np.linspace(0.0, 1.0, new_len)
    x = np.column_stack([np.interp(t_new, t_old, x[:, i]) for i in range(N_FEAT)])
    t_new2  = np.linspace(0.0, 1.0, N_TIMESTEPS)
    x = np.column_stack([np.interp(t_new2, np.linspace(0.0, 1.0, new_len), x[:, i])
                         for i in range(N_FEAT)])
    return x.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SideMountDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict], label2idx: dict,
                 scaler: StandardScaler, augment_flag: bool = False):
        self.samples      = samples
        self.label2idx    = label2idx
        self.scaler       = scaler
        self.augment_flag = augment_flag
        self._rng         = np.random.default_rng(42)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s   = self.samples[i]
        x   = s["features"].copy()
        if self.augment_flag:
            x = augment(x, self._rng)
        x   = self.scaler.transform(x).astype(np.float32)
        return torch.from_numpy(x), self.label2idx[s["label"]]


# ── Model ─────────────────────────────────────────────────────────────────────

class _ResBlock1D(nn.Module):
    def __init__(self, ch: int, ks: int = 3, dil: int = 1):
        super().__init__()
        pad      = dil * (ks - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch), nn.GELU(),
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch),
        )

    def forward(self, x):
        return F.gelu(self.net(x) + x)


class SideMountClassifier(nn.Module):
    def __init__(self, n_classes: int, n_feat: int = N_FEAT, hidden: int = 256):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.enc  = nn.Sequential(
            _ResBlock1D(hidden, dil=1), _ResBlock1D(hidden, dil=2),
            _ResBlock1D(hidden, dil=4), _ResBlock1D(hidden, dil=8),
            _ResBlock1D(hidden, dil=1), _ResBlock1D(hidden, dil=2),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        h = self.proj(x).permute(0, 2, 1)   # (B, hidden, T)
        h = self.enc(h).mean(dim=2)          # (B, hidden)
        return self.head(h)                  # (B, n_classes)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load labels ──────────────────────────────────────────────────────────
    with open(LABELS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    samples = []
    for row in rows:
        label = row["label"].strip().upper()
        if label == "SKIP" or not label or not row["csv_path"]:
            continue
        if len(label) != 1 or not label.isalpha():
            continue
        csv_path = DATA_ROOT / row["csv_path"]
        if not csv_path.exists():
            continue
        raw = load_csv(csv_path)
        if raw is None:
            continue
        features = preprocess(raw)
        samples.append({"label": label, "features": features})

    print(f"Loaded {len(samples)} samples")

    from collections import Counter
    dist = Counter(s["label"] for s in samples)
    print("Per-letter counts:", dict(sorted(dist.items())))

    letters    = sorted(dist.keys())
    label2idx  = {l: i for i, l in enumerate(letters)}
    idx2label  = {i: l for l, i in label2idx.items()}
    n_classes  = len(letters)
    print(f"Classes ({n_classes}): {letters}")

    # ── Train / val split (stratified) ───────────────────────────────────────
    random.seed(42)
    by_class = {l: [] for l in letters}
    for s in samples:
        by_class[s["label"]].append(s)

    train_samples, val_samples = [], []
    for l, slist in by_class.items():
        random.shuffle(slist)
        cut = max(1, int(len(slist) * 0.8))
        train_samples.extend(slist[:cut])
        val_samples.extend(slist[cut:])

    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    # ── Fit scaler on train features ─────────────────────────────────────────
    all_train = np.vstack([s["features"] for s in train_samples])
    scaler    = StandardScaler()
    scaler.fit(all_train)

    train_ds = SideMountDataset(train_samples, label2idx, scaler, augment_flag=True)
    val_ds   = SideMountDataset(val_samples,   label2idx, scaler, augment_flag=False)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    # ── Class weights ─────────────────────────────────────────────────────────
    y_train = np.array([label2idx[s["label"]] for s in train_samples])
    cw      = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    cw_t    = torch.tensor(cw, dtype=torch.float32).to(device)

    model     = SideMountClassifier(n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=cw_t)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total   += len(y)
        scheduler.step()

        # val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                val_correct += (model(x).argmax(1) == y).sum().item()
                val_total   += len(y)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total

        if epoch % 10 == 0 or epoch == 1:
            print(f"Ep {epoch:3d}/{args.epochs}  "
                  f"loss={train_loss/train_total:.4f}  "
                  f"train={train_acc:.3f}  val={val_acc:.3f}"
                  + ("  *best*" if val_acc > best_val_acc else ""))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved: {MODEL_OUT}")

    # ── Save scaler and label map ─────────────────────────────────────────────
    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)
    with open(LABEL_OUT, "wb") as f:
        pickle.dump({"label2idx": label2idx, "idx2label": idx2label}, f)

    print(f"Scaler: {SCALER_OUT}")
    print(f"Labels: {LABEL_OUT}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int,   default=100)
    parser.add_argument("--lr",     type=float, default=3e-4)
    args = parser.parse_args()
    train(args)
