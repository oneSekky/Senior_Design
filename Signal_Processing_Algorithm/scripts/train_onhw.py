"""
train_onhw.py
=============
Trains a 52-class character classifier on the OnHW-chars dataset using
only the rear/end accelerometer (channels 3-5 of the 13-channel .npy files).

Architecture
  IMUEncoder    -- 1D ResNet encoder, variable-length input → fixed-dim
                   feature vector.  Saved separately so it can be reused
                   as a pre-trained backbone for Option-2 transfer learning
                   (IMU → stroke image decoder).
  IMUClassifier -- IMUEncoder + dropout + linear classification head.

Outputs (all written to Signal_Processing_Algorithm/models/)
  best_onhw_classifier.pt   -- best val-accuracy full model state dict
  encoder_onhw.pt           -- encoder weights only  (for transfer)
  scaler_onhw.pkl           -- StandardScaler fitted on train ch 3-5
  label_map_onhw.pkl        -- {char_to_idx, idx_to_char} dicts
  onhw_training_history.png -- loss + accuracy curves
  onhw_confusion_matrix.png -- per-class confusion matrix on val set
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR.parent / "models"
DATA_DIR   = Path(r"C:\Users\sekan\Downloads\onhw-chars_2021-06-30"
                  r"\onhw-chars_2021-06-30\onhw2_both_indep_0")

# ── Config ────────────────────────────────────────────────────────────────────

END_ACC_CH      = [3, 4, 5]   # rear/end accelerometer in the 13-ch .npy files
N_IN_CH         = 3
N_CLASSES       = 52
HIDDEN          = 256
FEAT_DIM        = 256
DROPOUT         = 0.4
EPOCHS          = 150
BATCH_SIZE      = 128
LR              = 3e-4
WEIGHT_DECAY    = 2e-4
LABEL_SMOOTHING = 0.1
# Sequences longer than this percentile of train lengths are truncated
MAX_LEN_PCTILE  = 99

# Augmentation
AUG_SCALE_RANGE   = (0.80, 1.20)   # amplitude scale
AUG_NOISE_STD     = 0.05            # additive Gaussian noise (after scaling)
AUG_WARP_RANGE    = (0.80, 1.20)   # time-warp factor (resample to this × length)
AUG_CHAN_DROP_P   = 0.15            # probability of zeroing one random channel


# ── Model ─────────────────────────────────────────────────────────────────────

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


class IMUEncoder(nn.Module):
    """Variable-length IMU sequence → fixed-dim feature vector.

    Accepts optional `lengths` tensor so padding positions are masked out
    of the global average pool.  When lengths=None (e.g. single unpadded
    sequence at inference) it falls back to a plain global mean.

    This module is saved separately as encoder_onhw.pt so it can be
    transplanted into an image-generation decoder for Option-2 transfer.
    """

    def __init__(self, in_ch: int = 3, hidden: int = 256, out_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 7, padding=3),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            _ResBlock1D(hidden, dil=1),
            _ResBlock1D(hidden, dil=2),
            _ResBlock1D(hidden, dil=4),
            _ResBlock1D(hidden, dil=8),
            _ResBlock1D(hidden, dil=1),
            _ResBlock1D(hidden, dil=2),
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, in_ch)
        h = self.stem(x.permute(0, 2, 1))   # (B, hidden, T)
        h = self.blocks(h)                   # (B, hidden, T)

        if lengths is not None:
            B, C, T = h.shape
            mask = (torch.arange(T, device=h.device)[None, :]
                    < lengths[:, None])          # (B, T)
            h = ((h * mask.unsqueeze(1).float()).sum(-1)
                 / lengths.float().unsqueeze(1)) # (B, hidden)
        else:
            h = h.mean(-1)                       # (B, hidden)

        return self.proj(h)                      # (B, out_dim)


class IMUClassifier(nn.Module):
    def __init__(self, n_classes: int = 52, in_ch: int = 3,
                 hidden: int = 256, feat_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = IMUEncoder(in_ch, hidden, feat_dim)
        self.drop    = nn.Dropout(dropout)
        self.head    = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.drop(self.encoder(x, lengths)))


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment_sequence(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentations to a (T, C) float32 sequence."""
    # Amplitude scale
    x = x * float(rng.uniform(*AUG_SCALE_RANGE))
    # Additive noise
    x = x + rng.normal(0, AUG_NOISE_STD, x.shape).astype(np.float32)
    # Time warp: resample to a random fraction of the original length
    factor  = float(rng.uniform(*AUG_WARP_RANGE))
    new_len = max(5, int(len(x) * factor))
    old_t   = np.linspace(0, 1, len(x))
    new_t   = np.linspace(0, 1, new_len)
    x = np.column_stack([np.interp(new_t, old_t, x[:, i])
                         for i in range(x.shape[1])]).astype(np.float32)
    # Channel dropout: zero one random axis
    if rng.random() < AUG_CHAN_DROP_P:
        x[:, int(rng.integers(0, x.shape[1]))] = 0.0
    return x


# ── Dataset ───────────────────────────────────────────────────────────────────

class OnHWDataset(Dataset):
    def __init__(self, X, y, label_map, scaler, channels, max_len,
                 augment: bool = False):
        self.X         = X
        self.y         = y
        self.label_map = label_map
        self.scaler    = scaler
        self.channels  = channels
        self.max_len   = max_len
        self.augment   = augment
        self._rng      = np.random.default_rng()   # per-worker rng

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx][:, self.channels].astype(np.float32)
        x = x[: self.max_len]
        if self.augment:
            x = augment_sequence(x, self._rng)
        x = self.scaler.transform(x).astype(np.float32)
        return torch.from_numpy(x), self.label_map[self.y[idx]], len(x)


def collate_fn(batch):
    xs, ys, lengths = zip(*batch)
    B, C = len(xs), xs[0].shape[1]
    max_len = max(lengths)
    padded = torch.zeros(B, max_len, C)
    for i, (x, l) in enumerate(zip(xs, lengths)):
        padded[i, :l] = x
    return (padded,
            torch.tensor(ys,      dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long))


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_and_clean(data_dir: Path, split: str):
    X = np.load(data_dir / f"X_{split}.npy", allow_pickle=True)
    y = np.load(data_dir / f"y_{split}.npy", allow_pickle=True)

    lengths = np.array([s.shape[0] for s in X])

    # Drop degenerate samples
    valid = lengths >= 5
    X, y, lengths = X[valid], y[valid], lengths[valid]

    # Per-class 3-sigma outlier removal
    keep = np.ones(len(X), dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        cls_lens = lengths[idx]
        mean, std = cls_lens.mean(), cls_lens.std()
        keep[idx[np.abs(cls_lens - mean) > 3 * std]] = False

    n_removed = (~keep).sum()
    X, y = X[keep], y[keep]
    print(f"  {split}: kept {len(X)} / {valid.sum()}  "
          f"({n_removed} outliers removed)")
    return X, y


def build_label_map(y_train):
    chars   = sorted(set(y_train.tolist()))
    ordered = sorted(c for c in chars if c.isupper()) + \
              sorted(c for c in chars if c.islower())
    c2i = {c: i for i, c in enumerate(ordered)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c


# ── Train / eval loops ────────────────────────────────────────────────────────

def run_epoch(model, loader, device, criterion,
              optimizer=None, amp_scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            if training:
                optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(x, lengths)
                loss   = criterion(logits, y)
            if training:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_history(t_losses, v_losses, t_accs, v_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#111")
    for ax in (ax1, ax2):
        ax.set_facecolor("#222")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ep = range(1, len(t_losses) + 1)
    ax1.plot(ep, t_losses, color="#44aaff", label="train")
    ax1.plot(ep, v_losses, color="#ff8844", label="val")
    ax1.set_title("Loss", color="white")
    ax1.set_xlabel("epoch", color="white")
    ax1.set_ylabel("CE loss", color="white")
    ax1.legend()

    ax2.plot(ep, t_accs, color="#44aaff", label="train")
    ax2.plot(ep, v_accs, color="#ff8844", label="val")
    ax2.set_title("Accuracy", color="white")
    ax2.set_xlabel("epoch", color="white")
    ax2.set_ylabel("accuracy", color="white")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "onhw_training_history.png",
                dpi=100, facecolor="#111", bbox_inches="tight")
    plt.close()


def save_confusion_matrix(model, loader, device, idx_to_char):
    from sklearn.metrics import confusion_matrix as sk_cm
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, lengths = x.to(device), lengths.to(device)
            with autocast("cuda"):
                logits = model(x, lengths)
            preds.extend(logits.argmax(1).cpu().tolist())
            trues.extend(y.tolist())

    n = len(idx_to_char)
    cm     = sk_cm(trues, preds, labels=list(range(n)))
    labels = [idx_to_char[i] for i in range(n)]

    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, color="white", fontsize=7)
    ax.set_yticklabels(labels, color="white", fontsize=7)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True",      color="white")
    ax.set_title("Confusion Matrix (val set)", color="white")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "onhw_confusion_matrix.png",
                dpi=100, facecolor="#111", bbox_inches="tight")
    plt.close()

    overall = sum(p == t for p, t in zip(preds, trues)) / len(trues)
    print(f"  Val accuracy (best model): {overall:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU  : {props.name}")
        print(f"  VRAM : {props.total_memory / 1e9:.1f} GB")
        print(f"  AMP  : enabled (mixed precision)")

    MODELS_DIR.mkdir(exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    X_train, y_train = load_and_clean(DATA_DIR, "train")
    X_val,   y_val   = load_and_clean(DATA_DIR, "test")

    char_to_idx, idx_to_char = build_label_map(y_train)
    with open(MODELS_DIR / "label_map_onhw.pkl", "wb") as f:
        pickle.dump({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}, f)

    # Fit scaler on training data only
    train_flat = np.vstack([s[:, END_ACC_CH].astype(np.float32) for s in X_train])
    scaler = StandardScaler().fit(train_flat)
    with open(MODELS_DIR / "scaler_onhw.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train_lens = np.array([s.shape[0] for s in X_train])
    max_len    = int(np.percentile(train_lens, MAX_LEN_PCTILE))
    print(f"  Sequence length 99th pct (max_len): {max_len}")
    print(f"  Classes: {len(char_to_idx)}")

    train_ds = OnHWDataset(X_train, y_train, char_to_idx, scaler, END_ACC_CH, max_len,
                           augment=True)
    val_ds   = OnHWDataset(X_val,   y_val,   char_to_idx, scaler, END_ACC_CH, max_len,
                           augment=False)

    # num_workers>0 requires the script to be run as __main__ (already true)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = IMUClassifier(N_CLASSES, N_IN_CH, HIDDEN, FEAT_DIM, DROPOUT).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: IMUClassifier   params={n_params:,}   in_ch={N_IN_CH}")

    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=EPOCHS, eta_min=1e-5)
    amp_scaler = GradScaler("cuda")
    criterion  = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    t_losses, v_losses, t_accs, v_accs = [], [], [], []

    print(f"\n {'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}"
          f"  {'Train Acc':>10}  {'Val Acc':>10}  {'Best':>8}  {'LR':>10}")
    print("-" * 76)

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_acc = run_epoch(model, train_loader, device, criterion,
                                  optimizer, amp_scaler)
        v_loss, v_acc = run_epoch(model, val_loader,   device, criterion)
        scheduler.step()

        t_losses.append(t_loss); v_losses.append(v_loss)
        t_accs.append(t_acc);   v_accs.append(v_acc)

        improved = v_acc > best_val_acc
        if improved:
            best_val_acc = v_acc
            torch.save(model.state_dict(),
                       MODELS_DIR / "best_onhw_classifier.pt")
            torch.save(model.encoder.state_dict(),
                       MODELS_DIR / "encoder_onhw.pt")

        lr     = scheduler.get_last_lr()[0]
        marker = "  *" if improved else ""
        print(f" {epoch:>6}  {t_loss:>10.4f}  {v_loss:>10.4f}"
              f"  {t_acc:>10.4f}  {v_acc:>10.4f}"
              f"  {best_val_acc:>8.4f}  {lr:>10.2e}{marker}")

        if epoch % 10 == 0 or epoch == 1:
            save_history(t_losses, v_losses, t_accs, v_accs)

    # ── Final artifacts ───────────────────────────────────────────────────────
    save_history(t_losses, v_losses, t_accs, v_accs)

    print("\nGenerating confusion matrix on best model...")
    model.load_state_dict(
        torch.load(MODELS_DIR / "best_onhw_classifier.pt",
                   map_location=device, weights_only=True))
    save_confusion_matrix(model, val_loader, device, idx_to_char)

    print(f"\nDone.")
    print(f"  Classifier : {MODELS_DIR / 'best_onhw_classifier.pt'}")
    print(f"  Encoder    : {MODELS_DIR / 'encoder_onhw.pt'}")
    print(f"  Scaler     : {MODELS_DIR / 'scaler_onhw.pkl'}")
    print(f"  Label map  : {MODELS_DIR / 'label_map_onhw.pkl'}")


if __name__ == "__main__":
    main()
