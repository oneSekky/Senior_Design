"""
predict_side_mount.py

Run the saved best_side_mount_model.pt on the full dataset and save
paginated result images to models/predictions/.

Usage:
    python Signal_Processing_Algorithm/scripts/predict_side_mount.py
    python Signal_Processing_Algorithm/scripts/predict_side_mount.py --rows 10
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Import shared code from the training script ────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from train_side_mount import (
    load_dataset,
    IMUToImage,
    MODELS_DIR,
    HIDDEN_DIM,
    SNAP_THRESHOLD,
)

ROWS_PER_PAGE = 12   # samples per output PNG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=ROWS_PER_PAGE,
                        help=f'Rows (samples) per output page (default {ROWS_PER_PAGE})')
    parser.add_argument('--max', type=int, default=None,
                        help='Max total samples to show (default: all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load model ─────────────────────────────────────────────────────────────
    model_pt = MODELS_DIR / 'best_side_mount_model.pt'
    scaler_pt = MODELS_DIR / 'scaler_side_mount.pkl'

    if not model_pt.exists():
        print(f'ERROR: no saved model at {model_pt}')
        return

    # ── Load dataset ────────────────────────────────────────────────────────────
    print('Loading dataset...')
    X, y_target, y_raw, labels = load_dataset()
    print(f'  {len(X)} samples loaded')

    # ── Scale IMU ───────────────────────────────────────────────────────────────
    if scaler_pt.exists():
        with open(scaler_pt, 'rb') as f:
            scaler = pickle.load(f)
        n, t, feat = X.shape
        X_sc = scaler.transform(X.reshape(-1, feat)).reshape(n, t, feat).astype(np.float32)
        print(f'  Scaler loaded from {scaler_pt.name}')
    else:
        from sklearn.preprocessing import StandardScaler
        print('  WARNING: no saved scaler, fitting fresh (may differ slightly from training)')
        n, t, feat = X.shape
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X.reshape(-1, feat)).reshape(n, t, feat).astype(np.float32)

    # ── Build and load model ────────────────────────────────────────────────────
    n_feat = X_sc.shape[2]
    model  = IMUToImage(n_feat).to(device)
    model.load_state_dict(torch.load(model_pt, weights_only=True, map_location=device))
    model.eval()
    print(f'  Model loaded from {model_pt.name}')

    # ── Run inference ───────────────────────────────────────────────────────────
    N = len(X_sc) if args.max is None else min(args.max, len(X_sc))
    print(f'  Running inference on {N} samples...')
    with torch.no_grad():
        X_t  = torch.from_numpy(X_sc[:N]).float().to(device)
        # run in batches to avoid OOM
        preds = []
        for i in range(0, N, 64):
            logits = model(X_t[i:i+64])
            preds.append(torch.sigmoid(logits).cpu().numpy())
        preds = np.concatenate(preds, axis=0)   # (N, H, W)

    # ── Save paginated PNGs ─────────────────────────────────────────────────────
    out_dir = MODELS_DIR / 'predictions'
    out_dir.mkdir(exist_ok=True)

    rows  = args.rows
    pages = (N + rows - 1) // rows
    print(f'  Saving {pages} page(s) to {out_dir}/')

    for page in range(pages):
        start = page * rows
        end   = min(start + rows, N)
        n_show = end - start

        fig, axes = plt.subplots(n_show, 4, figsize=(12, 2.5 * n_show))
        if n_show == 1:
            axes = axes.reshape(1, -1)

        axes[0, 0].set_title('GT (raw photo)',      fontsize=7)
        axes[0, 1].set_title('GT (stroke target)',  fontsize=7)
        axes[0, 2].set_title('Pred (raw sigmoid)',  fontsize=7)
        axes[0, 3].set_title(f'Pred >{SNAP_THRESHOLD}', fontsize=7)
        fig.suptitle(f'Page {page+1}/{pages}  —  samples {start+1}–{end}', fontsize=10)

        for pi in range(n_show):
            idx    = start + pi
            p_soft = preds[idx]
            p_hard = (p_soft >= SNAP_THRESHOLD).astype(np.float32)

            axes[pi, 0].imshow(y_raw[idx],    cmap='gray', vmin=0, vmax=1)
            axes[pi, 1].imshow(y_target[idx], cmap='gray', vmin=0, vmax=1)
            axes[pi, 2].imshow(p_soft,        cmap='hot',  vmin=0, vmax=1)
            axes[pi, 3].imshow(p_hard,        cmap='gray', vmin=0, vmax=1)
            axes[pi, 0].set_ylabel(labels[idx].split('/')[-1], fontsize=5,
                                   rotation=0, labelpad=45)
            for c in range(4):
                axes[pi, c].set_xticks([]); axes[pi, c].set_yticks([])

        plt.tight_layout()
        out_path = out_dir / f'page_{page+1:03d}.png'
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f'  Saved {out_path.name}')

    print(f'\nDone. Results in {out_dir}')


if __name__ == '__main__':
    main()
