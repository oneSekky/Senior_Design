import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
import glob

CSV_DIR = Path(r"C:\Users\sekan\Documents\Senior_Design\Test_Data\side_mount\split_csvs")

FEATURE_COLS = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]',
                'gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
ACTIVITY_THRESHOLD = 50.0
SMOOTH_WINDOW = 10
MIN_ACTIVE_SAMPLES = 20
N_TIMESTEPS = 100

b, a = signal.butter(3, 0.5, 'high', fs=104)

def get_active_len(csv_path):
    df = pd.read_csv(csv_path, skiprows=1)
    data = df[FEATURE_COLS].values.astype(np.float32)
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i])
    magnitude = np.sqrt((filtered[:, :3] ** 2).sum(axis=1))
    smoothed = np.convolve(magnitude, np.ones(SMOOTH_WINDOW)/SMOOTH_WINDOW, mode='same')
    above = smoothed > ACTIVITY_THRESHOLD
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, val in enumerate(above):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    used_fallback = best_len < MIN_ACTIVE_SAMPLES
    return best_len, used_fallback, magnitude.max(), smoothed.max()

# Gather stats across all boxes
all_lens = []
fallback_count = 0
total = 0
short_count = 0  # active segment < 40 samples (less than half N_TIMESTEPS)

print(f"{'Page':<30} {'Box':<12} {'ActiveLen':>10} {'Fallback':>9} {'MaxAcc':>9}")
for page_dir in sorted(CSV_DIR.iterdir()):
    if not page_dir.is_dir():
        continue
    for csv_path in sorted(page_dir.glob('box_*.csv')):
        alen, fb, maxacc, smaxacc = get_active_len(csv_path)
        all_lens.append(alen)
        total += 1
        if fb:
            fallback_count += 1
        if alen < 40:
            short_count += 1
            print(f"{page_dir.name:<30} {csv_path.stem:<12} {alen:>10} {str(fb):>9} {maxacc:>9.1f}")

print(f"\nTotal boxes: {total}")
print(f"Fallback (no run >= {MIN_ACTIVE_SAMPLES}): {fallback_count} ({100*fallback_count/total:.1f}%)")
print(f"Short active segment (< 40 samples): {short_count} ({100*short_count/total:.1f}%)")
print(f"\nActive segment length distribution:")
lens = np.array(all_lens)
for threshold in [20, 40, 60, 80, 100, 120, 150, 180, 208]:
    count = (lens >= threshold).sum()
    print(f"  >= {threshold:3d} samples: {count:4d} / {total} ({100*count/total:.1f}%)")
print(f"\n  mean={lens.mean():.1f}  median={np.median(lens):.1f}  std={lens.std():.1f}  min={lens.min()}  max={lens.max()}")

# Also check: for boxes where active segment > N_TIMESTEPS=100, we're truncating.
# What's the distribution of active segment length relative to 100?
over_100 = (lens > 100).sum()
print(f"\n  Segments > 100 (being truncated): {over_100} ({100*over_100/total:.1f}%)")
under_40 = (lens < 40).sum()
print(f"  Segments < 40 (mostly padding): {under_40} ({100*under_40/total:.1f}%)")

# Look at a few specific examples to understand the signal pattern better
print("\n=== Sample active segment analysis (first 5 boxes of page 1) ===")
page1 = sorted(CSV_DIR.iterdir())[0]
for csv_path in sorted(page1.glob('box_*.csv'))[:10]:
    df = pd.read_csv(csv_path, skiprows=1)
    data = df[FEATURE_COLS].values.astype(np.float32)
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i])
    magnitude = np.sqrt((filtered[:, :3] ** 2).sum(axis=1))
    smoothed = np.convolve(magnitude, np.ones(SMOOTH_WINDOW)/SMOOTH_WINDOW, mode='same')
    above = smoothed > ACTIVITY_THRESHOLD

    # find all runs
    runs = []
    cur_start, cur_len = 0, 0
    for i, val in enumerate(above):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > 0:
                runs.append((cur_start, cur_len))
            cur_len = 0
    if cur_len > 0:
        runs.append((cur_start, cur_len))

    print(f"  {csv_path.stem}: {len(runs)} runs above threshold, lengths={[r[1] for r in runs]}, max_acc={magnitude.max():.0f}")
