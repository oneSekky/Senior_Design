import pandas as pd
import numpy as np
from scipy import signal

CSV_PATH = r"C:\Users\sekan\Documents\Senior_Design\Test_Data\side_mount\csvs\box-page-1-2sec.csv"

with open(CSV_PATH) as f:
    lines = f.readlines()
header_idx = next(i for i, l in enumerate(lines) if 'acc' in l.lower())
df = pd.read_csv(CSV_PATH, skiprows=header_idx)

b, a = signal.butter(3, 0.5, 'high', fs=104)
acc_cols = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']
filtered = np.zeros((len(df), 3), dtype=np.float32)
for i, col in enumerate(acc_cols):
    filtered[:, i] = signal.filtfilt(b, a, df[col].values.astype(np.float32))

magnitude = np.sqrt((filtered**2).sum(axis=1))

gyro_cols = ['gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
gyro = df[gyro_cols].values.astype(np.float32)
gyro_mag = np.sqrt((gyro**2).sum(axis=1))

SAMPLES_PER_BOX = 208
COLS = 10
ROWS = 14

# For each col=0 box (start of a new row), look at where in the 208 samples
# the activity is high vs low. This tells us where the newline movement is vs actual writing.
print("=== Per-box activity distribution: col=0 boxes (start of new row) ===")
print("Showing acc_mag in 4 quartiles of each 208-sample box")
for row in range(ROWS):
    box_idx = row * COLS + 0
    start = box_idx * SAMPLES_PER_BOX
    end = start + SAMPLES_PER_BOX
    if end > len(df):
        break
    chunk = magnitude[start:end]
    q1 = chunk[:52]
    q2 = chunk[52:104]
    q3 = chunk[104:156]
    q4 = chunk[156:208]
    print(f"  Row {row:02d} col=0: Q1={q1.mean():.0f}  Q2={q2.mean():.0f}  Q3={q3.mean():.0f}  Q4={q4.mean():.0f}  | max={chunk.max():.0f} at t={chunk.argmax()}")

print("\n=== Per-box activity distribution: col=9 boxes (end of row, transition coming) ===")
for row in range(ROWS):
    box_idx = row * COLS + 9
    start = box_idx * SAMPLES_PER_BOX
    end = start + SAMPLES_PER_BOX
    if end > len(df):
        break
    chunk = magnitude[start:end]
    q1 = chunk[:52]
    q2 = chunk[52:104]
    q3 = chunk[104:156]
    q4 = chunk[156:208]
    print(f"  Row {row:02d} col=9: Q1={q1.mean():.0f}  Q2={q2.mean():.0f}  Q3={q3.mean():.0f}  Q4={q4.mean():.0f}  | max={chunk.max():.0f} at t={chunk.argmax()}")

print("\n=== Per-box activity distribution: col=4 (mid-row) boxes ===")
for row in range(ROWS):
    box_idx = row * COLS + 4
    start = box_idx * SAMPLES_PER_BOX
    end = start + SAMPLES_PER_BOX
    if end > len(df):
        break
    chunk = magnitude[start:end]
    q1 = chunk[:52]
    q2 = chunk[52:104]
    q3 = chunk[104:156]
    q4 = chunk[156:208]
    print(f"  Row {row:02d} col=4: Q1={q1.mean():.0f}  Q2={q2.mean():.0f}  Q3={q3.mean():.0f}  Q4={q4.mean():.0f}  | max={chunk.max():.0f} at t={chunk.argmax()}")

# Find where the "quiet" (not-writing) period is within each box
# by using a rolling window to find the minimum-activity segment
print("\n=== Finding quiet period location within each box ===")
WINDOW = 30  # ~0.3 sec window
print(f"{'Box':>5} {'Row':>4} {'Col':>4} {'QuietStart':>12} {'QuietMean':>12} {'ActiveMean':>12} {'Ratio':>8}")
for box_idx in range(ROWS * COLS):
    row = box_idx // COLS
    col = box_idx % COLS
    start = box_idx * SAMPLES_PER_BOX
    end = start + SAMPLES_PER_BOX
    if end > len(df):
        break
    chunk = magnitude[start:end]
    # Find quietest 30-sample window
    min_mean = float('inf')
    min_pos = 0
    for w in range(0, SAMPLES_PER_BOX - WINDOW):
        m = chunk[w:w+WINDOW].mean()
        if m < min_mean:
            min_mean = m
            min_pos = w
    active_mean = chunk.mean()
    ratio = active_mean / max(min_mean, 1)
    print(f"{box_idx:>5} {row:>4} {col:>4} {min_pos:>12} {min_mean:>12.1f} {active_mean:>12.1f} {ratio:>8.2f}")
