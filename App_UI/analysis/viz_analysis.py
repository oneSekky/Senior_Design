import numpy as np
import pandas as pd
from scipy import signal as sp_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

# Load
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'data_logs', 'Data_Log_26_05_04_11_53_38.csv'),
                 comment='#', skiprows=1)
df.columns = df.columns.str.strip()
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

acc  = df[['acc_x[mg]','acc_y[mg]','acc_z[mg]']].values.astype(np.float32)
gyro = df[['gyro_x[mdps]','gyro_y[mdps]','gyro_z[mdps]']].values.astype(np.float32)
raw  = np.hstack([acc, gyro])
N    = len(raw)
FS   = 104

# Detection signal (acc-delta)
deltas  = np.diff(acc, axis=0)
det_mag = np.concatenate([[0.0], np.sqrt((deltas**2).sum(axis=1))])
SMOOTH_WIN = 5
kernel   = np.ones(SMOOTH_WIN) / SMOOTH_WIN
smoothed = np.convolve(det_mag, kernel, mode='same')

# Find calibration flat baseline end
FLAT_THRESHOLD = 5.0
FLAT_DURATION  = 130
PICKUP_SETTLE  = 52

flat_count   = 0
baseline_end = -1
for i, v in enumerate(smoothed):
    if v < FLAT_THRESHOLD:
        flat_count += 1
        if flat_count >= FLAT_DURATION:
            baseline_end = i
            break
    else:
        flat_count = 0

print(f"Baseline period ends at sample {baseline_end} ({baseline_end/FS:.2f}s)")

pickup_idx = baseline_end
for i in range(baseline_end, N):
    if smoothed[i] > FLAT_THRESHOLD:
        pickup_idx = i
        break
print(f"Pen pickup at sample {pickup_idx} ({pickup_idx/FS:.2f}s)")

write_start = pickup_idx + PICKUP_SETTLE
print(f"Writing starts at sample {write_start} ({write_start/FS:.2f}s)")

# Stroke detector on writing portion
THRESHOLD  = 18.4   # baseline_p99(~2.3) * 8.0
STROKE_END = 30
WORD_GAP   = 83

sm_write  = smoothed[write_start:]
strokes   = []
in_stroke = False
stroke_start = 0
quiet_count  = 0

for i, v in enumerate(sm_write):
    abs_i = i + write_start
    if v > THRESHOLD:
        if not in_stroke:
            in_stroke    = True
            stroke_start = abs_i
        quiet_count = 0
    else:
        if in_stroke:
            quiet_count += 1
            if quiet_count >= STROKE_END:
                strokes.append((stroke_start, abs_i - quiet_count))
                in_stroke   = False
                quiet_count = 0
if in_stroke:
    strokes.append((stroke_start, write_start + len(sm_write) - 1))

# Sweep thresholds for comparison
print("\nThreshold sweep (stroke_end=30):")
for thr_try in [9.2, 12.0, 15.0, 18.4, 22.0, 26.0]:
    s2, in2, q2, sc2 = [], False, 0, 0
    for i, v in enumerate(sm_write):
        if v > thr_try:
            if not in2: in2, sc2 = True, i + write_start
            q2 = 0
        else:
            if in2:
                q2 += 1
                if q2 >= STROKE_END:
                    s2.append((sc2, i + write_start - q2)); in2 = False; q2 = 0
    if in2: s2.append((sc2, write_start + len(sm_write) - 1))
    print(f"  thr={thr_try:5.1f} -> {len(s2):3d} strokes")

print(f"\nDetected {len(strokes)} strokes (thr={THRESHOLD}, stroke_end={STROKE_END})")

gaps = []
for i in range(1, len(strokes)):
    gaps.append(strokes[i][0] - strokes[i-1][1])

if gaps:
    print("Gap sizes (samples, desc):", sorted(gaps, reverse=True)[:25])
    print(f"  min={min(gaps)}  median={np.median(gaps):.0f}  max={max(gaps)}")

word_gap_indices = [i for i, g in enumerate(gaps) if g >= WORD_GAP]
print(f"Word gaps at gap indices: {word_gap_indices}  (>={WORD_GAP} samp)")

# Build 11 features for writing portion
data = raw[write_start:].copy().astype(np.float32)
M = len(data)

b_grav,  a_grav  = sp_signal.butter(3, 0.5, 'high', fs=FS)
b_drift, a_drift = sp_signal.butter(2, 0.3, 'high', fs=FS)

filt = np.zeros_like(data)
for i in range(6):
    filt[:, i] = sp_signal.filtfilt(b_grav, a_grav, data[:, i])

mag_f = np.sqrt((filt[:, :3]**2).sum(axis=1))

dt  = 1.0 / FS
vel = np.cumsum(filt[:, :2] * dt, axis=0)
if M > 9:
    for i in range(2):
        vel[:, i] = sp_signal.filtfilt(b_drift, a_drift, vel[:, i])

pos = np.cumsum(vel * dt, axis=0)
if M > 9:
    for i in range(2):
        pos[:, i] = sp_signal.filtfilt(b_drift, a_drift, pos[:, i])

feat = np.hstack([filt, mag_f[:, None], vel, pos])

feat_names = [
    'acc_x (HP filt)', 'acc_y (HP filt)', 'acc_z (HP filt)',
    'gyr_x (HP filt)', 'gyr_y (HP filt)', 'gyr_z (HP filt)',
    'acc magnitude', 'vel_x', 'vel_y', 'pos_x', 'pos_y'
]

# Time axes
t_write = np.arange(M) / FS

def abs_to_t(s):
    return (s - write_start) / FS

stroke_spans = [(abs_to_t(s), abs_to_t(e)) for s, e in strokes]
wg_spans = [(abs_to_t(strokes[i][1]), abs_to_t(strokes[i+1][0]))
            for i in range(len(strokes)-1) if gaps[i] >= WORD_GAP]

def shade_strokes(ax):
    for ts, te in stroke_spans:
        ax.axvspan(ts, te, color='#2244aa', alpha=0.22)
    for ts, te in wg_spans:
        ax.axvspan(ts, te, color='#aa4422', alpha=0.18)

def style_ax(ax, title):
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#aaa', labelsize=7)
    ax.set_title(title, color='#ddd', fontsize=8, loc='left', pad=2)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.set_xlim(0, t_write[-1])

n_rows = 2 + 11
fig, axes = plt.subplots(n_rows, 1, figsize=(24, 2.0 * n_rows))
fig.patch.set_facecolor('#111')

# Row 0: detection signal
ax = axes[0]
ax.plot(t_write, smoothed[write_start:], color='#44aaff', lw=0.6, label='acc-delta smoothed')
ax.axhline(THRESHOLD, color='#ff5555', lw=1.0, ls='--', label=f'threshold={THRESHOLD}')
ax.axhline(FLAT_THRESHOLD, color='#ffaa22', lw=0.7, ls=':', label=f'flat={FLAT_THRESHOLD}')
shade_strokes(ax)
ax.legend(fontsize=7, facecolor='#222', labelcolor='#ccc', loc='upper right')
ax.set_ylabel('mg/samp', color='#888', fontsize=7)
style_ax(ax, f'Acc-Delta Detection  |  strokes detected={len(strokes)}  word gaps={len(wg_spans)}')

# Row 1: 2D position XY trajectory
ax = axes[1]
ax.set_facecolor('#1a1a1a')
ax.plot(pos[:, 0], pos[:, 1], color='#333', lw=0.4)
cmap = plt.cm.plasma
for k, (s, e) in enumerate(strokes):
    sl = slice(s - write_start, e - write_start + 1)
    c  = cmap(k / max(len(strokes) - 1, 1))
    ax.plot(pos[sl, 0], pos[sl, 1], lw=1.1, color=c)
ax.set_title('2D Position Trajectory (pos_x vs pos_y) — strokes colored in time order (purple=early, yellow=late)',
             color='#ddd', fontsize=8, loc='left', pad=2)
ax.tick_params(colors='#aaa', labelsize=7)
ax.set_aspect('equal', adjustable='datalim')
for sp in ax.spines.values(): sp.set_edgecolor('#333')
ax.set_xlabel('pos_x (m)', color='#888', fontsize=7)
ax.set_ylabel('pos_y (m)', color='#888', fontsize=7)

# Rows 2-12: 11 features
colors = ['#ff6666','#66ff66','#6699ff','#ffaa44','#aa66ff','#44ffdd',
          '#ffff55','#ff44aa','#44aaff','#aaffaa','#ffcc88']
for fi in range(11):
    ax = axes[2 + fi]
    ax.plot(t_write, feat[:, fi], color=colors[fi], lw=0.55)
    shade_strokes(ax)
    ax.set_ylabel(feat_names[fi].split(' ')[0], color='#888', fontsize=6)
    style_ax(ax, feat_names[fi])

axes[-1].set_xlabel('Time since writing start (s)', color='#aaa', fontsize=9)

patch_s = mpatches.Patch(color='#2244aa', alpha=0.5, label='stroke (active)')
patch_w = mpatches.Patch(color='#aa4422', alpha=0.4, label='word gap')
fig.legend(handles=[patch_s, patch_w], loc='upper right',
           facecolor='#222', labelcolor='#ccc', fontsize=8, bbox_to_anchor=(1.0, 0.99))

plt.suptitle(
    f'IMU Writing Analysis  |  thr={THRESHOLD} (baseline_p99*8)  stroke_end={STROKE_END} samp  word_gap={WORD_GAP} samp  '
    f'detected={len(strokes)} strokes  {len(wg_spans)} word gaps',
    color='white', fontsize=10
)
plt.tight_layout(rect=[0, 0, 1, 0.998])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'writing_analysis.png')
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#111')
print(f"\nSaved {out}")
