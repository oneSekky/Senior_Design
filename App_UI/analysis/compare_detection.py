"""
compare_detection.py — Side-by-side comparison of old fixed-threshold vs
new adaptive valley detection on Data_Log_26_05_04_11_53_38.csv.

Runs the new StrokeBuffer logic in Python (offline) across a sweep of
STROKE_END values to find the best letter-level segmentation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load data ─────────────────────────────────────────────────────────────────
CSV = os.path.join(os.path.dirname(__file__), '..', 'data_logs', 'Data_Log_26_05_04_11_53_38.csv')
df  = pd.read_csv(CSV, comment='#', skiprows=1)
df.columns = df.columns.str.strip()
acc  = df[['acc_x[mg]','acc_y[mg]','acc_z[mg]']].values.astype(np.float32)
gyro = df[['gyro_x[mdps]','gyro_y[mdps]','gyro_z[mdps]']].values.astype(np.float32)
samples = np.hstack([acc, gyro])   # (N, 6)
N = len(samples)
FS = 104

# ── Shared detection signal ───────────────────────────────────────────────────
deltas  = np.diff(acc, axis=0)
raw_mag = np.concatenate([[0.0], np.sqrt((deltas**2).sum(axis=1))])
SMOOTH_WIN = 10
kernel   = np.ones(SMOOTH_WIN) / SMOOTH_WIN
smoothed = np.convolve(raw_mag, kernel, mode='same')

# Write-start detection (calibration flat baseline protocol)
FLAT_THRESHOLD = 5.0
FLAT_DURATION  = 130
PICKUP_SETTLE  = 52
flat_count = 0
baseline_end = -1
for i, v in enumerate(smoothed):
    if v < FLAT_THRESHOLD:
        flat_count += 1
        if flat_count >= FLAT_DURATION:
            baseline_end = i; break
    else:
        flat_count = 0

pickup_idx = baseline_end
for i in range(baseline_end, N):
    if smoothed[i] > FLAT_THRESHOLD:
        pickup_idx = i; break

write_start = pickup_idx + PICKUP_SETTLE
print(f"Write start: sample {write_start}  ({write_start/FS:.1f}s)")

sm_write = smoothed[write_start:]
samp_write = samples[write_start:]
t_write  = np.arange(len(sm_write)) / FS

# ── Helper: run fixed-threshold detector ────────────────────────────────────
def run_fixed(sm, thr, stroke_end, word_gap=83):
    strokes, gaps = [], []
    in_s, qs, ss = False, 0, 0
    for i, v in enumerate(sm):
        if v > thr:
            if not in_s: in_s, ss = True, i
            qs = 0
        else:
            if in_s:
                qs += 1
                if qs >= stroke_end:
                    strokes.append((ss, i - qs))
                    in_s, qs = False, 0
    if in_s: strokes.append((ss, len(sm)-1))
    for i in range(1, len(strokes)):
        gaps.append(strokes[i][0] - strokes[i-1][1])
    return strokes, gaps

# ── Helper: run new valley detector ─────────────────────────────────────────
def run_valley(samp, stroke_end, word_gap=83,
               valley_frac=0.40, peak_win=52, smooth_win=10,
               min_abs=8.0, warmup=20, min_stroke=20):
    from collections import deque
    mag_win  = deque(maxlen=smooth_win)
    peak_win_d = deque(maxlen=peak_win)
    prev_acc = None
    active_buf, valley_buf = [], []
    quiet_count = 0
    is_active = False
    warmup_left = warmup
    strokes, gaps = [], []
    stroke_start_idx = 0

    for i, s in enumerate(samp):
        s = s.astype(np.float32)
        if prev_acc is None:
            prev_acc = s[:3].copy(); warmup_left -= 1; continue
        if warmup_left > 0:
            warmup_left -= 1; prev_acc = s[:3].copy(); continue

        delta = s[:3] - prev_acc
        prev_acc = s[:3].copy()
        rm = float(np.sqrt((delta**2).sum()))
        mag_win.append(rm)
        sm_v = float(np.mean(mag_win))

        if sm_v >= min_abs:
            peak_win_d.append(sm_v)
        rp   = float(max(peak_win_d)) if peak_win_d else 0.0
        thr  = max(min_abs, valley_frac * rp)
        act  = sm_v >= thr

        if is_active:
            if act:
                if valley_buf:
                    active_buf.extend(valley_buf); valley_buf = []
                active_buf.append(s)
                quiet_count = 0
            else:
                valley_buf.append(s)
                quiet_count += 1
                if quiet_count >= stroke_end:
                    if len(active_buf) >= min_stroke:
                        strokes.append((stroke_start_idx, i - quiet_count))
                    active_buf = []; valley_buf = []; quiet_count = 0
                    is_active = False
        else:
            if act:
                is_active = True; quiet_count = 0; valley_buf = []
                active_buf = [s]; stroke_start_idx = i

    if is_active and len(active_buf) >= min_stroke:
        strokes.append((stroke_start_idx, len(samp)-1))

    for i in range(1, len(strokes)):
        gaps.append(strokes[i][0] - strokes[i-1][1])
    return strokes, gaps

# ── Sweep ─────────────────────────────────────────────────────────────────────
stroke_ends  = [5, 8, 12, 15, 20, 25, 30, 40, 50]
fixed_thrs   = [9.2, 12.0, 15.0]
valley_fracs = [0.30, 0.40, 0.50]

print("\n-- Fixed threshold (old approach) ------------------")
print(f"{'thr':>6}  {'se':>4}  {'strokes':>8}  {'min_gap':>8}  {'med_gap':>8}")
for thr in fixed_thrs:
    for se in [15, 20, 30, 50]:
        st, gp = run_fixed(sm_write, thr, se)
        mg = min(gp) if gp else 0
        md = int(np.median(gp)) if gp else 0
        print(f"{thr:6.1f}  {se:4d}  {len(st):8d}  {mg:8d}  {md:8d}")

print("\n-- Valley detection (new approach) -----------------")
print(f"{'frac':>6}  {'se':>4}  {'strokes':>8}  {'min_gap':>8}  {'med_gap':>8}")
for frac in valley_fracs:
    for se in stroke_ends:
        st, gp = run_valley(samp_write, se, valley_frac=frac)
        mg = min(gp) if gp else 0
        md = int(np.median(gp)) if gp else 0
        print(f"{frac:6.2f}  {se:4d}  {len(st):8d}  {mg:8d}  {md:8d}")

# ── Best config for plot ───────────────────────────────────────────────────────
# Pick the valley config that gets closest to expected ~55 strokes
best_frac, best_se = 0.50, 8
st_valley, gp_valley = run_valley(samp_write, best_se, valley_frac=best_frac)
st_fixed,  gp_fixed  = run_fixed(sm_write, 9.2, 30)

print(f"\nPlotting: valley frac={best_frac} se={best_se} -> {len(st_valley)} strokes")
print(f"Plotting: fixed thr=9.2 se=30 -> {len(st_fixed)} strokes")

# ── Build adaptive threshold signal for plotting ──────────────────────────────
from collections import deque as _deque
peak_win_d = _deque(maxlen=52)
mag_win_d  = _deque(maxlen=10)
prev = None
adap_thr  = np.zeros(len(sm_write))
adap_sm   = np.zeros(len(sm_write))
for i, s in enumerate(samp_write):
    s = s.astype(np.float32)
    if prev is None: prev = s[:3].copy(); continue
    delta = s[:3] - prev; prev = s[:3].copy()
    rm = float(np.sqrt((delta**2).sum()))
    mag_win_d.append(rm)
    sv = float(np.mean(mag_win_d))
    if sv >= 8.0: peak_win_d.append(sv)
    rp = float(max(peak_win_d)) if peak_win_d else 0.0
    adap_thr[i] = max(8.0, 0.40 * rp)
    adap_sm[i]  = sv

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(24, 12), sharex=True)
fig.patch.set_facecolor('#111')

def shade(ax, spans, color, alpha=0.22):
    for s, e in spans:
        ax.axvspan(s/FS, e/FS, color=color, alpha=alpha)

def style(ax, title):
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#aaa', labelsize=8)
    ax.set_title(title, color='#ddd', fontsize=9, loc='left', pad=3)
    for sp in ax.spines.values(): sp.set_edgecolor('#444')
    ax.set_xlim(0, t_write[-1])

# Row 0: OLD fixed threshold
ax = axes[0]
ax.plot(t_write, sm_write, color='#44aaff', lw=0.6, label='smoothed acc-delta')
ax.axhline(9.2,  color='#ff5555', lw=1.0, ls='--', label='thr=9.2')
ax.axhline(15.0, color='#ff9933', lw=0.7, ls=':', label='thr=15.0')
shade(ax, st_fixed, '#2244aa')
wg_f = [(st_fixed[i][1], st_fixed[i+1][0]) for i in range(len(st_fixed)-1) if gp_fixed[i]>=83]
shade(ax, wg_f, '#aa4422', alpha=0.18)
ax.set_ylabel('mg/samp', color='#888', fontsize=7)
ax.legend(fontsize=7, facecolor='#222', labelcolor='#ccc', loc='upper right')
style(ax, f'OLD: Fixed threshold (thr=9.2, stroke_end=30)  →  {len(st_fixed)} strokes  |  expected ~55')

# Row 1: NEW valley detection
ax = axes[1]
ax.plot(t_write, adap_sm,  color='#44aaff', lw=0.6, label='smoothed acc-delta')
ax.plot(t_write, adap_thr, color='#55ff55', lw=0.9, ls='--', label=f'adaptive thr (40% × peak)')
ax.fill_between(t_write, adap_thr, alpha=0.07, color='#55ff55')
shade(ax, st_valley, '#2244aa')
wg_v = [(st_valley[i][1], st_valley[i+1][0]) for i in range(len(st_valley)-1) if gp_valley[i]>=83]
shade(ax, wg_v, '#aa4422', alpha=0.18)
ax.set_ylabel('mg/samp', color='#888', fontsize=7)
ax.legend(fontsize=7, facecolor='#222', labelcolor='#ccc', loc='upper right')
style(ax, f'NEW: Valley detection (frac={best_frac}, stroke_end={best_se})  →  {len(st_valley)} strokes  |  expected ~55')

# Row 2: Zoomed first 15s — letter-level detail
ax = axes[2]
end15 = int(15 * FS)
t15   = t_write[:end15]
ax.plot(t15, adap_sm[:end15],  color='#44aaff', lw=0.8)
ax.plot(t15, adap_thr[:end15], color='#55ff55', lw=1.0, ls='--', label='adaptive thr')
shade(ax, [(s, e) for s, e in st_valley if s < end15], '#2244aa')
shade(ax, [(s, e) for s, e in st_fixed  if s < end15], '#ff4444', alpha=0.12)
ax.set_ylabel('mg/samp', color='#888', fontsize=7)
ax.set_xlabel('Time since write start (s)', color='#aaa', fontsize=9)
ax.legend(fontsize=7, facecolor='#222', labelcolor='#ccc', loc='upper right')
style(ax, f'ZOOM first 15s — blue=valley strokes  red=old fixed strokes')
ax.set_xlim(0, 15)

patch_s = mpatches.Patch(color='#2244aa', alpha=0.5, label='stroke (active)')
patch_w = mpatches.Patch(color='#aa4422', alpha=0.4, label='word gap')
fig.legend(handles=[patch_s, patch_w], loc='upper center', ncol=2,
           facecolor='#222', labelcolor='#ccc', fontsize=8)

plt.suptitle(
    'OLD fixed-threshold vs NEW valley detection — same data file\n'
    '"jumping to Brazil" + "Sekander I don\'t think this is working very well"  ~55 expected strokes',
    color='white', fontsize=10
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(__file__), 'output', 'detection_comparison.png')
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#111')
print(f"\nSaved {out}")
