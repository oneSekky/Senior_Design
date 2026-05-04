"""
compare_detectors.py  —  Stroke-detection algorithm comparison
===============================================================
CSV: Data_Log_26_05_04_10_35_15.csv
Expected: 15 strokes (8-letter word, 2-letter word, 5-letter word), 2 word gaps

Key finding from first run:
  - All algorithms merge letters within a word into one long stroke because
    inter-letter pauses are shorter than STROKE_END_SAMP=500ms.
  - HP-acc magnitude correctly finds both word gaps (2/2).
  - Gyro-delta p10 during writing < threshold → fails for wrist-rigid writers.

This script:
  1. Computes 5 detection signals and auto-thresholds them.
  2. Runs the state machine for each.
  3. Sweeps STROKE_END_SAMP (10–52) to find what inter-letter gap this data has.
  4. Produces a comparison plot + sweep plot.

State machine (identical to stroke_buffer.py)
  active    : smoothed > thr
  inactive  : smoothed <= thr
  stroke_end: STROKE_END_SAMP consecutive inactive after active  -> stroke_complete
  word_gap  : WORD_GAP_SAMP  consecutive inactive (post stroke)  -> word_gap
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, filtfilt

# ── Constants ────────────────────────────────────────────────────────────────
FS              = 104
SMOOTH_WINDOW   = 10
MIN_STROKE_SAMP = 10        # shorter minimum for the sweep (was 20)
STROKE_END_SAMP = 52        # 500 ms — swept below
WORD_GAP_SAMP   = 83        # 800 ms — fixed
REST_S          = 3.0
THRESH_FACTOR   = 4.0

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_logs', 'Data_Log_26_05_04_10_35_15.csv')
EXPECTED_STROKES   = 15
EXPECTED_WORD_GAPS = 2

# ── Load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV, comment="#", header=0)
df.columns = df.columns.str.strip()
t_us = df["time[us]"].values.astype(np.float64)
t_s  = (t_us - t_us[0]) / 1e6
N    = len(t_s)
ax_r = df["acc_x[mg]"].values.astype(np.float32)
ay_r = df["acc_y[mg]"].values.astype(np.float32)
az_r = df["acc_z[mg]"].values.astype(np.float32)
gx_r = df["gyro_x[mdps]"].values.astype(np.float32)
gy_r = df["gyro_y[mdps]"].values.astype(np.float32)
gz_r = df["gyro_z[mdps]"].values.astype(np.float32)
dt   = 1.0 / FS
rest_n = int(REST_S * FS)

print(f"Loaded {N} samples  ({t_s[-1]:.1f} s)  rest_window={REST_S}s")

# ── Helpers ───────────────────────────────────────────────────────────────────
def smooth(x):
    return np.convolve(x, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW, mode="same")

def auto_thr(signal, factor=THRESH_FACTOR):
    return float(np.percentile(smooth(signal)[:rest_n], 99)) * factor

def run_detector(signal, threshold, stroke_end=STROKE_END_SAMP,
                 word_gap=WORD_GAP_SAMP, min_samp=MIN_STROKE_SAMP):
    sm           = smooth(signal)
    is_active    = False
    inactive_cnt = 0
    gap_fired    = False
    buf_start    = 0
    strokes      = []
    gaps         = []
    for i in range(N):
        if sm[i] > threshold:
            if not is_active:
                buf_start = i
            is_active    = True
            inactive_cnt = 0
            gap_fired    = False
        else:
            if is_active:
                inactive_cnt += 1
                if inactive_cnt >= stroke_end:
                    end_i = i - stroke_end
                    if (end_i - buf_start) >= min_samp:
                        strokes.append((buf_start, end_i))
                    is_active    = False
                    inactive_cnt = 0
            else:
                if not gap_fired:
                    inactive_cnt += 1
                    if inactive_cnt >= word_gap:
                        gaps.append(i)
                        gap_fired = True
    return sm, strokes, gaps

# ── BUILD SIGNALS ─────────────────────────────────────────────────────────────

# Alg 1: HP-filtered acc magnitude (filtfilt offline = best-case zero-phase)
b_hp, a_hp = butter(3, 0.5, "high", fs=FS)
fax = filtfilt(b_hp, a_hp, ax_r)
fay = filtfilt(b_hp, a_hp, ay_r)
faz = filtfilt(b_hp, a_hp, az_r)
sig1 = np.sqrt(fax**2 + fay**2 + faz**2).astype(np.float32)

# Alg 2: Frame-to-frame acc delta (jerk of acc, no filter needed)
sig2 = np.sqrt(np.diff(ax_r, prepend=ax_r[0])**2 +
               np.diff(ay_r, prepend=ay_r[0])**2 +
               np.diff(az_r, prepend=az_r[0])**2).astype(np.float32)

# Alg 3: Frame-to-frame gyro delta (cancels ~6700 mdps DC bias)
sig3 = np.sqrt(np.diff(gx_r, prepend=gx_r[0])**2 +
               np.diff(gy_r, prepend=gy_r[0])**2 +
               np.diff(gz_r, prepend=gz_r[0])**2).astype(np.float32)

# Alg 4: Integrated XY velocity (HP-acc -> cumsum -> drift-HP -> mag)
b_dr, a_dr = butter(2, 0.3, "high", fs=FS)
vx = filtfilt(b_dr, a_dr, np.cumsum(fax * dt))
vy = filtfilt(b_dr, a_dr, np.cumsum(fay * dt))
sig4 = np.sqrt(vx**2 + vy**2).astype(np.float32)

# Alg 5: OR-fusion of normalised {HP-acc, acc-delta, velocity}
#   Each signal / rest_p99 -> normalised. Gyro excluded (rigid-wrist issue).
def rp99(s): return max(float(np.percentile(smooth(s)[:rest_n], 99)), 1e-6)
sig5 = np.maximum(np.maximum(sig1/rp99(sig1), sig2/rp99(sig2)),
                  sig4/rp99(sig4)).astype(np.float32)

signals = [
    (sig1, "1. HP-acc magnitude",    "mg",          "#4488ff"),
    (sig2, "2. Acc delta (jerk)",     "mg/samp",     "#ff8844"),
    (sig3, "3. Gyro delta",           "mdps/samp",   "#cc44cc"),
    (sig4, "4. XY velocity",          "mg*s",        "#44cc88"),
    (sig5, "5. Combined OR-fusion",   "norm",        "#ffcc00"),
]

# ── AUTO-THRESHOLD + RUN AT DEFAULT STROKE_END=52 ────────────────────────────
results = []
for sig, name, unit, col in signals:
    thr = auto_thr(sig)
    sm, strokes, gaps = run_detector(sig, thr)
    results.append((sig, thr, sm, strokes, gaps, name, unit, col))

# ── PRINT NOISE FLOOR / THRESHOLD DIAGNOSTICS ────────────────────────────────
print("\n--- Signal diagnostics (smoothed) ---")
print(f"  {'Algorithm':<30} {'rest_p99':>10} {'thr':>10} {'wrt_p10':>10} {'wrt_p50':>10}")
for sig, thr, sm, strokes, gaps, name, unit, col in results:
    sm_rest  = smooth(sig)[:rest_n]
    sm_write = smooth(sig)[rest_n:]
    print(f"  {name:<30} {np.percentile(sm_rest,99):>10.2f} {thr:>10.2f} "
          f"{np.percentile(sm_write,10):>10.2f} {np.percentile(sm_write,50):>10.2f}")

# ── SUMMARY TABLE AT DEFAULT STROKE_END=52 ───────────────────────────────────
print("\n" + "="*72)
print(f"Results at STROKE_END_SAMP={STROKE_END_SAMP} ({STROKE_END_SAMP/FS:.2f}s)")
print(f"  {'Algorithm':<30} {'Strokes':>8} {'Gaps':>6}  vs expected")
print("-"*72)
for sig, thr, sm, strokes, gaps, name, unit, col in results:
    ds = len(strokes) - EXPECTED_STROKES
    dw = len(gaps)    - EXPECTED_WORD_GAPS
    print(f"  {name:<30} {len(strokes):>8} {len(gaps):>6}  strokes {ds:+d}  gaps {dw:+d}")
print(f"  {'EXPECTED':<30} {EXPECTED_STROKES:>8} {EXPECTED_WORD_GAPS:>6}")
print("="*72)

# ── GAP ANALYSIS ─────────────────────────────────────────────────────────────
print(f"\nLongest below-threshold gap IN WRITING WINDOW "
      f"(need >{STROKE_END_SAMP/FS:.2f}s to end stroke):")
for sig, thr, sm, strokes, gaps, name, unit, col in results:
    sm_w = smooth(sig)[rest_n:]
    below = sm_w < thr
    max_gap = cur = 0
    for b in below:
        cur = cur+1 if b else 0
        max_gap = max(max_gap, cur)
    # Distribution of gap lengths
    run_lens = []
    cur = 0
    for b in below:
        if b: cur += 1
        else:
            if cur: run_lens.append(cur)
            cur = 0
    if cur: run_lens.append(cur)
    long_enough = sum(1 for r in run_lens if r >= STROKE_END_SAMP)
    print(f"  {name:<30}  max={max_gap/FS:.2f}s  "
          f"gaps>={STROKE_END_SAMP/FS:.2f}s: {long_enough}  "
          f"pct_below={below.mean()*100:.0f}%")

# ── STROKE_END SWEEP — find which value gives 15 strokes ─────────────────────
print(f"\n--- STROKE_END_SAMP sweep (WORD_GAP_SAMP fixed={WORD_GAP_SAMP}) ---")
sweep_vals = list(range(5, 55, 5))
print(f"  {'samp':>6} {'ms':>6}  " +
      "  ".join(f"{r[5][:12]:>12}" for r in results))
print("-" * (14 + 14 * len(results)))

sweep_table = {name: [] for _, _, _, _, _, name, _, _ in results}
for sv in sweep_vals:
    row = f"  {sv:>4}  {sv*1000//FS:>4}ms  "
    for sig, thr, sm, strokes, gaps, name, unit, col in results:
        _, s, _ = run_detector(sig, thr, stroke_end=sv)
        sweep_table[name].append(len(s))
        row += f"{len(s):>12}  "
    print(row)

print(f"\n  (target = {EXPECTED_STROKES} strokes)")

# Find best STROKE_END for each algorithm
print("\n  Best STROKE_END_SAMP per algorithm:")
for sig, thr, sm, strokes_def, gaps, name, unit, col in results:
    counts = sweep_table[name]
    # find value closest to EXPECTED_STROKES (prefer higher samp for stability)
    diffs = [abs(c - EXPECTED_STROKES) for c in counts]
    best_idx = int(np.argmin(diffs))
    best_sv  = sweep_vals[best_idx]
    best_cnt = counts[best_idx]
    print(f"    {name:<30}  best_samp={best_sv} ({best_sv*1000//FS}ms) "
          f"-> {best_cnt} strokes")

# ── PLOT 1: signal comparison at default STROKE_END ──────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(16, 17), sharex=True, facecolor="#1a1a1a")
fig.suptitle(
    f"Stroke Detection Comparison  |  {CSV.split(chr(92))[-1]}\n"
    f"STROKE_END={STROKE_END_SAMP} ({STROKE_END_SAMP/FS:.2f}s)  |  "
    f"Expected: {EXPECTED_STROKES} strokes, {EXPECTED_WORD_GAPS} gaps  |  "
    f"Green=stroke  Red=word-gap  White-dashed=thr",
    fontsize=9, color="white", fontweight="bold"
)
for ax_p, (sig, thr, sm, strokes, gaps, name, unit, col) in zip(axes, results):
    ax_p.set_facecolor("#111")
    ax_p.plot(t_s, sm, color=col, lw=0.8, alpha=0.9)
    ax_p.axhline(thr, color="white", lw=0.9, ls="--", alpha=0.6)
    for s, e in strokes:
        ax_p.axvspan(t_s[s], t_s[min(e, N-1)], alpha=0.22, color="lime", zorder=0)
    for g in gaps:
        ax_p.axvline(t_s[g], color="red", lw=1.2, ls=":", alpha=0.85)
    ds = len(strokes) - EXPECTED_STROKES
    dw = len(gaps)    - EXPECTED_WORD_GAPS
    ax_p.set_title(
        f"{name}  thr={thr:.2f} {unit}  "
        f"strokes={len(strokes)}({ds:+d})  gaps={len(gaps)}({dw:+d})",
        fontsize=8, loc="left", color="white"
    )
    ax_p.set_ylabel(unit, fontsize=7, color="#aaa")
    ax_p.tick_params(colors="#777", labelsize=7)
    for sp in ax_p.spines.values(): sp.set_edgecolor("#333")
    ax_p.set_ylim(0, thr * 5)
axes[-1].set_xlabel("Time (s)", fontsize=9, color="#aaa")
plt.tight_layout(rect=[0, 0.01, 1, 1])
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'detector_comparison.png'),
            dpi=130, facecolor="#1a1a1a")
plt.close()

# ── PLOT 2: STROKE_END sweep ─────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor="#1a1a1a")
ax2.set_facecolor("#111")
ax2.axhline(EXPECTED_STROKES, color="white", lw=1.5, ls="--", alpha=0.7,
            label=f"Target={EXPECTED_STROKES}")
for sig, thr, sm, strokes, gaps, name, unit, col in results:
    ax2.plot(sweep_vals, sweep_table[name], color=col, lw=1.5, marker="o",
             markersize=4, label=name)
ax2.set_xlabel("STROKE_END_SAMP (samples)", color="#aaa", fontsize=10)
ax2.set_ylabel("Strokes detected", color="#aaa", fontsize=10)
ax2.set_title("Stroke count vs STROKE_END_SAMP\n"
              "(lower = more sensitive to short inter-letter pauses)",
              color="white", fontsize=10)
ax2.tick_params(colors="#777")
for sp in ax2.spines.values(): sp.set_edgecolor("#333")
ax2.legend(fontsize=8, framealpha=0.2, labelcolor="white")
sec_ax = ax2.secondary_xaxis("top",
    functions=(lambda x: x/FS*1000, lambda x: x*FS/1000))
sec_ax.set_xlabel("STROKE_END duration (ms)", color="#aaa", fontsize=9)
sec_ax.tick_params(colors="#777")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'detector_sweep.png'),
            dpi=130, facecolor="#1a1a1a")
plt.close()

print("\nSaved: detector_comparison.png  +  detector_sweep.png")
