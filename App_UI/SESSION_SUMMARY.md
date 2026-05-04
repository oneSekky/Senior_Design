# Session Summary — App_UI Stroke Detection & Signal Analysis Overhaul
**Date:** 2026-05-04  
**Branch:** main

---

## Overview

This session covered four major areas:
1. Fixing broken stroke detection with a new adaptive valley algorithm
2. Adding async inference to the Qt app
3. Exploring 3D pen position reconstruction from high-rate IMU data
4. Reorganising the App_UI folder

---

## 1. Root Cause: Why Detection Was Broken

### Problem
Writing `"Sekander I don't think this is working very well"` (~55 expected strokes) was
producing only 2–12 super-strokes, none matching real letters.

### Diagnosis
Data log `Data_Log_26_05_04_11_53_38.csv` was analysed (`viz_analysis.py`,
`compare_detection.py`). The fixed threshold of `9.2 mg/sample` (baseline p99 × 4)
was too low: hand tremor and pen repositioning between letters keeps the acc-delta
signal continuously above any reasonable fixed value. All letters inside a word merged
into one long stroke.

Raising the multiplier from 4× to 8× helped slightly but didn't fix the fundamental
issue — a fixed amplitude threshold cannot self-adapt to different writing strengths.

---

## 2. New Stroke Detection: Valley-Based Adaptive Threshold

### Key Insight
Instead of comparing the signal against a fixed floor, compare it against itself:
a stroke boundary is the relative valley between peaks, not an absolute quiet level.

### Algorithm (`stroke_buffer.py` — complete rewrite)

```
detection signal = frame-to-frame acc delta (jerk): ||acc[n] - acc[n-1]||
```

Gravity (DC offset) cancels automatically — no HP filter needed.

```
thr = max(MIN_ABSOLUTE, VALLEY_FRACTION × rolling_peak)
```

- `rolling_peak` = max of last `PEAK_WINDOW` smoothed values **that were above
  MIN_ABSOLUTE** (quiet periods don't deflate the peak)
- `VALLEY_FRACTION = 0.50` — signal must drop below 50% of recent peak to be quiet
- `MIN_ABSOLUTE = 8.0` mg/sample — absolute floor; anything below this is not writing
- `PEAK_WINDOW = 52` samples (~0.5 s at 104 Hz)
- `STROKE_END_SAMPLES = 10` (~96 ms) — consecutive quiet samples to commit a boundary
- `WORD_GAP_SAMPLES = 83` (~800 ms) — unchanged

**Valley buffering:** quiet samples are held in a side buffer. If activity resumes before
`STROKE_END_SAMPLES`, they are re-absorbed into the active stroke. This prevents
short hesitations inside a letter from prematurely ending the stroke.

### Result
Sweep on the test file: `VALLEY_FRACTION=0.50, STROKE_END=8` → **57 strokes**
vs 55 expected. Fixed threshold best case: 17 strokes. The new algorithm is
self-normalising — no per-user amplitude tuning required.

### Calibration
`set_threshold()` is now a **no-op** (kept for API compatibility). Calibration is still
used for **timing parameters** (`stroke_end`, `word_gap`) via `_analyze_cal_phrase`.
The calibration phrase analysis was updated to use the same valley-detection logic
internally when finding gap boundaries.

### Key constants (back-compat aliases kept)
```python
ACTIVITY_THRESHOLD = MIN_ABSOLUTE   # imported by main.py
_CAL_THRESHOLD     = 5.0
```

---

## 3. Async Inference (`main.py`)

### Problem
`_run_inference()` was calling the model synchronously on the Qt main thread,
blocking the UI during inference.

### Fix
- Added `ThreadPoolExecutor(max_workers=1)` — single worker serialises inference
  so GPU/model state is never accessed concurrently
- Result delivered back to Qt via `pyqtSignal(object)` → `_on_inference_result()`
- `closeEvent` calls `executor.shutdown(wait=False)` to clean up on exit

```python
def _run_inference(self, stroke):
    future = self._executor.submit(self._inference.predict, stroke.copy())
    future.add_done_callback(self._inference_done)

def _inference_done(self, future):
    pred = future.result()
    if pred is not None:
        self._inference_result.emit(pred)   # safe Qt signal

@pyqtSlot(object)
def _on_inference_result(self, pred):
    self.canvas.add_letter(pred)
    self._count_label.setText(f"{self.canvas.letter_count} letters")
```

---

## 4. Linear Resampling in Preprocessing (`inference.py`)

### Problem
Variable-length strokes were zero-padded to 239 timesteps, which adds a silent
prefix and misaligns the trajectory with what the model was trained on.

### Fix
Replace zero-padding with **linear interpolation** (time-normalization):

```python
if N != N_TIMESTEPS:
    x_old = np.linspace(0.0, 1.0, N)
    x_new = np.linspace(0.0, 1.0, N_TIMESTEPS)
    features = np.column_stack([
        np.interp(x_new, x_old, features[:, i])
        for i in range(features.shape[1])
    ])
```

This preserves the spatial trajectory shape regardless of writing speed.

---

## 5. Bug Fix: Missing `FS` Import in `main.py`

`_analyze_cal_phrase` used `FS` but it wasn't imported from `stroke_buffer`.
Fixed by adding `FS` to the import line:

```python
from stroke_buffer import MIN_ABSOLUTE, VALLEY_FRACTION, PEAK_WINDOW, FS
```

---

## 6. 3D Pen Position Reconstruction (Analysis)

### Data
`Data_Log_26_05_04_12_13_22.csv` — 6756.8 Hz capture, 60 879 samples = 9.01 s.

### Approach
1. Estimate gravity vector `G` from stillest 0.3 s window **within first 2 seconds**
   (flat-on-paper baseline). Restricting to the first 2 s is critical — searching the
   whole file picks up a window at t=7.96 s when the pen is in a completely different
   orientation.
2. Subtract `G` → dynamic acceleration (writing motion only)
3. Downsample to ~500 Hz (anti-alias + decimate)
4. Double-integrate: acc → velocity → position
5. Project onto writing plane (perpendicular to gravity unit vector)

### Why double integration works conceptually
Newton's 2nd law: `a = F/m`. Integrating measured acceleration gives velocity;
integrating velocity gives displacement. Gravity is a constant DC offset that cancels
after subtraction. The sensor measures the actual net force on the pen tip, so the
trajectory IS recoverable — the only limitation is integration drift from residual
bias.

### HP Filter vs ZUPT
- **HP filter on position/velocity** removes slow drift but also kills the genuine
  left-to-right progression across the page (inter-letter spacing IS slow drift).
  This approach was tried and discarded.
- **ZUPT (Zero Velocity Update):** during quiet moments (pen between letters),
  force velocity to zero. This bounds drift to at most one stroke's integration time
  while preserving inter-letter spacing. `position_zupt.py` implements this.
  The ZUPT threshold still needs tuning — the p10-based estimate was too high
  (421 mg), classifying 98% of samples as quiet.

### Would calibration fix gravity drift?
Yes, significantly. 6-position calibration (bias, scale factor, cross-axis sensitivity)
reduces accelerometer bias from 1–5 mg to < 0.1 mg. At dt = 148 µs and v = v₀ + a×dt,
the drift over 9 s drops from ~40 cm to ~4 mm — enough for legible letter reconstruction.

---

## 7. Folder Reorganisation

The `App_UI/` root was cluttered with analysis scripts, data logs, and outputs.

### New Structure
```
App_UI/
├── main.py              # Qt app entry point
├── canvas.py            # letter canvas widget
├── data_source.py       # BLE / replay data source
├── inference.py         # model preprocessing + inference
├── recorder.py          # IMU recorder
├── stroke_buffer.py     # stroke segmentation
├── requirements.txt
├── README.md
├── profile.json         # default profile
│
├── analysis/            # analysis scripts (offline)
│   ├── output/          # generated PNGs, PDFs
│   ├── compare_detection.py
│   ├── compare_detectors.py
│   ├── diagnose_app_vs_train.py
│   ├── build_test_replay.py
│   ├── position_3d.py
│   ├── position_3d_v2.py
│   ├── position_zupt.py
│   └── viz_analysis.py
│
├── data_logs/           # raw CSV captures
│   ├── Data_Log_26_05_04_10_35_15.csv
│   ├── Data_Log_26_05_04_11_53_38.csv
│   ├── Data_Log_26_05_04_12_13_22.csv
│   └── ...
│
└── profiles/            # saved calibration profiles
    ├── cal2-sek.json
    ├── sekander.json
    └── sek2.json
```

### Path Updates Applied
All 8 analysis scripts were updated after the move:
- `sys.path` inserts now reference `..` (parent = `App_UI/`)
- CSV paths use `../data_logs/<file>.csv`
- Output paths use `output/<file>.png`
- `_ROOT` in scripts that reference `Senior_Design/` got an extra `.parent`
- `out_path` in `build_test_replay.py` now writes to `../data_logs/test_replay.imu.json`
- `viz_analysis.py` had `os.chdir()` removed; all paths are now `__file__`-relative

---

## Decisions & Rationale

| Decision | Rationale |
|---|---|
| Valley detection instead of fixed threshold | Fixed threshold fundamentally can't work for continuous writing — tremor keeps signal above any fixed level |
| `VALLEY_FRACTION = 0.50` | Data shows inter-letter valleys at 25–35 mg, peaks at 60–80 mg; 50% gives clean separation |
| `STROKE_END = 10` samples (96 ms) | At 104 Hz, inter-letter gaps are ~100–200 ms; 288 ms (old 30-sample default) was too long |
| Keep calibration for timing | Stroke-end and word-gap timing still benefits from per-session calibration; amplitude calibration removed |
| `ThreadPoolExecutor(max_workers=1)` | Single worker prevents concurrent model access while keeping UI responsive |
| Linear resampling over zero-padding | Preserves trajectory shape; zero-padding misaligns the signal relative to training data |
| ZUPT over HP filter for position | HP filter kills inter-letter spacing; ZUPT bounds drift while preserving real motion |
| Gravity window restricted to first 2 s | Stillest window in whole file was at t=7.96 s (pen in air, different orientation) |
