"""
stroke_buffer.py — Adaptive valley-based stroke segmentation.

Detection signal: frame-to-frame acc delta (jerk) ||acc[n] - acc[n-1]||.
  Cancels gravity (constant offset → zero delta).

Instead of a fixed amplitude threshold, strokes are segmented by finding
valleys in the smoothed signal relative to the local rolling peak:

    quiet = smoothed < max(MIN_ABSOLUTE, VALLEY_FRACTION * rolling_peak)

VALLEY_FRACTION makes detection self-normalising: a light writer and a heavy
writer both trigger on the same relative dip — no per-user amplitude tuning.

A stroke ends after STROKE_END_SAMPLES consecutive quiet samples (set by
calibration). A word gap fires after WORD_GAP_SAMPLES quiet samples. Both
parameters are still calibratable; only the amplitude threshold is gone.

Callbacks are invoked synchronously from the thread that calls feed().
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

FS = 104

WARMUP_SAMPLES    = 20     # ~200 ms — let smoothing window fill after reset
SMOOTH_WINDOW     = 10     # samples for running-mean smoothing
MIN_STROKE_SAMPLES = 20    # strokes shorter than this are discarded

# Relative-threshold parameters
MIN_ABSOLUTE    = 8.0      # mg/sample floor — below this = definitely not writing
VALLEY_FRACTION = 0.50     # quiet when signal < fraction × rolling peak
PEAK_WINDOW     = 52       # rolling peak window, samples (~0.5 s @ 104 Hz)

# Timing defaults (overridden by calibration via set_stroke_end / set_word_gap)
STROKE_END_SAMPLES = 10    # consecutive quiet samples → stroke complete
WORD_GAP_SAMPLES   = 83    # consecutive quiet samples → word gap

# Back-compat aliases (imported by main.py)
ACTIVITY_THRESHOLD = MIN_ABSOLUTE
_CAL_THRESHOLD     = 5.0


class _CausalHPFilter:
    """Per-channel causal IIR high-pass filter with persistent state."""

    def __init__(self, cutoff: float, fs: int, order: int, n_channels: int):
        self._b, self._a = butter(order, cutoff, "high", fs=fs)
        zi_1ch = lfilter_zi(self._b, self._a)
        self._zi = np.tile(zi_1ch[:, None], (1, n_channels)).copy()

    def process(self, sample: np.ndarray) -> np.ndarray:
        x = sample[np.newaxis, :]
        out = np.empty_like(x)
        for ch in range(x.shape[1]):
            y, self._zi[:, ch] = lfilter(self._b, self._a, x[:, ch], zi=self._zi[:, ch])
            out[0, ch] = y[0]
        return out[0]

    def reset(self) -> None:
        self._zi[:] = 0.0


class StrokeBuffer:
    """
    Feed raw IMU samples one at a time via feed(). When a stroke is complete
    (activity followed by STROKE_END_SAMPLES frames of quiet) on_stroke_complete
    is called with a (N, 6) float32 array of raw samples.
    on_word_gap is called when quiet exceeds WORD_GAP_SAMPLES.

    Detection uses a relative (self-normalising) threshold:
        thr = max(MIN_ABSOLUTE, VALLEY_FRACTION * rolling_peak)
    where rolling_peak is the max of the last PEAK_WINDOW smoothed values
    that were themselves above MIN_ABSOLUTE.

    STROKE_END and WORD_GAP are still calibratable timing parameters.
    set_threshold() is retained for API compatibility but is a no-op.
    """

    def __init__(
        self,
        on_stroke_complete: Callable[[np.ndarray], None],
        on_word_gap: Optional[Callable[[], None]] = None,
    ) -> None:
        self.on_stroke_complete = on_stroke_complete
        self.on_word_gap        = on_word_gap

        self._stroke_end: int   = STROKE_END_SAMPLES
        self._word_gap: int     = WORD_GAP_SAMPLES

        self._mag_win:  deque[float] = deque(maxlen=SMOOTH_WINDOW)
        self._peak_win: deque[float] = deque(maxlen=PEAK_WINDOW)
        self._prev_acc: np.ndarray | None = None

        self._active_buf: list  = []   # raw samples during active phase
        self._valley_buf: list  = []   # raw samples buffered during quiet valley
        self._quiet_count:  int = 0    # consecutive quiet samples
        self._is_active:   bool = False
        self._word_gap_fired: bool = False
        self._warmup_count: int = WARMUP_SAMPLES

    # ── Public API ────────────────────────────────────────────────────────────

    def feed(self, sample) -> None:
        """sample: [ax, ay, az, gx, gy, gz] in mg / mdps."""
        s = np.asarray(sample, dtype=np.float32)

        # Prime + warmup
        if self._prev_acc is None:
            self._prev_acc = s[:3].copy()
            self._warmup_count -= 1
            return
        if self._warmup_count > 0:
            self._warmup_count -= 1
            self._prev_acc = s[:3].copy()
            return

        # Acc-delta magnitude
        delta = s[:3] - self._prev_acc
        self._prev_acc = s[:3].copy()
        raw_mag = float(np.sqrt((delta ** 2).sum()))

        # Smooth
        self._mag_win.append(raw_mag)
        smoothed = float(np.mean(self._mag_win))

        # Update rolling peak only from active samples so quiet valleys don't
        # deflate it — the peak represents "recent writing amplitude".
        if smoothed >= MIN_ABSOLUTE:
            self._peak_win.append(smoothed)

        rolling_peak = float(max(self._peak_win)) if self._peak_win else 0.0
        thr = max(MIN_ABSOLUTE, VALLEY_FRACTION * rolling_peak)

        is_active = smoothed >= thr

        # ── State machine ─────────────────────────────────────────────────────
        if self._is_active:
            if is_active:
                # Still active: absorb any buffered valley back into stroke
                if self._valley_buf:
                    self._active_buf.extend(self._valley_buf)
                    self._valley_buf = []
                self._active_buf.append(s)
                self._quiet_count    = 0
                self._word_gap_fired = False
            else:
                # Gone quiet: buffer sample in valley
                self._valley_buf.append(s)
                self._quiet_count += 1

                # Word gap (fires once per gap)
                if (not self._word_gap_fired
                        and self._quiet_count >= self._word_gap
                        and self.on_word_gap is not None):
                    self.on_word_gap()
                    self._word_gap_fired = True

                # Stroke end
                if self._quiet_count >= self._stroke_end:
                    data = np.array(self._active_buf, dtype=np.float32)
                    if len(data) >= MIN_STROKE_SAMPLES:
                        self.on_stroke_complete(data)
                    self._active_buf  = []
                    self._valley_buf  = []
                    self._quiet_count = 0
                    self._is_active   = False
                    self._word_gap_fired = False
        else:
            if is_active:
                # Onset of new stroke
                self._is_active      = True
                self._word_gap_fired = False
                self._quiet_count    = 0
                self._valley_buf     = []
                self._active_buf     = [s]
            else:
                # Idle: count for word gap between strokes
                if not self._word_gap_fired:
                    self._quiet_count += 1
                    if (self._quiet_count >= self._word_gap
                            and self.on_word_gap is not None):
                        self.on_word_gap()
                        self._word_gap_fired = True

    def flush(self) -> None:
        """Force-complete any in-progress stroke (call on disconnect)."""
        if self._active_buf and self._is_active:
            data = np.array(self._active_buf, dtype=np.float32)
            if len(data) >= MIN_STROKE_SAMPLES:
                self.on_stroke_complete(data)
        self._reset_state()

    def reset(self) -> None:
        """Discard all buffered data without firing callbacks."""
        self._reset_state()

    def set_threshold(self, threshold: float) -> None:
        """No-op — threshold is now self-calibrating via valley detection."""
        pass

    def set_stroke_end(self, samples: int) -> None:
        self._stroke_end = max(5, samples)

    def set_word_gap(self, samples: int) -> None:
        self._word_gap = max(self._stroke_end + 5, samples)

    # Kept for API compatibility — no longer used
    def start_calibration(self, on_progress, on_done, n_strokes: int = 3) -> None:
        pass

    def cancel_calibration(self) -> None:
        pass

    def _reset_state(self) -> None:
        self._mag_win.clear()
        self._peak_win.clear()
        self._prev_acc    = None
        self._active_buf  = []
        self._valley_buf  = []
        self._quiet_count = 0
        self._is_active   = False
        self._word_gap_fired = False
        self._warmup_count   = WARMUP_SAMPLES
