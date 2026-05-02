"""
stroke_buffer.py — Real-time stroke segmentation from a live IMU stream.

Approach:
  - Apply a per-channel causal Butterworth HP filter to remove gravity.
  - Compute filtered-acc magnitude for activity detection.
  - Raw (unfiltered) samples are accumulated so that the inference engine can
    apply filtfilt (zero-phase) on the full completed stroke.

Callbacks are invoked synchronously from the thread that calls feed().
When used with Qt signals (DataSource -> main thread slot -> feed()), all
callbacks run on the main thread — no locking needed.

Note: this is only used for live pen data (USB / BLE).
Replay mode bypasses this entirely and drives inference directly from the
recorded event timestamps.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Optional

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

FS = 104

# How long to wait after connect before enabling stroke detection.
# The causal HP filter needs time to adapt to the resting gravity signal.
# At startup lfilter_zi assumes a unit-step input, not the actual ~1000 mg
# resting acc_z, so the first ~500 samples produce a large spurious transient.
WARMUP_SAMPLES = 500            # ~4.8 s

ACTIVITY_THRESHOLD = 20.0       # mg on HP-filtered acc magnitude
SMOOTH_WINDOW = 10              # samples for running mean smoothing
MIN_STROKE_SAMPLES = 20         # shorter strokes ignored
STROKE_END_SAMPLES = 52         # ~500 ms silence → letter complete
WORD_GAP_SAMPLES = 83           # ~800 ms silence → word gap


class _CausalHPFilter:
    """Per-channel causal IIR high-pass filter with persistent state."""

    def __init__(self, cutoff: float, fs: int, order: int, n_channels: int):
        self._b, self._a = butter(order, cutoff, "high", fs=fs)
        zi_1ch = lfilter_zi(self._b, self._a)              # (order,)
        self._zi = np.tile(zi_1ch[:, None], (1, n_channels)).copy()  # (order, C)

    def process(self, sample: np.ndarray) -> np.ndarray:
        """sample: (n_channels,) -> filtered (n_channels,)"""
        x = sample[np.newaxis, :]   # (1, C)
        out = np.empty_like(x)
        for ch in range(x.shape[1]):
            y, self._zi[:, ch] = lfilter(
                self._b, self._a, x[:, ch], zi=self._zi[:, ch]
            )
            out[0, ch] = y[0]
        return out[0]

    def reset(self) -> None:
        self._zi[:] = 0.0


class StrokeBuffer:
    """
    Feed raw IMU samples one at a time via feed().  When a stroke is complete
    (activity followed by STROKE_END_SAMPLES frames of silence) on_stroke_complete
    is called with a (N, 6) float32 array of *raw* samples (no filtering applied).
    on_word_gap is called when silence exceeds WORD_GAP_SAMPLES.

    A warmup period of WARMUP_SAMPLES is observed after each reset() to let the
    HP filter adapt to the resting gravity signal before detection begins.
    """

    def __init__(
        self,
        on_stroke_complete: Callable[[np.ndarray], None],
        on_word_gap: Optional[Callable[[], None]] = None,
    ) -> None:
        self.on_stroke_complete = on_stroke_complete
        self.on_word_gap = on_word_gap

        self._hp = _CausalHPFilter(cutoff=0.5, fs=FS, order=3, n_channels=3)
        self._mag_win: deque[float] = deque(maxlen=SMOOTH_WINDOW)

        self._raw_buf: list[np.ndarray] = []
        self._inactive_count = 0
        self._is_active = False
        self._word_gap_fired = False
        self._warmup_count = WARMUP_SAMPLES

    # ── Public API ───────────────────────────────────────────────────────────

    def feed(self, sample) -> None:
        """sample: sequence of 6 floats [ax, ay, az, gx, gy, gz] in mg/mdps."""
        s = np.asarray(sample, dtype=np.float32)

        # Always run the HP filter to keep state current
        filt_acc = self._hp.process(s[:3])

        # During warmup: filter state is settling — don't detect strokes
        if self._warmup_count > 0:
            self._warmup_count -= 1
            return

        mag = float(np.sqrt((filt_acc ** 2).sum()))
        self._mag_win.append(mag)
        smoothed = float(np.mean(self._mag_win))

        if smoothed > ACTIVITY_THRESHOLD:
            self._is_active = True
            self._inactive_count = 0
            self._word_gap_fired = False
            self._raw_buf.append(s)
        else:
            if self._is_active:
                self._raw_buf.append(s)
                self._inactive_count += 1
                if self._inactive_count >= STROKE_END_SAMPLES:
                    data = np.array(self._raw_buf[:-STROKE_END_SAMPLES], dtype=np.float32)
                    if len(data) >= MIN_STROKE_SAMPLES:
                        self.on_stroke_complete(data)
                    self._raw_buf = []
                    self._is_active = False
                    self._inactive_count = 0
            else:
                if not self._word_gap_fired:
                    self._inactive_count += 1
                    if self._inactive_count >= WORD_GAP_SAMPLES and self.on_word_gap:
                        self.on_word_gap()
                        self._word_gap_fired = True

    def flush(self) -> None:
        """Force-complete any in-progress stroke (call on disconnect)."""
        if self._raw_buf and self._is_active:
            data = np.array(self._raw_buf, dtype=np.float32)
            if len(data) >= MIN_STROKE_SAMPLES:
                self.on_stroke_complete(data)
        self._reset_state()

    def reset(self) -> None:
        """Discard all buffered data without firing callbacks."""
        self._reset_state()

    # ── Commented-out calibration support ────────────────────────────────────
    # Uncomment and wire up after the physical pen is working.
    #
    # def set_threshold(self, threshold: float) -> None:
    #     """Dynamically update the activity threshold (e.g. from calibration)."""
    #     global ACTIVITY_THRESHOLD
    #     ACTIVITY_THRESHOLD = threshold
    #
    # def start_calibration(self, on_calibration_done: Callable[[float], None]) -> None:
    #     """
    #     Record one stroke and call on_calibration_done with the suggested
    #     ACTIVITY_THRESHOLD = (min smoothed mag during stroke) / 2.
    #     Call reset() first to clear any in-flight state.
    #     """
    #     self._calibrating = True
    #     self._cal_callback = on_calibration_done
    #     self._cal_mags: list[float] = []
    #
    # In feed(), inside the `if smoothed > ACTIVITY_THRESHOLD` branch, add:
    #     if getattr(self, '_calibrating', False):
    #         self._cal_mags.append(smoothed)
    # And in the stroke-complete branch, add:
    #     if getattr(self, '_calibrating', False) and self._cal_mags:
    #         suggested = min(self._cal_mags) / 2.0
    #         self._calibrating = False
    #         self._cal_callback(suggested)

    def _reset_state(self) -> None:
        self._raw_buf = []
        self._mag_win.clear()
        self._hp.reset()
        self._inactive_count = 0
        self._is_active = False
        self._word_gap_fired = False
        self._warmup_count = WARMUP_SAMPLES
