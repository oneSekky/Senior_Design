"""
recorder.py — Raw IMU stream recorder for later replay.

Records every sample that arrives from the data source, plus timestamped
events (stroke_complete, word_gap) so that the replay can optionally
fast-forward to known boundaries.

File format (.imu.json):
{
  "version": 1,
  "created": "ISO-8601 string",
  "sample_rate": 104,
  "columns": ["t", "acc_x[mg]", ...],   # 7 entries, t in seconds
  "samples": [[t, ax, ay, az, gx, gy, gz], ...],
  "events":  [{"t": float, "type": str, "n": int}, ...]
}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class IMURecorder:
    VERSION = 1

    def __init__(self) -> None:
        self._samples: list[list] = []
        self._events: list[dict] = []
        self._t0: Optional[float] = None
        self._recording = False

    # ── Control ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._samples = []
        self._events = []
        self._t0 = time.monotonic()
        self._recording = True

    def stop(self) -> None:
        self._recording = False

    # ── Data ingestion ───────────────────────────────────────────────────────

    def record_sample(self, sample) -> None:
        """sample: sequence of 6 floats [ax, ay, az, gx, gy, gz]."""
        if not self._recording:
            return
        t = round(time.monotonic() - self._t0, 5)
        row = [t] + [round(float(v), 4) for v in sample[:6]]
        self._samples.append(row)

    def record_event(self, event_type: str) -> None:
        """Log a named event at the current wall-clock position."""
        if not self._recording:
            return
        t = round(time.monotonic() - self._t0, 5)
        self._events.append({"t": t, "type": event_type, "n": len(self._samples)})

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "version": self.VERSION,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sample_rate": 104,
            "columns": [
                "t", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]",
                "gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]",
            ],
            "samples": self._samples,
            "events": self._events,
        }
        Path(path).write_text(json.dumps(data, separators=(",", ":")))

    @staticmethod
    def load(path: str) -> dict:
        return json.loads(Path(path).read_text())

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def duration_s(self) -> float:
        if not self._samples:
            return 0.0
        return self._samples[-1][0]
