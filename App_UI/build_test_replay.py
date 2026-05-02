"""
build_test_replay.py — Generate a large test_replay.imu.json from box-page-10-2sec.

Uses all 140 boxes. Between each letter, inserts 500 samples of realistic resting
gravity (acc_z ≈ 1000 mg + small noise, gyro ≈ 0 mdps) so the causal HP filter
settles cleanly and the stroke detector fires once per letter.

Run from the App_UI directory:
    python build_test_replay.py
"""

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

FS = 104          # Hz
REST_SAMPLES = 500  # ~4.8 s between letters — enough for filter to settle

_ROOT = Path(__file__).parent.parent
PAGE_DIR = _ROOT / "Test_Data" / "side_mount" / "split_csvs" / "box-page-15-2sec"
OUT_PATH = Path(__file__).parent / "test_replay.imu.json"

COLUMNS = ["t", "acc_x[mg]", "acc_y[mg]", "acc_z[mg]",
           "gyro_x[mdps]", "gyro_y[mdps]", "gyro_z[mdps]"]


def rest_samples(n: int, t_start: float) -> list[list]:
    """Generate n samples of realistic resting gravity."""
    rows = []
    for i in range(n):
        t = t_start + i / FS
        ax = random.gauss(0.0, 4.0)
        ay = random.gauss(0.0, 4.0)
        az = random.gauss(1000.0, 8.0)
        gx = random.gauss(0.0, 15.0)
        gy = random.gauss(0.0, 15.0)
        gz = random.gauss(0.0, 15.0)
        rows.append([round(t, 6), round(ax, 3), round(ay, 3), round(az, 3),
                     round(gx, 3), round(gy, 3), round(gz, 3)])
    return rows


def load_box(csv_path: Path) -> list[list]:
    """Load a split CSV → list of [ax, ay, az, gx, gy, gz] rows (mg / mdps)."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for line in reader:
            if not line or line[0].startswith("#"):
                continue
            try:
                float(line[0])
            except ValueError:
                continue  # header row
            # columns: time[us], acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ...
            rows.append([float(line[1]), float(line[2]), float(line[3]),
                         float(line[4]), float(line[5]), float(line[6])])
    return rows


def main():
    random.seed(42)

    # Sorted so box_00_00 → box_13_09 in reading order
    box_files = sorted(PAGE_DIR.glob("box_*.csv"))
    print(f"Found {len(box_files)} boxes in {PAGE_DIR.name}")

    samples = []
    events = []
    t = 0.0

    # Start with resting gravity so the filter settles before the first stroke
    initial_rest = rest_samples(REST_SAMPLES, t)
    samples.extend(initial_rest)
    t += REST_SAMPLES / FS

    for idx, box_path in enumerate(box_files):
        stroke_rows = load_box(box_path)
        if not stroke_rows:
            print(f"  skip empty: {box_path.name}")
            continue

        n_stroke = len(stroke_rows)
        stroke_start_sample = len(samples)

        for row in stroke_rows:
            samples.append([round(t, 6)] + row)
            t += 1.0 / FS

        events.append({
            "t": samples[stroke_start_sample][0],
            "type": "stroke_complete",
            "n": n_stroke,
            "box": box_path.stem,
        })

        # Rest between letters
        rest = rest_samples(REST_SAMPLES, t)
        samples.extend(rest)
        t += REST_SAMPLES / FS

        if (idx + 1) % 10 == 0:
            print(f"  processed {idx + 1}/{len(box_files)} boxes "
                  f"({len(samples)} samples so far, t={t:.1f}s)")

    payload = {
        "version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "sample_rate": FS,
        "page": "box-page-15-2sec",
        "columns": COLUMNS,
        "samples": samples,
        "events": events,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = OUT_PATH.stat().st_size / 1e6
    duration_s = len(samples) / FS
    print(f"\nWrote {OUT_PATH}")
    print(f"  {len(samples):,} samples  |  {len(events)} strokes  |  "
          f"{duration_s:.0f} s ({duration_s/60:.1f} min)  |  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
