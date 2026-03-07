"""
split_page_csvs.py

Splits each page CSV into 140 box chunks (14 rows x 10 cols) using IMU-detected
row boundaries.

Algorithm:
  1. Smooth acc_y with a 500-sample box-car filter (~5 s at 104 Hz).
     This removes per-stroke noise while preserving the slow row-level drift.
  2. Determine the dominant lateral direction from the mean of the smoothed
     signal (positive mean -> look for local minima; negative -> local maxima).
     The carriage return is always the brief reversal between rows, which shows
     up as a local extremum in the smoothed signal.
  3. Find the 13 most prominent extrema (one per carriage return), each
     separated by at least 800 samples (~8 s).
  4. Use these 13 sample indices to divide the recording into 14 row segments.
  5. Split each row evenly into 10 equal-length column chunks.
  6. Each chunk is written with BOX_PADDING samples of overlap on each side,
     clamped to the row boundary (never crossing a carriage return).
     This ensures the full letter stroke is always inside the chunk regardless
     of the unknown phase offset between IMU start and the first box boundary.

Falls back to fixed 208-sample splitting if exactly 13 extrema are not found.

Output: Test_Data/side_mount/split_csvs/<stem>/box_<row>_<col>.csv
"""

import os
import shutil
import csv as csv_mod

COLS = 10
ROWS = 14
SAMPLES_PER_BOX_NOMINAL = 208        # 2 s @ 104 Hz  (fallback only)
SMOOTH_WIN = 500                     # samples (~5 s) for acc_y smoothing
EXTREMUM_SEP = 800                   # minimum samples between carriage returns
BOX_PADDING = 34                     # ~0.33 s overlap on each side

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Test_Data', 'side_mount', 'csvs')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Test_Data', 'side_mount', 'split_csvs')


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_csvs(src_dir):
    """Return sorted list of (stem, filename) for .csv files directly in src_dir."""
    files = []
    for fname in sorted(os.listdir(src_dir)):
        if fname.lower().endswith('.csv') and os.path.isfile(os.path.join(src_dir, fname)):
            stem = os.path.splitext(fname)[0]
            files.append((stem, fname))
    return files


def load_csv_raw(path):
    """
    Return (header_line, comment_lines, data_rows).
    data_rows is a list of string-value lists (one per sample).
    """
    with open(path, 'r', newline='') as f:
        lines = f.readlines()
    header_idx = next(
        i for i, l in enumerate(lines)
        if 'time' in l.lower() or 'acc' in l.lower()
    )
    comment_lines = lines[:header_idx]
    header_line   = lines[header_idx]
    data_rows = []
    for line in lines[header_idx + 1:]:
        row = line.strip().split(',')
        if row and row[0]:
            data_rows.append(row)
    return header_line, comment_lines, data_rows


# ---------------------------------------------------------------------------
# Row-boundary detection
# ---------------------------------------------------------------------------

def _box_smooth(values, win):
    """Box-car (moving-average) smoothing."""
    N = len(values)
    out = [0.0] * N
    for i in range(N):
        s = max(0, i - win // 2)
        e = min(N, i + win // 2)
        out[i] = sum(values[s:e]) / (e - s)
    return out


def detect_row_boundaries(data_rows, n_cr=ROWS - 1):
    """
    Detect the n_cr carriage-return events and return their sample indices.

    Returns a list of n_cr sample indices (sorted ascending) or None on failure.
    """
    N = len(data_rows)
    try:
        acc_y = [float(r[2]) for r in data_rows]
    except (IndexError, ValueError):
        return None

    smooth_ay = _box_smooth(acc_y, SMOOTH_WIN)

    # Decide whether to look for local minima or maxima:
    # writing rows drift toward large positive OR large negative acc_y;
    # the carriage return is the brief dip back the other way.
    mean_ay = sum(smooth_ay) / N
    looking_for = 'min' if mean_ay >= 0 else 'max'

    # Find all local extrema with the required window separation
    extrema = []
    for i in range(EXTREMUM_SEP, N - EXTREMUM_SEP):
        window = smooth_ay[i - EXTREMUM_SEP: i + EXTREMUM_SEP + 1]
        if looking_for == 'min' and smooth_ay[i] == min(window):
            extrema.append((i, smooth_ay[i]))
        elif looking_for == 'max' and smooth_ay[i] == max(window):
            extrema.append((i, smooth_ay[i]))

    # Merge duplicates: if two extrema are closer than EXTREMUM_SEP, keep the
    # more extreme one
    filtered = []
    for e in extrema:
        if not filtered or e[0] - filtered[-1][0] > EXTREMUM_SEP:
            filtered.append(e)
        elif (looking_for == 'min' and e[1] < filtered[-1][1]) or \
             (looking_for == 'max' and e[1] > filtered[-1][1]):
            filtered[-1] = e

    if len(filtered) != n_cr:
        return None

    return [idx for idx, _ in filtered]


# ---------------------------------------------------------------------------
# Main splitting routine
# ---------------------------------------------------------------------------

def split_page(stem, csv_filename, verbose=True):
    src_path = os.path.join(SRC_DIR, csv_filename)
    header_line, comment_lines, data_rows = load_csv_raw(src_path)
    N = len(data_rows)

    if verbose:
        print(f'  {N} samples total  (nominal {ROWS * COLS * SAMPLES_PER_BOX_NOMINAL})')

    page_out = os.path.join(OUT_DIR, stem)
    os.makedirs(page_out, exist_ok=True)

    # --- detect row boundaries ---
    cr_samples = detect_row_boundaries(data_rows)

    if cr_samples is not None:
        row_boundaries = [0] + cr_samples + [N]
        method = 'IMU-detected'
    else:
        detected = len(cr_samples) if cr_samples is not None else 0
        if verbose:
            print(f'  WARNING: detected {detected} carriage returns '
                  f'(expected {ROWS - 1}); falling back to fixed-size splitting.')
        row_boundaries = [i * COLS * SAMPLES_PER_BOX_NOMINAL for i in range(ROWS + 1)]
        method = 'fixed-size fallback'

    if verbose:
        print(f'  Method: {method}')

    written = 0
    skipped = 0

    for row_idx in range(ROWS):
        row_s = row_boundaries[row_idx]
        row_e = min(row_boundaries[row_idx + 1], N)
        row_len = row_e - row_s

        if row_len < COLS:
            if verbose:
                print(f'  Row {row_idx:2d}: only {row_len} samples -- skipping')
            skipped += COLS
            continue

        box_size  = row_len // COLS
        remainder = row_len  % COLS

        if verbose:
            t0_us = float(data_rows[0][0])
            t_s = (float(data_rows[row_s][0]) - t0_us) / 1e6
            t_e_idx = min(row_e - 1, N - 1)
            t_e = (float(data_rows[t_e_idx][0]) - t0_us) / 1e6
            print(f'  Row {row_idx:2d}: samples {row_s:6d}-{row_e:6d} '
                  f'({t_s:.1f}s-{t_e:.1f}s) '
                  f'{row_len} samples -> {box_size}/box (+{remainder} extra)')

        for col_idx in range(COLS):
            extra = 1 if col_idx < remainder else 0
            box_s = row_s + col_idx * box_size + min(col_idx, remainder)
            box_e = box_s + box_size + extra

            if box_e > N:
                if verbose:
                    print(f'    box_{row_idx:02d}_{col_idx:02d}: out of data, skipping')
                skipped += 1
                continue

            # Pad each side by BOX_PADDING, clamped to row boundaries so we
            # never bleed across a carriage return into a different row.
            padded_s = max(row_s, box_s - BOX_PADDING)
            padded_e = min(row_e, box_e + BOX_PADDING)

            chunk = data_rows[padded_s:padded_e]

            # Reset timestamps to start at 0
            try:
                t0 = float(chunk[0][0])
                out_rows = []
                for r in chunk:
                    nr = list(r)
                    nr[0] = str(int(float(r[0]) - t0))
                    out_rows.append(nr)
            except (ValueError, IndexError):
                out_rows = chunk

            out_path = os.path.join(page_out, f'box_{row_idx:02d}_{col_idx:02d}.csv')
            with open(out_path, 'w', newline='') as f:
                for line in comment_lines:
                    f.write(line)
                f.write(header_line)
                writer = csv_mod.writer(f)
                writer.writerows(out_rows)
            written += 1

    if verbose:
        print(f'  Wrote {written} boxes, skipped {skipped}')
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    csvs = find_csvs(SRC_DIR)
    if not csvs:
        print(f'No .csv files found in {SRC_DIR}')
    for stem, filename in csvs:
        print(f'\n{filename}:')
        split_page(stem, filename)
    print('\nDone.')
