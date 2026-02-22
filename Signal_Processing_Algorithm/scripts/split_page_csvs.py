"""
split_page_csvs.py

Splits each page CSV into fixed 208-sample chunks (2 sec @ 104 Hz),
one per box (10 cols x 14 rows = 140 boxes per page).

Automatically processes all .csv files directly in SRC_DIR (not in subdirs).
Output folder per CSV is named after the file stem.

Output: Test_Data/side_mount/split_csvs/<stem>/box_<row>_<col>.csv
If a page runs short (sensor stopped early), trailing boxes are silently dropped.
"""

import os
import shutil
import pandas as pd

SAMPLES_PER_BOX = 208
COLS = 10
ROWS = 14
BOXES_PER_PAGE = COLS * ROWS  # 140

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Test_Data', 'side_mount', 'csvs')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Test_Data', 'side_mount', 'split_csvs')


def find_csvs(src_dir):
    """Return sorted list of (stem, filename) for all .csv files directly in src_dir."""
    files = []
    for fname in sorted(os.listdir(src_dir)):
        if fname.lower().endswith('.csv') and os.path.isfile(os.path.join(src_dir, fname)):
            stem = os.path.splitext(fname)[0]
            files.append((stem, fname))
    return files


def load_csv(path):
    """Load CSV, skipping comment/metadata lines before the data header."""
    with open(path, 'r') as f:
        lines = f.readlines()
    header_idx = next(
        i for i, l in enumerate(lines)
        if 'time' in l.lower() or 'acc' in l.lower()
    )
    comment_lines = lines[:header_idx]
    df = pd.read_csv(path, skiprows=header_idx)
    return df, comment_lines


def split_page(stem, csv_filename):
    src_path = os.path.join(SRC_DIR, csv_filename)
    df, comment_lines = load_csv(src_path)
    total = len(df)
    print(f'  {total} rows (need {SAMPLES_PER_BOX} per box, {BOXES_PER_PAGE} boxes)')

    page_out = os.path.join(OUT_DIR, stem)
    os.makedirs(page_out, exist_ok=True)

    written = 0
    for box_idx in range(BOXES_PER_PAGE):
        row = box_idx // COLS
        col = box_idx % COLS
        start = box_idx * SAMPLES_PER_BOX
        end = start + SAMPLES_PER_BOX
        if end > total:
            print(f'  Data ended at box {box_idx} (row={row}, col={col}); '
                  f'dropped {BOXES_PER_PAGE - box_idx} trailing box(es)')
            break
        chunk = df.iloc[start:end].copy()
        chunk['time[us]'] = chunk['time[us]'] - chunk['time[us]'].iloc[0]

        out_path = os.path.join(page_out, f'box_{row:02d}_{col:02d}.csv')
        with open(out_path, 'w') as f:
            for line in comment_lines:
                f.write(line)
        chunk.to_csv(out_path, mode='a', index=False)
        written += 1

    print(f'  Wrote {written} chunks to {page_out}')


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
