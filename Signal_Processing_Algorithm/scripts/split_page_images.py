"""
split_page_images.py

Finds grid lines by scanning the image and detecting dark horizontal/vertical
lines. Uses those lines to crop 140 letter boxes per page.

Automatically processes all .jpg files directly in SRC_DIR (not in subdirs).
Output folder per image is named after the file stem.

Grid layout: 14 rows x 10 cols = 140 boxes
  - 11 vertical lines  : left border + 9 internal + right border
  - 13 horizontal lines: dividers between the 14 rows (no top/bottom outer border)
  - Row 0 top  = hlines[0] - spacing  (extrapolated)
  - Row 13 bot = hlines[12] + spacing (extrapolated)
  -> row_bounds: [extrap_top] + hlines[0..12] + [extrap_bot] = 15 entries for 14 rows

Algorithm:
  1. Compute mean darkness of each column across all rows -> peaks = vertical lines.
  2. Compute mean darkness of each row across all columns -> peaks = horizontal lines.
  3. Crop each cell as 64x64 grayscale PNG.

Output: Test_Data/side_mount/split_images/<stem>/box_<row>_<col>.png
"""

import os
import shutil

import cv2
import numpy as np

COLS = 10
ROWS = 14
N_VLINES = 11  # left border + 9 internal dividers + right border
N_HLINES = 13  # 13 drawn horizontal lines (no top/bottom outer border)

STRIP_WIDTH = 20
MIN_LINE_GAP = 50
TARGET_SIZE = (64, 64)
CELL_PAD = 30
EDGE_CROP_TOP = 20
EDGE_CROP_BOTTOM = 20
EDGE_CROP_LEFT = 80
EDGE_CROP_RIGHT = 20

# Set True to output one strip image per row + per col for easy validation.
# Set False to output individual 64x64 PNGs for training.
DEBUG_LINES_ONLY = False

SRC_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "Test_Data", "side_mount", "images"
)
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "Test_Data", "side_mount", "split_images"
)


def find_images(src_dir):
    """Return sorted list of (stem, filename) for all .jpg files directly in src_dir."""
    files = []
    for fname in sorted(os.listdir(src_dir)):
        if fname.lower().endswith(".jpg") and os.path.isfile(
            os.path.join(src_dir, fname)
        ):
            stem = os.path.splitext(fname)[0]
            files.append((stem, fname))
    return files


def find_lines(gray, axis, n_lines):
    """
    Find n_lines grid lines along the given axis.

    axis=0: vertical lines   -- mean darkness per column (averages over rows)
    axis=1: horizontal lines -- mean darkness per row (averages over columns)

    Grid lines score high because they are dark across the full image extent.
    Returns a sorted list of n_lines pixel coordinates.
    """
    dark = 255 - gray.astype(np.float32)
    profile = dark.mean(axis=0) if axis == 0 else dark.mean(axis=1)

    kernel = np.ones(STRIP_WIDTH) / STRIP_WIDTH
    profile = np.convolve(profile, kernel, mode="same")

    threshold = profile.mean() + 0.5 * profile.std()
    above = np.where(profile > threshold)[0]
    if len(above) == 0:
        raise RuntimeError(f"No dark lines found (axis={axis})")

    clusters = []
    run = [above[0]]
    for idx in above[1:]:
        if idx - run[-1] <= STRIP_WIDTH:
            run.append(idx)
        else:
            clusters.append(run)
            run = [idx]
    clusters.append(run)

    peaks = []
    for c in clusters:
        best = c[int(np.argmax(profile[c]))]
        peaks.append((float(profile[best]), best))
    peaks.sort(key=lambda x: -x[0])

    print(
        f"    axis={axis}: {len(peaks)} candidates (need {n_lines}), "
        f"top scores: {[(round(v, 1), p) for v, p in peaks[: max(n_lines, 8)]]}"
    )

    selected = []
    for score, pos in peaks:
        if all(abs(pos - s) >= MIN_LINE_GAP for s in selected):
            selected.append(pos)
        if len(selected) == n_lines:
            break

    if len(selected) < n_lines:
        print(
            f"    WARNING: Only {len(selected)} well-separated lines found, need {n_lines} (axis={axis}). "
            f"Will attempt fallback."
        )
        # Lower threshold and retry
        threshold2 = profile.mean()
        above2 = np.where(profile > threshold2)[0]
        if len(above2) > 0:
            clusters2 = []
            run2 = [above2[0]]
            for idx in above2[1:]:
                if idx - run2[-1] <= STRIP_WIDTH:
                    run2.append(idx)
                else:
                    clusters2.append(run2)
                    run2 = [idx]
            clusters2.append(run2)
            peaks2 = []
            for c in clusters2:
                best = c[int(np.argmax(profile[c]))]
                peaks2.append((float(profile[best]), best))
            peaks2.sort(key=lambda x: -x[0])
            selected = []
            for score, pos in peaks2:
                if all(abs(pos - s) >= MIN_LINE_GAP for s in selected):
                    selected.append(pos)
                if len(selected) == n_lines:
                    break

    return sorted(selected)


def validate_line_spacing(lines, n_lines, label):
    """Check if detected lines have roughly uniform spacing.
    Returns True if OK, False if spacing is too irregular."""
    if len(lines) < 2:
        return False
    gaps = [lines[i + 1] - lines[i] for i in range(len(lines) - 1)]
    median_gap = float(np.median(gaps))
    if median_gap == 0:
        return False
    for g in gaps:
        if abs(g - median_gap) / median_gap > 0.35:
            print(f"  WARNING: irregular {label} spacing: gaps={gaps}, median={median_gap:.0f}")
            return False
    return True


def evenly_spaced_lines(first, last, n_lines):
    """Generate n_lines evenly spaced positions from first to last."""
    return [int(round(first + i * (last - first) / (n_lines - 1))) for i in range(n_lines)]


def detect_vlines_in_band(gray, y_start, y_end, n_lines):
    """Detect vertical lines in a horizontal band. Returns list or None."""
    band = gray[y_start:y_end, :]
    if band.shape[0] < 10:
        return None
    try:
        raw = find_lines(band, axis=0, n_lines=n_lines)
    except RuntimeError:
        return None
    if len(raw) < n_lines:
        return None
    return raw


def interpolate_vlines(top_vlines, bot_vlines, top_y, bot_y, target_y):
    """Linearly interpolate vline x-positions between top and bottom bands."""
    if bot_y == top_y:
        return top_vlines
    t = (target_y - top_y) / (bot_y - top_y)
    return [int(round(tv + t * (bv - tv))) for tv, bv in zip(top_vlines, bot_vlines)]


def crop_page(stem, img_filename):
    src_path = os.path.join(SRC_DIR, img_filename)
    img = cv2.imread(src_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    gray_inner = gray[
        EDGE_CROP_TOP : h - EDGE_CROP_BOTTOM, EDGE_CROP_LEFT : w - EDGE_CROP_RIGHT
    ]

    # Detect horizontal lines globally (these are reliable on all pages)
    raw_hlines = find_lines(gray_inner, axis=1, n_lines=N_HLINES)
    hlines = [y + EDGE_CROP_TOP for y in raw_hlines]

    if len(hlines) < N_HLINES or not validate_line_spacing(hlines, N_HLINES, "hline"):
        print(f"  Falling back to evenly spaced hlines ({len(hlines)} detected, need {N_HLINES})")
        first_h = hlines[0] if hlines else EDGE_CROP_TOP
        last_h = hlines[-1] if hlines else h - EDGE_CROP_BOTTOM
        hlines = evenly_spaced_lines(first_h, last_h, N_HLINES)

    print(f"  hlines ({len(hlines)}): {hlines}")

    row_spacing = (hlines[-1] - hlines[0]) / (len(hlines) - 1)
    extrap_top = int(round(hlines[0] - row_spacing))
    extrap_bot = int(round(hlines[-1] + row_spacing))
    row_bounds = (
        [max(0, extrap_top)]
        + [int(round(y)) for y in hlines]
        + [min(h - 1, extrap_bot)]
    )
    # 1 + 13 + 1 = 15 entries -> 14 row intervals

    print(f"  row_spacing={row_spacing:.1f}px")

    # Detect vertical lines in top and bottom halves to model tilt.
    # Use wide bands (multiple rows) for reliable detection.
    inner_left = EDGE_CROP_LEFT
    inner_right = w - EDGE_CROP_RIGHT

    mid_hline = hlines[len(hlines) // 2]  # middle horizontal line
    top_band_y = (EDGE_CROP_TOP, mid_hline)
    bot_band_y = (mid_hline, h - EDGE_CROP_BOTTOM)

    top_vl = detect_vlines_in_band(gray[:, inner_left:inner_right],
                                   top_band_y[0], top_band_y[1], N_VLINES)
    bot_vl = detect_vlines_in_band(gray[:, inner_left:inner_right],
                                   bot_band_y[0], bot_band_y[1], N_VLINES)

    # Also get global vlines as ultimate fallback
    raw_vlines_global = find_lines(gray_inner, axis=0, n_lines=N_VLINES)
    global_vlines = [x + EDGE_CROP_LEFT for x in raw_vlines_global]
    if len(global_vlines) < N_VLINES or not validate_line_spacing(global_vlines, N_VLINES, "vline"):
        print(f"  Global vlines irregular, using evenly spaced fallback")
        first_v = global_vlines[0] if global_vlines else EDGE_CROP_LEFT
        last_v = global_vlines[-1] if global_vlines else w - EDGE_CROP_RIGHT
        global_vlines = evenly_spaced_lines(first_v, last_v, N_VLINES)

    # Offset to full-image coords
    if top_vl is not None:
        top_vl = [x + inner_left for x in top_vl]
    if bot_vl is not None:
        bot_vl = [x + inner_left for x in bot_vl]

    # Validate and fall back as needed
    use_interpolation = True
    if top_vl is None or not validate_line_spacing(top_vl, N_VLINES, "top-band vline"):
        top_vl = global_vlines
        use_interpolation = False
    if bot_vl is None or not validate_line_spacing(bot_vl, N_VLINES, "bot-band vline"):
        bot_vl = global_vlines
        use_interpolation = False

    top_center_y = (top_band_y[0] + top_band_y[1]) / 2
    bot_center_y = (bot_band_y[0] + bot_band_y[1]) / 2

    if use_interpolation:
        shift = [b - t for t, b in zip(top_vl, bot_vl)]
        print(f"  Tilt correction: top vlines={top_vl}")
        print(f"  Tilt correction: bot vlines={bot_vl}")
        print(f"  Tilt correction: shift top->bot = {shift}")

    page_out = os.path.join(OUT_DIR, stem)
    os.makedirs(page_out, exist_ok=True)

    csv_box_idx = 0
    written = 0
    col_cells = [[] for _ in range(COLS)]

    for img_row in range(ROWS):
        row_cells = []
        row_center_y = (row_bounds[img_row] + row_bounds[img_row + 1]) / 2

        if use_interpolation:
            vlines = interpolate_vlines(top_vl, bot_vl,
                                        top_center_y, bot_center_y, row_center_y)
        else:
            vlines = global_vlines

        for img_col in range(COLS):
            y_top = row_bounds[img_row] + CELL_PAD
            y_bot = row_bounds[img_row + 1] - CELL_PAD
            x_left = vlines[img_col] + CELL_PAD
            x_right = vlines[img_col + 1] - CELL_PAD

            y_top = max(0, y_top)
            y_bot = min(h - 1, y_bot)
            x_left = max(0, x_left)
            x_right = min(w - 1, x_right)

            if y_bot <= y_top or x_right <= x_left:
                print(f"  WARNING: empty crop at row={img_row} col={img_col}")
                placeholder = np.full(TARGET_SIZE, 200, dtype=np.uint8)
                if DEBUG_LINES_ONLY:
                    row_cells.append(placeholder)
                    col_cells[img_col].append(placeholder)
                csv_box_idx += 1
                continue

            cell = gray[y_top:y_bot, x_left:x_right]
            cell_resized = cv2.resize(cell, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            if DEBUG_LINES_ONLY:
                row_cells.append(cell_resized)
                col_cells[img_col].append(cell_resized)
            else:
                csv_row = csv_box_idx // COLS
                csv_col = csv_box_idx % COLS
                out_path = os.path.join(
                    page_out, f"box_{csv_row:02d}_{csv_col:02d}.png"
                )
                cv2.imwrite(out_path, cell_resized)

            csv_box_idx += 1
            written += 1

        if DEBUG_LINES_ONLY and row_cells:
            cv2.imwrite(
                os.path.join(page_out, f"row_{img_row:02d}.png"),
                np.concatenate(row_cells, axis=1),
            )

    if DEBUG_LINES_ONLY:
        for img_col, cells in enumerate(col_cells):
            if cells:
                cv2.imwrite(
                    os.path.join(page_out, f"col_{img_col:02d}.png"),
                    np.concatenate(cells, axis=0),
                )

    print(
        f"  Wrote {written} cells" + (" [DEBUG strip mode]" if DEBUG_LINES_ONLY else "")
    )


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    images = find_images(SRC_DIR)
    if not images:
        print(f"No .jpg files found in {SRC_DIR}")
    for stem, filename in images:
        print(f"\n{filename}:")
        crop_page(stem, filename)
    print("\nDone.")
