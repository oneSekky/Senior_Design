import cv2
import numpy as np
import os

# Load page image
img_path = r"C:\Users\sekan\Documents\Senior_Design\Test_Data\side_mount\images\box-page-1-2sec_1.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"Image size: {w}x{h}")

EDGE_CROP_TOP    = 20
EDGE_CROP_BOTTOM = 20
EDGE_CROP_LEFT   = 80
EDGE_CROP_RIGHT  = 20
STRIP_WIDTH = 20
MIN_LINE_GAP = 50
N_VLINES = 11
N_HLINES = 13

gray_inner = gray[EDGE_CROP_TOP:h-EDGE_CROP_BOTTOM, EDGE_CROP_LEFT:w-EDGE_CROP_RIGHT]

def find_lines(gray, axis, n_lines):
    dark = (255 - gray.astype(np.float32))
    profile = dark.mean(axis=0) if axis == 0 else dark.mean(axis=1)
    kernel = np.ones(STRIP_WIDTH) / STRIP_WIDTH
    profile = np.convolve(profile, kernel, mode='same')
    threshold = profile.mean() + 0.5 * profile.std()
    above = np.where(profile > threshold)[0]
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
    selected = []
    for score, pos in peaks:
        if all(abs(pos - s) >= MIN_LINE_GAP for s in selected):
            selected.append(pos)
        if len(selected) == n_lines:
            break
    return sorted(selected)

vlines = [x + EDGE_CROP_LEFT for x in find_lines(gray_inner, axis=0, n_lines=N_VLINES)]
hlines = [y + EDGE_CROP_TOP  for y in find_lines(gray_inner, axis=1, n_lines=N_HLINES)]

print(f"vlines: {vlines}")
print(f"hlines: {hlines}")

# Cell widths (between consecutive vlines)
cell_widths = [vlines[i+1] - vlines[i] for i in range(len(vlines)-1)]
print(f"Cell widths (px): {cell_widths}")
print(f"Mean cell width: {np.mean(cell_widths):.1f}px")

row_spacing = (hlines[-1] - hlines[0]) / (len(hlines) - 1)
extrap_top = int(round(hlines[0] - row_spacing))
extrap_bot = int(round(hlines[-1] + row_spacing))
row_bounds = [max(0, extrap_top)] + [int(round(y)) for y in hlines] + [min(h-1, extrap_bot)]
cell_heights = [row_bounds[i+1] - row_bounds[i] for i in range(len(row_bounds)-1)]
print(f"Cell heights (px): {cell_heights}")
print(f"Mean cell height: {np.mean(cell_heights):.1f}px")

# How many px does CELL_PAD=4 take off each side?
# After pad, cell is (width - 2*CELL_PAD) x (height - 2*CELL_PAD)
for pad in [4, 8, 10, 12, 15]:
    eff_w = np.mean(cell_widths) - 2*pad
    eff_h = np.mean(cell_heights) - 2*pad
    pct_w = 100 * eff_w / np.mean(cell_widths)
    pct_h = 100 * eff_h / np.mean(cell_heights)
    print(f"CELL_PAD={pad:2d}: effective cell {eff_w:.0f}x{eff_h:.0f}px ({pct_w:.0f}% x {pct_h:.0f}% of original)")

# Look at a specific vline — how wide is the dark line in pixels?
print("\nVline darkness profiles (showing ±10px around each vline):")
for vl in vlines[:3]:
    region = gray[:, max(0,vl-10):vl+10]
    col_dark = 255 - region.mean(axis=0)
    print(f"  vline at x={vl}: darkness = {[round(v,0) for v in col_dark]}")
