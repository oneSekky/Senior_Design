# Side-Mount Training Pipeline — Instructions

End-to-end guide for going from raw sensor CSVs and page scan photos to a trained handwriting generation model.

---

## Overview

The pipeline has three stages:

```
Raw CSVs + Page Photos
        |
        v
  split_page_csvs.py        -- splits each page CSV into one file per letter box
  split_page_images.py      -- crops each page photo into one PNG per letter box
        |
        v
  train_side_mount.py       -- trains the CNN on the paired CSV/PNG data
        |
        v
  models/best_side_mount_model.keras
```

---

## Prerequisites

Install dependencies:

```
pip install tensorflow numpy pandas scipy scikit-learn pillow opencv-python matplotlib
```

All scripts are in `Signal_Processing_Algorithm/scripts/`. Run them from the repo root or from within that folder — paths are relative to the script location.

---

## Step 1: Collect Data

### Sensor CSV

- Record with the STEVAL-MKI229A (LSM6DSO16IS) at **104 Hz**
- Write letters continuously across the entire boxed page, one letter per box, left-to-right, top-to-bottom
- Each box should take exactly **2 seconds** (the sensor is set to record at a fixed rate)
- The resulting CSV should contain approximately **140 × 208 = 29,120 rows** for a full 14-row × 10-col page
- Place the raw CSV in:
  ```
  Test_Data/side_mount/csvs/<name>.csv
  ```
  Example: `box-page-3-2sec.csv`

### Page Photo

- Photograph or scan the completed page after writing
- The photo should be a `.jpg` taken roughly overhead (portrait orientation)
- Place it in:
  ```
  Test_Data/side_mount/images/<name>_1.jpg
  ```
  Example: `box-page-3-2sec_1.jpg`

> The `_1` suffix on the image filename is a convention from the camera app. The CSV and image share the same base name (`box-page-3-2sec`); the scripts handle the suffix mismatch automatically.

---

## Step 2: Split the CSV

```
python Signal_Processing_Algorithm/scripts/split_page_csvs.py
```

- Reads every `.csv` directly in `Test_Data/side_mount/csvs/` (ignores subdirectories)
- Splits each file into 208-sample chunks (one per box), resets timestamps to 0
- Outputs to `Test_Data/side_mount/split_csvs/<stem>/box_RR_CC.csv`

Expected output per page:
```
box-page-3-2sec.csv:
  29120 rows (need 208 per box, 140 boxes)
  Wrote 140 chunks to .../split_csvs/box-page-3-2sec
```

If the page ran short (sensor stopped early), trailing boxes are silently dropped. Fewer than ~100 boxes usable is a sign of a bad recording.

---

## Step 3: Split the Page Image

```
python Signal_Processing_Algorithm/scripts/split_page_images.py
```

- Reads every `.jpg` directly in `Test_Data/side_mount/images/`
- Auto-detects the grid lines from the image using column/row darkness profiles
- Crops each of the 140 cells to a 64×64 grayscale PNG
- Outputs to `Test_Data/side_mount/split_images/<stem>/box_RR_CC.png`

Expected output per page:
```
box-page-3-2sec_1.jpg:
  vlines (11): [...]
  hlines (13): [...]
  Wrote 140 cells
```

### Debugging image splits

If the crop looks wrong, enable debug mode in the script:

```python
# split_page_images.py line 46
DEBUG_LINES_ONLY = True
```

Re-run — instead of individual box PNGs it will write one strip image per row (`row_00.png`, etc.) and one per column (`col_00.png`, etc.) so you can verify the grid was detected correctly. Set back to `False` before training.

### Common image issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: Only N lines found, need 11` | Photo badly skewed, overexposed, or left margin too dark | Re-photograph; check `EDGE_CROP_LEFT` constant |
| Boxes shifted one column right/left | Grid detection grabbed the page margin as a vertical line | Increase `EDGE_CROP_LEFT` in the script |
| Rows cropped incorrectly | First/last row extrapolation off | Check spacing; ensure all 13 internal h-lines are visible |

---

## Step 4: Move Bad Data (if needed)

If a page has unusable data (bad recording, wrong timing, misaligned image), move both the CSV and image into `bad_data/` subfolders so the scripts skip them:

```
Test_Data/side_mount/csvs/bad_data/<name>.csv
Test_Data/side_mount/images/bad_data/<name>_1.jpg
```

The scripts only process files directly in the `csvs/` and `images/` folders, not in subdirectories.

After moving bad data you do **not** need to re-run the split scripts unless you want to regenerate — they output to separate folders and won't touch bad_data/.

---

## Step 5: Train the Model

```
python Signal_Processing_Algorithm/scripts/train_side_mount.py
```

The script:
1. Scans `split_csvs/` for page folders and matches each to its corresponding `split_images/` folder
2. Loads each CSV/PNG pair as one training sample (208-timestep IMU → 64×64 image)
3. Applies a high-pass filter to remove gravity, computes magnitude as a 7th feature
4. Filters out near-blank images (< 0.5% stroke density)
5. Normalizes features with `StandardScaler`
6. Augments the training set 8× with noise, amplitude scaling, time warping (IMU) and translation, rotation, elastic deformation (image)
7. Trains a **1D-CNN encoder → temporal attention pooling → Conv2D decoder** with 85/15 train/test split and **focal loss** (γ=2, α=0.75)
8. Saves outputs to `Signal_Processing_Algorithm/models/`:
   - `best_side_mount_model.keras` — best checkpoint by val_loss
   - `scaler_side_mount.pkl` — scaler (required for inference)
   - `training_history_side_mount.png` — focal loss / MAE curves
   - `predictions_side_mount.png` — ground truth vs predicted (raw, thresh >0.2, thresh >0.4) for 8 test samples

### Model architecture

```
Input (150, 7)
  → Conv1D stack with strided downsampling → (19, 128)
  → Learned temporal attention → (128,)
  → Dense → Reshape (8, 8, 32)
  → Three ×2 UpSampling2D + Conv2D stages → (64, 64, 1)
```

~720k parameters total.

### What to watch during training

- `val_loss` should trend downward over the first 20–40 epochs
- `ReduceLROnPlateau` halving the LR once or twice is normal and healthy
- Training stops early (`EarlyStopping`, patience=40) if val_loss stops improving
- If val_loss is `NaN` from the start, a CSV likely contains bad/zero data
- If `pred_max < 0.15` is reported at the end, the model did not learn stroke structure — check `ACTIVITY_THRESHOLD` and image quality

---

## File Layout Reference

```
Test_Data/side_mount/
  csvs/
    box-page-1-2sec.csv          ← raw page recording
    box-page-2-2sec.csv
    bad_data/
      box-page-4-2sec.csv        ← excluded from training
  images/
    box-page-1-2sec_1.jpg        ← page scan
    box-page-2-2sec_1.jpg
    bad_data/
      box-page-4-2sec_1.jpg
  split_csvs/
    box-page-1-2sec/
      box_00_00.csv  ...  box_13_09.csv
    box-page-2-2sec/
      box_00_00.csv  ...  box_13_09.csv
  split_images/
    box-page-1-2sec_1/
      box_00_00.png  ...  box_13_09.png
    box-page-2-2sec_1/
      box_00_00.png  ...  box_13_09.png

Signal_Processing_Algorithm/
  scripts/
    split_page_csvs.py
    split_page_images.py
    train_side_mount.py
  models/
    best_side_mount_model.keras
    scaler_side_mount.pkl
    training_history_side_mount.png
    predictions_side_mount.png
```

---

## Adding More Training Data

To add a new page, repeat Steps 1–3 for the new CSV/image pair. You do not need to re-split existing pages — the split scripts wipe and rebuild their output directories each run, so just let them re-process everything, or manually add only the new page's split folder.

Then re-run Step 5. The training script always reads all available split data.
