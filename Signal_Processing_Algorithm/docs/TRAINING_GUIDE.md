# Training Guide: Accelerometer-to-Handwriting Model

This guide explains the exact step-by-step process to train the handwriting recognition model on new data.

## Prerequisites

- Python 3.x installed
- Required packages: `pip install -r requirements.txt`
- SVG files with handwritten letters
- Corresponding CSV files with accelerometer data

---

## Step-by-Step Training Process

### Step 1: Prepare SVG Files

**Starting Point:** You have an SVG file containing multiple handwritten letters with labels.

**Example:** `a1-11.svg` contains letters numbered 1-11

#### 1.1: Separate Individual Letters

Run the interactive SVG separator:

```bash
cd Signal_Processing_Algorithm/scripts
python interactive_svg_separator.py
```

**What to do:**
1. The UI will load your SVG file
2. Click and drag to draw a box around ONE letter
3. Press Space to save the selected letter
4. Repeat for all letters in the SVG
5. Close the window when done

**Output:** Individual SVG files named `a_write_1.svg`, `a_write_2.svg`, etc.

**Important:** 
- Don't include the number labels in your selection - only the actual letter strokes
- Make sure each selection contains exactly one complete letter

---

### Step 2: Convert SVG to PNG

Once you have individual SVG files, convert them to PNG images for training:

```bash
python convert_svgs_to_pngs.py
```

**What this does:**
- Reads all `a_write_*.svg` files
- Converts each to a 64x64 pixel grayscale PNG
- Saves to `a_write_pngs/` folder

**Output:** PNG files (`a_write_1.png`, `a_write_2.png`, etc.)

---

### Step 3: Prepare CSV Accelerometer Data

**Requirements:**
- CSV files with columns: `acc_x[mg]`, `acc_y[mg]`, `acc_z[mg]`, `gyro_x[mdps]`, `gyro_y[mdps]`, `gyro_z[mdps]`
- First row should be header/metadata (will be skipped automatically)
- File naming: Match your SVG/PNG files (e.g., `a_w_1.csv` corresponds to `a_write_1.png`)

**Expected format:**
```
# STEVAL-MKI229A (LSM6DSO16IS)
acc_x[mg],acc_y[mg],acc_z[mg],gyro_x[mdps],gyro_y[mdps],gyro_z[mdps]
-123.4,456.7,-89.0,12.3,-45.6,78.9
...
```

Place CSV files in: `Test_Data/alphabet_write/`

---

### Step 4: Train the Model

Run the training script:

```bash
python train_handwriting_improved.py
```

**What happens during training:**

1. **Data Loading:** Loads CSV and PNG pairs
2. **Preprocessing:**
   - Applies high-pass filter to remove gravity
   - Computes acceleration magnitude (7th feature)
   - Normalizes data using StandardScaler
   - Pads/truncates sequences to 200 timesteps
3. **Model Architecture:**
   - Input: (200 timesteps, 7 features)
   - Bidirectional LSTM layers (128 and 64 units)
   - Dense layers
   - CNN decoder with upsampling (16x16 -> 32x32 -> 64x64)
   - Output: 64x64 pixel image
4. **Training:**
   - 150 epochs
   - Batch size: 4
   - Combined loss: 30% MSE + 70% BCE (for sharper edges)
   - Early stopping with patience=10

**Expected duration:** ~10-30 minutes depending on dataset size

**Output files:**
- `best_handwriting_model_improved.keras` (trained model)
- `scaler_improved.pkl` (data normalization scaler)
- `training_history_improved.png` (loss/accuracy plots)

---

### Step 5: Test the Model

Test your trained model on new data:

```bash
python test_on_all_alphabet_v2.py
```

**What to do:**
1. Script will ask you to select a threshold (0.1, 0.2, 0.3, or 0.4)
2. Choose **0.2** for best results currently
3. View the prediction grid showing all outputs

**Troubleshooting:**
- If outputs are too faint: Use lower threshold (0.1 or 0.2)
- If outputs are too noisy: Use higher threshold (0.3 or 0.4)

---

## File Organization

After training, your files should be organized as:

```
Signal_Processing_Algorithm/
  ├── scripts/
  │   ├── interactive_svg_separator.py
  │   ├── convert_svgs_to_pngs.py
  │   ├── train_handwriting_improved.py
  │   └── test_on_all_alphabet_v2.py
  │
  ├── models/
  │   ├── best_handwriting_model_improved.keras
  │   └── scaler_improved.pkl
  │
  ├── outputs/
  │   └── training_history_improved.png
  │
  └── a_write_pngs/
      ├── a_write_1.png
      ├── a_write_2.png
      └── ...
```

---

## Training Tips

### For Better Model Quality:

1. **More Data is Better:**
   - Current proof-of-concept: ~12 samples
   - Recommended: 50-100 samples per letter
   - Expected improvement: 5-10x better output quality

2. **Data Collection Best Practices:**
   - Vary writing speeds
   - Different writing styles/people
   - Consistent pen orientation during recording
   - Clean, clear letter strokes in SVG

3. **Data Augmentation (Future):**
   - Time warping for speed variation
   - Rotation for orientation changes
   - Noise injection for robustness

### Common Issues:

**Problem:** "Blurry" or averaged-looking outputs
- **Cause:** Too few training samples
- **Solution:** Collect 50+ samples per letter

**Problem:** Blank predictions
- **Cause:** Threshold too high
- **Solution:** Use threshold 0.2 instead of 0.3

**Problem:** Model not learning (loss not decreasing)
- **Cause:** Data mismatch or incorrect CSV format
- **Solution:** Verify CSV columns match expected format

---

## Expanding to Full Alphabet (A-Z)

To train on all 26 letters:

1. Repeat Steps 1-2 for each letter (b, c, d, etc.)
2. Modify `train_handwriting_improved.py` to load all letters
3. Update model output to 26-class classification OR
4. Train separate model per letter (simpler for now)

**Recommended approach:** Train one letter at a time initially, then merge models once pipeline is proven.

---

## Next Steps

After successful training on letter 'a':

1. Collect data for letters b-z
2. Increase training samples to 50-100 per letter
3. Implement real-time inference
4. Optimize for embedded deployment
5. Build user interface/demo application

---

## Questions or Issues?

If you encounter problems:
1. Check CSV file format matches expected columns
2. Verify PNG files are 64x64 grayscale
3. Ensure file naming is consistent (a_w_X.csv <-> a_write_X.png)
4. Review training logs for specific error messages
