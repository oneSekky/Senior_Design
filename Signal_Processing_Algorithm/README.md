# Signal Processing Algorithm

## Overview

This folder contains two main pipelines for processing accelerometer/gyroscope data from a pen-mounted IMU sensor:

1. **Trajectory Reconstruction** - Convert IMU data into actual pen stroke coordinates (X,Y paths)
2. **Letter Recognition** - Classify strokes into recognized letters

**Status**: ✅ Complete and functional

## Files

- **`data_reader.py`** - CSV data loader
- **`stroke_segmentation.py`** - Detects individual letter strokes
- **`trajectory_reconstruction.py`** - Converts IMU data to X,Y coordinates
- **`feature_extraction.py`** - Extracts features for classification
- **`classifier.py`** - DTW-based classifier
- **`letter_recognition_pipeline.py`** - Complete letter recognition pipeline

## 1. Trajectory Reconstruction

Converts accelerometer data into actual pen strokes (X,Y coordinates).

### How It Works

```
IMU Data → Remove Gravity → Filter Noise → Double Integration → X,Y Trajectory
```

1. Removes gravity component from acceleration
2. Applies low-pass filter to reduce noise
3. Integrates acceleration → velocity → position
4. Outputs 2D (X,Y) or 3D coordinates

### Usage

```python
from data_reader import read_accelerometer_csv
from trajectory_reconstruction import reconstruct_trajectory, save_trajectory_svg

# Load data
df = read_accelerometer_csv('path/to/data.csv')

# Reconstruct trajectory
trajectory, velocity, time = reconstruct_trajectory(df, cutoff_freq=3.0)

# trajectory is now an array of (X, Y) coordinates in meters

# Save as SVG for graphics software
save_trajectory_svg(trajectory, 'output.svg')

# Or export as JSON for web apps
export_trajectory_json(trajectory, 'output.json')
```

### For Individual Strokes

```python
from stroke_segmentation import segment_strokes
from trajectory_reconstruction import reconstruct_stroke_trajectory

# Segment into individual letters
strokes = segment_strokes(df, threshold=200, min_samples=20)

# Reconstruct each stroke
for i, stroke in enumerate(strokes):
    traj = reconstruct_stroke_trajectory(stroke['data'])
    save_trajectory_svg(traj, f'letter_{i}.svg')
```

### Output Formats

- **NumPy array**: Direct use in Python (X,Y coordinates)
- **SVG**: Vector graphics for design software (Illustrator, Inkscape, etc.)
- **JSON**: For web applications and JavaScript

## 2. Letter Recognition

Classifies strokes into recognized letters using DTW template matching.

### How It Works

```
CSV File → Segment Strokes → Extract Features → DTW Matching → Letters
```

### Usage

```python
from letter_recognition_pipeline import LetterRecognitionPipeline

# Create pipeline
pipeline = LetterRecognitionPipeline(use_dtw=True)

# Train with labeled data
labels = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
pipeline.train('training_data.csv', labels)

# Recognize new data
text, results = pipeline.recognize('test_data.csv')
print(f"Recognized: {text}")
```

## Key Challenges & Solutions

### Challenge: Integration Drift
**Problem**: Double integration causes position errors to accumulate  
**Solution**: Zero-velocity updates (ZUPT) reset velocity during stationary periods

### Challenge: Gyroscope Data
**Problem**: Test3 has gyroscope disabled (all zeros)  
**Impact**: Much harder to detect stroke boundaries  
**Solution**: Enable gyroscope for all data collection

### Challenge: Stroke Segmentation
**Problem**: Different datasets need different thresholds  
**Solution**: Tune parameters per dataset or use adaptive thresholding

## Test Results

**Test2.csv (Hello World)** - With gyroscope:
- ✅ Good stroke segmentation (15-18 strokes)
- ✅ Trajectory reconstruction works
- Range: ~0.01-0.9 meters

**Test3.csv (Multiple A's)** - Without gyroscope:
- ⚠️ Poor segmentation (only 4 strokes detected)
- ✅ Trajectory still reconstructs
- **Recommendation**: Re-collect with gyroscope enabled

## Parameters to Tune

### Stroke Segmentation
```python
segment_strokes(df, 
    threshold=200,    # Motion intensity (higher = less sensitive)
    min_samples=20,   # Minimum stroke length
    merge_gap=100     # Merge nearby strokes
)
```

### Trajectory Reconstruction
```python
reconstruct_trajectory(df,
    cutoff_freq=3.0,  # Low-pass filter (lower = smoother, more lag)
    remove_z=True     # 2D vs 3D output
)
```

## Next Steps

### For Better Trajectory Reconstruction
1. Implement advanced drift correction (Kalman filter, UWB fusion)
2. Add orientation calibration
3. Use gyroscope for rotation tracking
4. Implement complementary filter for sensor fusion

### For Letter Recognition
1. Collect full alphabet training data (A-Z, multiple samples each)
2. Label training data properly
3. Try neural network (LSTM/GRU) instead of DTW
4. Add word/space detection

## Dependencies

```bash
pip install pandas numpy scipy fastdtw matplotlib
```

## Technical Notes

**Input Format**: CSV with columns:
- `time[us]` - Timestamp in microseconds
- `acc_x[mg]`, `acc_y[mg]`, `acc_z[mg]` - Acceleration in milligravity
- `gyro_x[mdps]`, `gyro_y[mdps]`, `gyro_z[mdps]` - Gyroscope (optional but recommended)

**Sensor**: LSM6DSO16IS (from your test data)

**Sampling Rates**: 416Hz or 833Hz (both work)

**Critical**: Gyroscope data is essential for good stroke detection!
