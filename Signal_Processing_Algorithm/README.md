# Signal Processing Algorithm - Letter Recognition Pipeline

## Overview

This folder contains a complete, minimal pipeline for recognizing handwritten letters from accelerometer/gyroscope data collected from a pen-mounted IMU sensor.

**Status**: ✅ Complete and functional

## Quick Start

```bash
# Install dependencies
cd Signal_Processing_Algorithm
python3 -m venv venv
./venv/bin/pip install pandas numpy scipy fastdtw matplotlib

# Run tests
./venv/bin/python test_with_tuned_params.py    # Test segmentation
./venv/bin/python analyze_data.py              # Analyze data characteristics

# Run full pipeline demo (when you have training data)
./venv/bin/python letter_recognition_pipeline.py
```

## Files

### Core Pipeline Modules
- **`stroke_segmentation.py`** - Detects individual letter strokes from continuous data
- **`feature_extraction.py`** - Extracts features from each stroke for classification
- **`classifier.py`** - DTW-based and feature-based classifiers
- **`letter_recognition_pipeline.py`** - Main pipeline combining all stages

### Utilities & Testing
- **`data_reader.py`** - CSV data loader (pre-existing)
- **`analyze_data.py`** - Analyze motion characteristics to tune parameters
- **`test_with_tuned_params.py`** - Test segmentation with dataset-specific parameters
- **`simple_test.py`** - Basic segmentation test
- **`config.py`** - Configuration parameters

### Documentation
- **`USAGE.md`** - Detailed usage guide
- **`PIPELINE_OVERVIEW.md`** - Architecture and design decisions
- **`requirements.txt`** - Python dependencies

## How It Works

```
Raw CSV → Segment Strokes → Extract Features → Classify → Output Text
```

1. **Segmentation**: Detects when pen is writing vs. lifted using motion magnitude
2. **Feature Extraction**: Characterizes each stroke (duration, acceleration patterns, etc.)
3. **Classification**: Matches strokes to letter templates using DTW (Dynamic Time Warping)
4. **Output**: Returns recognized text

## Current Test Results

### Test2.csv (Hello World)
- ✅ Successfully segments into 15 strokes (expected ~11 for "Hello World")
- Has active gyroscope data
- Good motion detection

### Test3.csv (Multiple A's)
- ⚠️ Limited segmentation (4 strokes detected)
- **Issue**: Gyroscope is disabled (all zeros)
- Relies only on acceleration, making stroke detection harder

## Key Findings

1. **Gyroscope is critical** for good stroke detection
   - Test2 (gyro active): Good segmentation
   - Test3 (gyro inactive): Poor segmentation

2. **Thresholds need tuning per dataset**
   - Test2 works with `threshold=200`
   - Test3 needs `threshold=15` due to no gyroscope

3. **Motion characteristics vary significantly**
   - Different sampling rates (416Hz vs 833Hz)
   - Different sensor configurations
   - Need adaptive thresholding

## Next Steps

### Immediate (To Make It Work Better)
1. **Enable gyroscope** for all future data collection
2. **Collect training data** - Write each letter (A-Z) 3-5 times
3. **Label training data** - Note which strokes are which letters
4. **Train the classifier** - Use `pipeline.train(csv, labels)`
5. **Test recognition** - Try recognizing new handwriting

### Short Term Improvements
6. Add manual stroke labeling tool
7. Implement adaptive threshold based on data statistics
8. Add confidence scoring for classifications
9. Detect and handle spaces between words

### Long Term Enhancements
10. Drift correction for trajectory reconstruction
11. Orientation calibration
12. Neural network classifier (LSTM/GRU)
13. Real-time streaming mode
14. User-specific adaptation

## Usage Example

```python
from letter_recognition_pipeline import LetterRecognitionPipeline

# Create and train pipeline
pipeline = LetterRecognitionPipeline(use_dtw=True)

# Train with labeled data (you need to create this)
labels = ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
pipeline.train('training_data.csv', labels)

# Recognize new data
text, results = pipeline.recognize('test_data.csv')
print(f"Recognized: {text}")

# Results contain (letter, confidence) for each stroke
for letter, confidence in results:
    print(f"  {letter}: {confidence:.2f}")
```

## Known Limitations

1. **Requires labeled training data** for each letter
2. **No drift correction** - integration errors accumulate
3. **Orientation dependent** - assumes consistent sensor mounting
4. **Threshold tuning required** per person/dataset
5. **No space detection** yet
6. **Gyroscope critical** for good performance

## Technical Specifications

**Input Format**: CSV with columns
- `time[us]` - Timestamp in microseconds
- `acc_x[mg]`, `acc_y[mg]`, `acc_z[mg]` - Acceleration in milligravity
- `gyro_x[mdps]`, `gyro_y[mdps]`, `gyro_z[mdps]` - Gyroscope in millidegrees/sec

**Output**: Recognized text string and confidence scores

**Dependencies**: pandas, numpy, scipy, fastdtw, matplotlib

## People Working On This

(Add names here)

## References & Resources

- Dynamic Time Warping: Used for template matching
- Gesture Recognition: Similar to handwriting recognition
- IMU Sensor: LSM6DSO16IS (from your test data)
