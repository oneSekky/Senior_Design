# Signal Processing Algorithm

## Overview

Signal processing algorithm for letter/handwriting data using accelerometer and gyroscope data.

**Status**: WIP

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
