"""
Stroke Segmentation - Detect individual letter strokes from accelerometer data
Uses motion intensity to identify when pen is writing vs. lifted
"""

import numpy as np
import pandas as pd


def calculate_motion_magnitude(df):
    """
    Calculate total motion magnitude from acceleration and gyroscope

    Parameters:
    -----------
    df : pd.DataFrame
        Accelerometer data with acc_x, acc_y, acc_z columns

    Returns:
    --------
    np.array
        Motion magnitude at each time step
    """
    # Calculate acceleration magnitude (remove gravity ~1000mg)
    acc_mag = np.sqrt(
        df["acc_x[mg]"] ** 2 + df["acc_y[mg]"] ** 2 + df["acc_z[mg]"] ** 2
    )
    acc_mag_normalized = np.abs(acc_mag - 1000)  # Remove gravity component

    # If gyroscope data available and active, include it
    if "gyro_x[mdps]" in df.columns:
        gyro_mag = np.sqrt(
            df["gyro_x[mdps]"] ** 2 + df["gyro_y[mdps]"] ** 2 + df["gyro_z[mdps]"] ** 2
        )
        # Check if gyroscope is actually active (not all zeros)
        if np.max(gyro_mag) > 1:  # Threshold to detect if gyro is active
            # Combine both signals (weighted)
            motion = acc_mag_normalized + gyro_mag / 100  # Scale gyro to similar range
        else:
            # Gyro inactive, use only acceleration
            motion = acc_mag_normalized
    else:
        motion = acc_mag_normalized

    return motion


def segment_strokes(df, threshold=50, min_samples=20, merge_gap=50):
    """
    Segment data into individual strokes based on motion intensity

    Parameters:
    -----------
    df : pd.DataFrame
        Accelerometer data
    threshold : float
        Motion threshold to detect active writing
    min_samples : int
        Minimum samples for a valid stroke
    merge_gap : int
        Maximum gap to merge nearby strokes (samples)

    Returns:
    --------
    list of dict
        List of strokes with start/end indices and data
    """
    motion = calculate_motion_magnitude(df)

    # Smooth the motion signal
    window = 5
    motion_smooth = np.convolve(motion, np.ones(window) / window, mode="same")

    # Find regions above threshold
    active = motion_smooth > threshold

    # Find transitions
    strokes = []
    start = None

    for i in range(len(active)):
        if active[i] and start is None:
            start = i
        elif not active[i] and start is not None:
            if i - start >= min_samples:
                strokes.append({"start": start, "end": i})
            start = None

    # Don't forget last stroke
    if start is not None and len(df) - start >= min_samples:
        strokes.append({"start": start, "end": len(df)})

    # Merge nearby strokes
    merged_strokes = []
    for stroke in strokes:
        if not merged_strokes:
            merged_strokes.append(stroke)
        else:
            last = merged_strokes[-1]
            if stroke["start"] - last["end"] < merge_gap:
                last["end"] = stroke["end"]  # Merge
            else:
                merged_strokes.append(stroke)

    # Add the actual data to each stroke
    for stroke in merged_strokes:
        stroke["data"] = df.iloc[stroke["start"] : stroke["end"]].copy()
        stroke["duration"] = (
            stroke["data"]["time[us]"].iloc[-1] - stroke["data"]["time[us]"].iloc[0]
        ) / 1_000_000

    return merged_strokes


def visualize_segmentation(df, strokes):
    """
    Print segmentation results

    Parameters:
    -----------
    df : pd.DataFrame
        Original data
    strokes : list
        Detected strokes
    """
    print(f"Detected {len(strokes)} strokes:")
    for i, stroke in enumerate(strokes):
        print(
            f"  Stroke {i + 1}: samples {stroke['start']}-{stroke['end']}, "
            f"duration {stroke['duration']:.2f}s"
        )
