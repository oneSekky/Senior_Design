"""
Feature Extraction - Extract meaningful features from each stroke
Creates a simple feature vector to represent each letter
"""

import numpy as np
import pandas as pd


def extract_stroke_features(stroke_data):
    """
    Extract features from a single stroke

    Parameters:
    -----------
    stroke_data : pd.DataFrame
        Data for one stroke

    Returns:
    --------
    dict
        Feature vector as dictionary
    """
    features = {}

    # Time features
    features["duration"] = (
        stroke_data["time[us]"].iloc[-1] - stroke_data["time[us]"].iloc[0]
    ) / 1_000_000
    features["num_samples"] = len(stroke_data)

    # Acceleration features
    acc_x = stroke_data["acc_x[mg]"].values
    acc_y = stroke_data["acc_y[mg]"].values
    acc_z = stroke_data["acc_z[mg]"].values

    # Remove gravity (approximate - assume z is mostly gravity)
    acc_x_clean = acc_x
    acc_y_clean = acc_y
    acc_z_clean = acc_z - 1000

    # Calculate trajectory by integrating (simplified - no drift correction)
    # Velocity approximation
    dt = np.diff(stroke_data["time[us]"].values) / 1_000_000
    vel_x = np.cumsum(acc_x_clean[:-1] * dt) * 0.00981  # Convert mg to m/s
    vel_y = np.cumsum(acc_y_clean[:-1] * dt) * 0.00981

    # Statistical features of acceleration
    features["acc_x_mean"] = np.mean(acc_x_clean)
    features["acc_y_mean"] = np.mean(acc_y_clean)
    features["acc_x_std"] = np.std(acc_x_clean)
    features["acc_y_std"] = np.std(acc_y_clean)
    features["acc_x_max"] = np.max(np.abs(acc_x_clean))
    features["acc_y_max"] = np.max(np.abs(acc_y_clean))

    # Velocity-based features
    if len(vel_x) > 0:
        features["vel_x_range"] = np.max(vel_x) - np.min(vel_x)
        features["vel_y_range"] = np.max(vel_y) - np.min(vel_y)
    else:
        features["vel_x_range"] = 0
        features["vel_y_range"] = 0

    # Gyroscope features (if available)
    if "gyro_x[mdps]" in stroke_data.columns:
        gyro_x = stroke_data["gyro_x[mdps]"].values
        gyro_y = stroke_data["gyro_y[mdps]"].values
        gyro_z = stroke_data["gyro_z[mdps]"].values

        features["gyro_x_mean"] = np.mean(gyro_x)
        features["gyro_y_mean"] = np.mean(gyro_y)
        features["gyro_z_mean"] = np.mean(gyro_z)
        features["gyro_magnitude"] = np.mean(np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2))

    # Direction changes (count peaks in acceleration)
    features["direction_changes_x"] = count_zero_crossings(np.diff(acc_x_clean))
    features["direction_changes_y"] = count_zero_crossings(np.diff(acc_y_clean))

    return features


def count_zero_crossings(signal):
    """Count number of zero crossings in a signal"""
    return np.sum(np.diff(np.sign(signal)) != 0)


def create_feature_vector(features):
    """
    Convert feature dict to normalized numpy array

    Parameters:
    -----------
    features : dict
        Feature dictionary

    Returns:
    --------
    np.array
        Feature vector
    """
    # Define feature order
    feature_keys = [
        "duration",
        "num_samples",
        "acc_x_mean",
        "acc_y_mean",
        "acc_x_std",
        "acc_y_std",
        "acc_x_max",
        "acc_y_max",
        "vel_x_range",
        "vel_y_range",
        "direction_changes_x",
        "direction_changes_y",
    ]

    # Add gyro features if available
    if "gyro_x_mean" in features:
        feature_keys.extend(
            ["gyro_x_mean", "gyro_y_mean", "gyro_z_mean", "gyro_magnitude"]
        )

    # Create vector
    vector = np.array([features.get(key, 0) for key in feature_keys])

    return vector


def normalize_features(feature_vectors):
    """
    Normalize feature vectors to 0 mean, unit variance

    Parameters:
    -----------
    feature_vectors : list of np.array
        List of feature vectors

    Returns:
    --------
    np.array
        Normalized feature matrix
    """
    if len(feature_vectors) == 0:
        return np.array([])

    features_matrix = np.vstack(feature_vectors)

    # Normalize (handle division by zero)
    mean = np.mean(features_matrix, axis=0)
    std = np.std(features_matrix, axis=0)
    std[std == 0] = 1  # Avoid division by zero

    normalized = (features_matrix - mean) / std

    return normalized
