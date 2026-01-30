"""
Basic CSV Data Reader for Accelerometer Data
Reads and displays accelerometer data from LSM6DSO16IS sensor
"""

import pandas as pd
import numpy as np


def read_accelerometer_csv(file_path):
    """
    Read accelerometer CSV data from LSM6DSO16IS sensor

    Parameters:
    -----------
    file_path : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the accelerometer data
    """
    # Read CSV, skipping the first comment line
    df = pd.read_csv(file_path, comment='#')

    return df


def extract_acceleration_data(df):
    """
    Extract the acceleration data (x, y, z) in mg units

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing raw sensor data

    Returns:
    --------
    pd.DataFrame
        DataFrame with time and acceleration columns only
    """
    # Extract relevant columns: time and acceleration in mg
    acc_data = df[['time[us]', 'acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]']].copy()

    # Convert time from microseconds to seconds
    acc_data['time[s]'] = acc_data['time[us]'] / 1_000_000

    return acc_data


def get_basic_stats(df):
    """
    Get basic statistics about the acceleration data

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing accelerometer data

    Returns:
    --------
    dict
        Dictionary containing basic statistics
    """
    stats = {
        'num_samples': len(df),
        'duration_seconds': (df['time[us]'].iloc[-1] - df['time[us]'].iloc[0]) / 1_000_000,
        'sampling_rate_hz': len(df) / ((df['time[us]'].iloc[-1] - df['time[us]'].iloc[0]) / 1_000_000),
        'acc_x_mean': df['acc_x[mg]'].mean(),
        'acc_y_mean': df['acc_y[mg]'].mean(),
        'acc_z_mean': df['acc_z[mg]'].mean(),
        'acc_x_std': df['acc_x[mg]'].std(),
        'acc_y_std': df['acc_y[mg]'].std(),
        'acc_z_std': df['acc_z[mg]'].std(),
    }

    return stats


def main():
    """
    Main function to demonstrate reading and processing CSV data
    """
    # Path to test data
    csv_file = '../Test_Data/Test2.csv'

    print("Reading accelerometer data...")
    df = read_accelerometer_csv(csv_file)

    print(f"\nLoaded {len(df)} samples")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nExtracting acceleration data...")
    acc_data = extract_acceleration_data(df)
    print(acc_data.head())

    print("\nBasic Statistics:")
    stats = get_basic_stats(df)
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

    print("\nData shape:", df.shape)
    print("Columns:", df.columns.tolist())


if __name__ == "__main__":
    main()
