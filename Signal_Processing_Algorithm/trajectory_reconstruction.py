"""
Trajectory Reconstruction - Convert IMU data to X,Y coordinates
Reconstructs the pen's path by integrating acceleration data
"""

import numpy as np
import pandas as pd

try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply Butterworth low-pass filter to reduce noise

    Parameters:
    -----------
    data : np.array
        Input signal
    cutoff : float
        Cutoff frequency in Hz
    fs : float
        Sampling frequency in Hz
    order : int
        Filter order

    Returns:
    --------
    np.array
        Filtered signal
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


def remove_gravity(acc_data):
    """
    Remove gravity component from acceleration
    Assumes sensor starts stationary to calibrate gravity direction

    Parameters:
    -----------
    acc_data : np.array
        Array of shape (n, 3) with [ax, ay, az]

    Returns:
    --------
    np.array
        Acceleration with gravity removed
    """
    # Estimate gravity as mean of first few samples (assuming stationary start)
    gravity = np.mean(acc_data[:100], axis=0)

    # Remove gravity
    acc_clean = acc_data - gravity

    return acc_clean


def integrate_acceleration(acc_data, dt, use_zero_velocity_update=True):
    """
    Integrate acceleration to get position
    acc -> velocity -> position (double integration)

    Parameters:
    -----------
    acc_data : np.array
        Acceleration data (n, 3) in m/s²
    dt : np.array
        Time steps in seconds
    use_zero_velocity_update : bool
        Apply zero-velocity updates to reduce drift

    Returns:
    --------
    velocity : np.array
        Velocity (n, 3) in m/s
    position : np.array
        Position (n, 3) in meters
    """
    # First integration: acceleration -> velocity
    velocity = np.zeros_like(acc_data)
    vel_integrated = cumtrapz(acc_data, dx=dt, axis=0)
    velocity[1:] = vel_integrated

    # Zero-velocity update (ZUPT) - detect stationary periods and reset velocity
    if use_zero_velocity_update:
        # Detect when acceleration is very small (pen stationary or lifted)
        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        stationary_threshold = 0.1  # m/s² (tune this)
        stationary_mask = acc_magnitude < stationary_threshold

        # Reset velocity during stationary periods
        velocity[stationary_mask] = 0

    # Second integration: velocity -> position
    position = np.zeros_like(velocity)
    pos_integrated = cumtrapz(velocity, dx=dt, axis=0)
    position[1:] = pos_integrated

    return velocity, position


def reconstruct_trajectory(df, cutoff_freq=5.0, remove_z=True):
    """
    Reconstruct 2D/3D trajectory from accelerometer data

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with accelerometer data
    cutoff_freq : float
        Low-pass filter cutoff frequency (Hz)
    remove_z : bool
        If True, only return X,Y (2D trajectory)

    Returns:
    --------
    trajectory : np.array
        Position data (n, 2) or (n, 3) in meters
    velocity : np.array
        Velocity data
    time : np.array
        Time in seconds
    """
    # Extract acceleration data (convert from mg to m/s²)
    acc_mg = df[["acc_x[mg]", "acc_y[mg]", "acc_z[mg]"]].values
    acc_ms2 = acc_mg * 0.00981  # Convert mg to m/s²

    # Calculate time steps
    time_us = df["time[us]"].values
    time_s = (time_us - time_us[0]) / 1e6
    dt = np.mean(np.diff(time_s))  # Average time step

    # Calculate sampling frequency
    fs = 1.0 / dt

    # Remove gravity
    acc_clean = remove_gravity(acc_ms2)

    # Apply low-pass filter to reduce noise
    acc_filtered = np.zeros_like(acc_clean)
    for i in range(3):
        acc_filtered[:, i] = butter_lowpass_filter(acc_clean[:, i], cutoff_freq, fs)

    # Integrate to get trajectory
    velocity, position = integrate_acceleration(acc_filtered, dt)

    # Return 2D or 3D trajectory
    if remove_z:
        trajectory = position[:, :2]  # Only X, Y
    else:
        trajectory = position

    return trajectory, velocity, time_s


def reconstruct_stroke_trajectory(stroke_data, cutoff_freq=5.0):
    """
    Reconstruct trajectory for a single stroke

    Parameters:
    -----------
    stroke_data : pd.DataFrame
        Single stroke data
    cutoff_freq : float
        Filter cutoff frequency

    Returns:
    --------
    trajectory : np.array
        2D trajectory (n, 2)
    """
    trajectory, _, _ = reconstruct_trajectory(stroke_data, cutoff_freq)

    # Center the trajectory (start at origin)
    trajectory = trajectory - trajectory[0]

    return trajectory


def plot_trajectory(trajectory, title="Reconstructed Trajectory"):
    """
    Plot a 2D trajectory

    Parameters:
    -----------
    trajectory : np.array
        2D trajectory (n, 2)
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2)
    plt.plot(trajectory[0, 0], trajectory[0, 1], "go", markersize=10, label="Start")
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", markersize=10, label="End")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    return plt


def save_trajectory_svg(trajectory, filename, scale=1000):
    """
    Save trajectory as SVG path for use in graphics applications

    Parameters:
    -----------
    trajectory : np.array
        2D trajectory (n, 2)
    filename : str
        Output SVG filename
    scale : float
        Scale factor (default 1000 converts meters to mm)
    """
    # Scale and flip Y (SVG has Y increasing downward)
    traj_scaled = trajectory * scale
    traj_scaled[:, 1] = -traj_scaled[:, 1]

    # Find bounding box
    min_x, min_y = traj_scaled.min(axis=0)
    max_x, max_y = traj_scaled.max(axis=0)

    # Add margin
    margin = 10
    width = max_x - min_x + 2 * margin
    height = max_y - min_y + 2 * margin

    # Shift to positive coordinates
    traj_shifted = traj_scaled - [min_x - margin, min_y - margin]

    # Create SVG path
    path_data = f"M {traj_shifted[0, 0]:.2f},{traj_shifted[0, 1]:.2f}"
    for point in traj_shifted[1:]:
        path_data += f" L {point[0]:.2f},{point[1]:.2f}"

    # Write SVG file
    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width:.2f}" height="{height:.2f}" viewBox="0 0 {width:.2f} {height:.2f}">
  <path d="{path_data}" stroke="black" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

    with open(filename, "w") as f:
        f.write(svg_content)

    print(f"Trajectory saved to {filename}")


def export_trajectory_json(trajectory, filename):
    """
    Export trajectory as JSON for use in web applications

    Parameters:
    -----------
    trajectory : np.array
        2D trajectory (n, 2)
    filename : str
        Output JSON filename
    """
    import json

    # Convert to list of points
    points = [{"x": float(x), "y": float(y)} for x, y in trajectory]

    data = {
        "points": points,
        "num_points": len(points),
        "bounds": {
            "min_x": float(trajectory[:, 0].min()),
            "max_x": float(trajectory[:, 0].max()),
            "min_y": float(trajectory[:, 1].min()),
            "max_y": float(trajectory[:, 1].max()),
        },
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Trajectory saved to {filename}")
