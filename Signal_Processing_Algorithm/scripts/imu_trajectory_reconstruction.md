# IMU to Trajectory Reconstruction: Implementation Guide

## Overview

The model takes a multivariate IMU time series as input (one CSV per letter) and outputs a sequence of (x, y) displacement vectors tracing the pen's path. The letter image is used only during training: you process it into a ground-truth trajectory, which becomes the regression target. At inference, only the CSV is needed.

---

## Step 1: Understand Your Data Format

Before writing any code, answer these questions about your CSVs:

- What columns are present? Expect something like: `timestamp`, `accel_x`, `accel_y`, `accel_z`, `gyro_x`, `gyro_y`, `gyro_z`, and possibly magnetometer or force channels.
- What is your sampling rate? Typical IMU pens sample at 100-400 Hz.
- Are sequences variable length? Letters written at different speeds will produce different row counts. This is addressed in Step 3.
- Do you have a pen-up/pen-down indicator? If so, keep it as a feature. It is highly informative.

---

## Step 2: Extract Ground-Truth Trajectories from the Images

This is the core preprocessing challenge. You need to convert each letter image into a sequence of `(dx, dy)` displacement vectors representing where the pen traveled over time.

Install dependencies:

```bash
pip install opencv-contrib-python scikit-image scipy numpy pillow
```

### 2a. Load and binarize the image

```python
import cv2
import numpy as np
from skimage.morphology import skeletonize

img = cv2.imread("letter_a.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
binary_bool = (binary // 255).astype(bool)
```

### 2b. Skeletonize to reduce strokes to single-pixel-width paths

```python
skeleton = skeletonize(binary_bool)
skeleton_uint8 = (skeleton * 255).astype(np.uint8)
```

### 2c. Trace the skeleton into an ordered sequence of (x, y) coordinates

This is the trickiest step. Find all white pixels, build a neighbor graph, and do a depth-first traversal starting from the topmost-leftmost pixel.

```python
def trace_skeleton(skel_img):
    coords = np.column_stack(np.where(skel_img > 0))  # (row, col) = (y, x)
    coord_set = set(map(tuple, coords))

    def neighbors(y, x):
        return [(y+dy, x+dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                if (dy, dx) != (0, 0) and (y+dy, x+dx) in coord_set]

    start = tuple(coords[0])
    visited = []
    stack = [start]
    seen = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        visited.append(node)
        for nb in neighbors(*node):
            if nb not in seen:
                stack.append(nb)

    return np.array(visited)  # shape (N, 2), columns are [y, x]
```

> **Note on multi-stroke letters:** The DFS approach above works well for single-stroke letters. For letters with multiple strokes or crossings (i, j, t, k, x), segment the skeleton by connected components and order them heuristically — e.g. left-to-right, top-to-bottom order of each stroke's starting pixel.

### 2d. Convert absolute coordinates to relative displacements

This is what the literature recommends. It removes dependence on absolute starting position and makes targets translation-invariant.

```python
def coords_to_displacements(coords):
    # coords: shape (N, 2)
    displacements = np.diff(coords, axis=0).astype(float)
    return displacements  # shape (N-1, 2)
```

### 2e. Normalize displacements

Divide by the maximum absolute displacement value observed across the full training set. Save this constant so you can inverse-transform at inference time.

```python
def compute_disp_scale(all_displacements):
    return np.max(np.abs(np.concatenate(all_displacements, axis=0)))

def normalize_displacements(displacements, scale):
    return displacements / scale
```

---

## Step 3: Preprocess the IMU CSV Data

### 3a. Load and select channels

Drop magnetometer if a tablet or magnetic surface was nearby during recording — it will be noisy. Keep: `accel_x`, `accel_y`, `accel_z`, `gyro_x`, `gyro_y`, `gyro_z`, and any force or pen-down channel.

```python
import pandas as pd

def load_imu(path, cols):
    df = pd.read_csv(path)
    return df[cols].values.astype(float)  # shape (T, C)
```

### 3b. Optional: high-pass filter to suppress sensor drift

Apply a high-pass Butterworth filter to accelerometer channels at ~0.5-1 Hz before normalization.

```python
from scipy.signal import butter, filtfilt

def highpass_filter(data, cutoff=0.5, fs=100, order=4):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, data, axis=0)
```

### 3c. Z-score normalize per channel

Compute mean and std from the training set only, then apply to validation and test sets.

```python
def compute_imu_stats(imu_list):
    stacked = np.concatenate(imu_list, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)

def normalize_imu(data, mean, std):
    return (data - mean) / (std + 1e-8)
```

### 3d. Resample to a fixed length

Every sequence gets resampled to a fixed length `L` using linear interpolation. `L = 256` or `512` is a good starting point. Apply the same resampling to the trajectory ground truth so input and target lengths match.

```python
from scipy.interpolate import interp1d

def resample_sequence(seq, target_len):
    T, C = seq.shape
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, C))
    for c in range(C):
        f = interp1d(old_t, seq[:, c], kind='linear')
        resampled[:, c] = f(new_t)
    return resampled
```

> **Note on padding vs. resampling:** Padding to max length is an alternative but introduces known problems when combined with z-normalization — zero-padding becomes statistically indistinguishable from genuine low-value data. Resampling is recommended to start.

---

## Step 4: Build the Dataset Class

```python
import torch
from torch.utils.data import Dataset

class HandwritingDataset(Dataset):
    def __init__(self, imu_list, traj_list, imu_mean, imu_std,
                 disp_scale, target_len=256):
        # imu_list: list of raw numpy arrays, shape (T_i, C)
        # traj_list: list of displacement arrays, shape (T_i-1, 2)
        self.samples = []
        for imu, traj in zip(imu_list, traj_list):
            imu_norm = normalize_imu(imu, imu_mean, imu_std)
            imu_rs = resample_sequence(imu_norm, target_len)
            traj_norm = normalize_displacements(traj, disp_scale)
            traj_rs = resample_sequence(traj_norm, target_len)
            self.samples.append((
                torch.tensor(imu_rs, dtype=torch.float32),
                torch.tensor(traj_rs, dtype=torch.float32)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imu, traj = self.samples[idx]
        # Conv1d expects (C, T), so transpose the IMU tensor
        return imu.T, traj  # shapes: (C, L), (L, 2)
```

---

## Step 5: Define the Model

A 1D CNN with dilated residual blocks. This architecture is directly grounded in Wehbi et al. (2022) and the TCN-based approach from Imbert et al. (IJDAR 2023). It maps `(B, C, L)` to `(B, L, 2)` without any downsampling, preserving the frame-by-frame alignment between input and output.

```python
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class IMUToTrajectory(nn.Module):
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            ResBlock1D(hidden, kernel_size=3, dilation=1),
            ResBlock1D(hidden, kernel_size=3, dilation=2),
            ResBlock1D(hidden, kernel_size=3, dilation=4),
            ResBlock1D(hidden, kernel_size=3, dilation=8),
        )
        self.output_proj = nn.Conv1d(hidden, 2, kernel_size=1)

    def forward(self, x):
        # x: (B, C, L)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)  # (B, 2, L)
        return x.permute(0, 2, 1)  # (B, L, 2)
```

The increasing dilation factors (1, 2, 4, 8) give the model a wide temporal receptive field without any stride or pooling, which is critical for preserving per-timestep output alignment.

---

## Step 6: Define the Loss Function

MSE on displacements, plus a smoothness regularization term that penalizes abrupt direction changes in the predicted trajectory.

```python
import torch.nn.functional as F

def trajectory_loss(pred, target, smooth_weight=0.1):
    mse = F.mse_loss(pred, target)
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
    smoothness = (pred_diff ** 2).mean()
    return mse + smooth_weight * smoothness
```

Start with `smooth_weight=0.1` and tune it if predictions are too jagged or too rounded.

---

## Step 7: Train the Model

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

def train(model, dataset, epochs=100, lr=1e-3, batch_size=32):
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10,
                                                      factor=0.5, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imu, traj in train_loader:
            imu, traj = imu.to(device), traj.to(device)
            optimizer.zero_grad()
            pred = model(imu)
            loss = trajectory_loss(pred, traj)
            loss.backward()
            # Gradient clipping stabilizes training on noisy IMU data
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imu, traj in val_loader:
                imu, traj = imu.to(device), traj.to(device)
                pred = model(imu)
                val_loss += trajectory_loss(pred, traj).item()

        scheduler.step(val_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"train={train_loss/len(train_loader):.4f} | "
                  f"val={val_loss/len(val_loader):.4f}")

    return model
```

---

## Step 8: Inference and Visualization

Load a CSV, preprocess it identically to training (same normalization constants, same resampling length), run the model, then reconstruct absolute position from predicted displacements via cumulative sum.

```python
import matplotlib.pyplot as plt

def reconstruct_trajectory(displacements, disp_scale):
    # displacements: numpy array shape (L, 2), normalized
    disp_unnorm = displacements * disp_scale
    coords = np.cumsum(disp_unnorm, axis=0)
    return coords  # (L, 2) absolute positions

def infer(model, imu_csv_path, cols, imu_mean, imu_std,
          disp_scale, target_len, device):
    imu_raw = load_imu(imu_csv_path, cols)
    imu_norm = normalize_imu(imu_raw, imu_mean, imu_std)
    imu_rs = resample_sequence(imu_norm, target_len)
    imu_tensor = torch.tensor(imu_rs.T, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_disp = model(imu_tensor).squeeze(0).cpu().numpy()

    return reconstruct_trajectory(pred_disp, disp_scale)

def plot_trajectory(coords):
    plt.figure(figsize=(4, 4))
    plt.plot(coords[:, 1], -coords[:, 0], lw=1.5, color='black')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
```

---

## Step 9: Evaluation Metrics

Three metrics to track reconstruction quality:

**MSE on displacements** is your training loss and is straightforward to interpret.

**Normalized trajectory error** divides MSE by the bounding box diagonal of the ground-truth letter, making the score scale-independent. Wehbi et al. (2022) report 0.176 on this metric as a benchmark.

```python
def normalized_error(pred_coords, gt_coords):
    bbox_diag = np.linalg.norm(gt_coords.max(axis=0) - gt_coords.min(axis=0))
    return np.sqrt(np.mean((pred_coords - gt_coords)**2)) / (bbox_diag + 1e-8)
```

**Frechet distance** measures geometric similarity of two curves as ordered paths, which is more meaningful than pointwise MSE when curves are similar in shape but slightly offset in time.

```bash
pip install similaritymeasures
```

```python
import similaritymeasures

def frechet(pred_coords, gt_coords):
    return similaritymeasures.frechet_dist(pred_coords, gt_coords)
```

---

## Step 10: Data Augmentation

The model is data-hungry. Since you are working per-letter, augment aggressively:

```python
def augment_imu(imu, noise_std=0.01, scale_range=(0.9, 1.1)):
    # Gaussian noise
    imu = imu + np.random.normal(0, noise_std, imu.shape)
    # Random amplitude scaling per channel
    scale = np.random.uniform(*scale_range, size=(1, imu.shape[1]))
    return imu * scale

def augment_trajectory(traj, scale_range=(0.85, 1.15)):
    # Scale trajectory uniformly (simulate different writing sizes)
    scale = np.random.uniform(*scale_range)
    return traj * scale
```

Apply these transforms inside `__getitem__` during training only, not during validation.

---

## Common Pitfalls

**Drift accumulation at inference.** Because you cumsum displacements, small per-step errors compound. Mitigate by keeping inputs short (single letters, not words), applying the high-pass filter in Step 3b, and using the smoothness loss in Step 6.

**Bad skeleton ordering.** The DFS tracer works for simple letters but will produce scrambled orderings for letters with crossings or pen lifts. Visually inspect 10-20 extracted trajectories before training to confirm they look like plausible pen paths.

**Normalization leakage.** Always compute `imu_mean`, `imu_std`, and `disp_scale` from the training split only. Apply the saved values to validation and test. Save them alongside the model checkpoint.

**Resampling distorts writing speed.** Resampling conflates fast writers and slow writers into the same fixed length. If you have enough data, group samples by sequence length quartile and train with a longer fixed length to preserve more speed information.

**IMU axes may not align with writing plane.** Depending on how the pen is held, the writing-relevant accelerations may be a linear combination of the raw axes. Consider adding a learned linear projection as the very first layer before the input_proj Conv1d, or pre-rotate axes using gravity calibration if your recording setup allows it.
