"""
train_side_mount.py

Trains the handwriting generation model on side-mount boxed-paper data.

Data layout:
  Test_Data/side_mount/split_csvs/<page_stem>/box_RR_CC.csv
  Test_Data/side_mount/split_images/<page_stem>/box_RR_CC.png

Architecture: 1D-CNN encoder (not LSTM) → convolutional upsampling decoder.
The encoder preserves the temporal axis through the bottleneck so the decoder
can use spatial attention over time-steps rather than a single flat vector.
This is the key fix vs. the previous BiLSTM→Dense approach, which squashed
all temporal/spatial information into a 1024-dim vector and then had to
re-invent 64×64 structure from scratch — producing only center blobs.

Loss: Focal loss (γ=2, α=0.75).  Better than weighted-BCE for ~97% background
because it dynamically down-weights easy negatives per pixel rather than using
a single scalar weight for the whole batch.

Model size: ~280k params (down from 10.8M) — matched to ~400 training samples.

Augmentation: noise, amplitude scale, time warp (IMU); small rotation + elastic
deformation (image).  Horizontal flip REMOVED — it is harmful for asymmetric
letters (b/d, p/q, etc.).
"""

import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
=======
import tensorflow as tf
from PIL import Image
from scipy import signal, ndimage
from sklearn.model_selection import train_test_split
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR / ".." / ".." / "Test_Data" / "side_mount"
CSV_DIR = DATA_ROOT / "split_csvs"
IMG_DIR = DATA_ROOT / "split_images"
MODELS_DIR = SCRIPT_DIR / ".." / "models"
MODELS_DIR.mkdir(exist_ok=True)

<<<<<<< HEAD
# ── Sampling / window ─────────────────────────────────────────────────────────
FS = 104
HEAD_PAD_S = 0.15
CORE_S = 2.00
TAIL_PAD_S = 0.15
TOTAL_S = HEAD_PAD_S + CORE_S + TAIL_PAD_S

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE = 64
N_TIMESTEPS = int(round(TOTAL_S * FS))  # 240
FEATURE_COLS = [
    "acc_x[mg]",
    "acc_y[mg]",
    "acc_z[mg]",
    "gyro_x[mdps]",
    "gyro_y[mdps]",
    "gyro_z[mdps]",
]

EPOCHS = 120
BATCH_SIZE = 16
HIDDEN_DIM = 256

# ── Loss settings ──────────────────────────────────────────────────────────────
# GT is rendered from clean extracted strokes (~2px wide).
# Dice loss directly penalises blob outputs — a blob against a thin-stroke GT
# gets Dice ≈ 0, forcing the model to be sparse and precise.
BCE_POS_WEIGHT = 2.0  # reduced from 6; Dice handles imbalance
DICE_WEIGHT = 0.8  # fraction of loss that is Dice; remainder is BCE
RENDER_WIDTH = 2
=======
# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE   = 64
N_TIMESTEPS  = 150
FEATURE_COLS = ['acc_x[mg]', 'acc_y[mg]', 'acc_z[mg]',
                'gyro_x[mdps]', 'gyro_y[mdps]', 'gyro_z[mdps]']
TEST_SIZE    = 0.15
EPOCHS       = 300
BATCH_SIZE   = 16

# ── Augmentation ───────────────────────────────────────────────────────────────
AUGMENT_FACTOR  = 8        # copies per real sample
AUG_NOISE_STD   = 0.04
AUG_SCALE_RANGE = (0.85, 1.15)
AUG_WARP_RANGE  = (0.80, 1.20)
AUG_IMG_SHIFT   = 3        # max pixel translation
AUG_IMG_ROT_DEG = 8        # max rotation (degrees) — replaces flip
AUG_ELASTIC_ALPHA = 6.0    # elastic deformation strength

# ── Active-segment detection ───────────────────────────────────────────────────
ACTIVITY_THRESHOLD = 40.0  # mg (lowered from 50 to catch lighter writing)
SMOOTH_WINDOW      = 10
MIN_ACTIVE_SAMPLES = 20
MERGE_GAP          = 15
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611


<<<<<<< HEAD
# Stroke extraction constants (mirrors train_stroke.py)
N_STROKES = 4
N_POINTS = 15

# ── Augmentation ───────────────────────────────────────────────────────────────
AUGMENT_FACTOR = 6
AUG_NOISE_STD = 0.04
AUG_SCALE_RANGE = (0.85, 1.15)
AUG_WARP_RANGE = (0.80, 1.20)
AUG_IMG_SHIFT = 3
AUG_IMG_ROT_DEG = 8
AUG_ELASTIC_ALPHA = 6.0
AUG_DRIFT_PROB = 0.25
AUG_DRIFT_MAX = 0.1
AUG_DROP_PROB = 0.25
AUG_DROP_LEN = (3, 10)


# ── Stroke GT helpers (same pipeline as train_stroke.py) ──────────────────────


def _order_points(pts):
    """Greedily order skeleton pixels into a connected path."""
    if len(pts) <= 2:
        return pts
    start = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])
    used = np.zeros(len(pts), dtype=bool)
    used[start] = True
    ordered = [start]
    tree = cKDTree(pts)
    while len(ordered) < len(pts):
        last = pts[ordered[-1]]
        _, idxs = tree.query(last, k=min(20, len(pts)))
        moved = False
        for idx in idxs:
            idx = int(idx)
            if not used[idx]:
                ordered.append(idx)
                used[idx] = True
                moved = True
                break
        if not moved:
            rem = np.where(~used)[0]
            if len(rem):
                ordered.append(int(rem[0]))
                used[rem[0]] = True
    return pts[np.array(ordered)]


def _extract_strokes(img_arr):
    """Extract up to N_STROKES ordered paths from a letter image."""
    strokes = np.zeros((N_STROKES, N_POINTS, 2), dtype=np.float32)
    valid = np.zeros(N_STROKES, dtype=bool)
    binary = (img_arr > 0.3).astype(bool)
    if not binary.any():
        return strokes, valid
    struct = ndimage.generate_binary_structure(2, 2)
    dilated = binary.copy()
    for _ in range(2):  # 2px: close gaps, preserve curves
        dilated = ndimage.binary_dilation(dilated, structure=struct)
    skel = skeletonize(dilated)
    labeled, n_comp = ndimage.label(skel, structure=struct)  # 8-connectivity
    components = []
    for i in range(1, n_comp + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < 3:
            continue
        pts = np.column_stack([ys, xs]).astype(np.float32)
        components.append(_order_points(pts))
    components.sort(key=lambda x: -len(x))
    components = components[:N_STROKES]
    H, W = img_arr.shape
    for i, pts in enumerate(components):
        idx = np.linspace(0, len(pts) - 1, N_POINTS).astype(int)
        s = pts[idx].astype(np.float32)
        s[:, 0] /= H
        s[:, 1] /= W
        strokes[i] = s
        valid[i] = True
    return strokes, valid


def make_stroke_target(img_arr):
=======
# ── Data loading ───────────────────────────────────────────────────────────────

def extract_active_segment(features, magnitude):
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611
    """
    Find the writing segment within a raw 208-sample box and return a
    (N_TIMESTEPS, n_features) float32 array.
    """
<<<<<<< HEAD
    strokes, valid = _extract_strokes(img_arr)
    if not valid.any():
        return np.zeros_like(img_arr)
    size = img_arr.shape[0]
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    for s in range(N_STROKES):
        if not valid[s]:
            continue
        pts = strokes[s]  # (P, 2): row, col
        pix = [(float(p[1] * size), float(p[0] * size)) for p in pts]
        if len(pix) >= 2:
            draw.line(pix, fill=255, width=RENDER_WIDTH)
    return np.array(canvas, dtype=np.float32) / 255.0
=======
    n_features = features.shape[1]
    smoothed = np.convolve(magnitude, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
                           mode='same')
    above = smoothed > ACTIVITY_THRESHOLD

    runs = []
    cur_start, cur_len = 0, 0
    for i, val in enumerate(above):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > 0:
                runs.append((cur_start, cur_start + cur_len))
            cur_len = 0
    if cur_len > 0:
        runs.append((cur_start, cur_start + cur_len))

    merged = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= MERGE_GAP:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611

    best_start, best_end = 0, 0
    for start, end in merged:
        if end - start > best_end - best_start:
            best_start, best_end = start, end

    if best_end - best_start >= MIN_ACTIVE_SAMPLES:
        seg = features[best_start:best_end]
    else:
        peak = int(np.argmax(smoothed))
        half  = N_TIMESTEPS // 2
        start = max(0, peak - half)
        end   = min(len(features), start + N_TIMESTEPS)
        start = max(0, end - N_TIMESTEPS)
        seg   = features[start:end]

    if len(seg) >= N_TIMESTEPS:
        seg = seg[:N_TIMESTEPS]
    else:
        pad = np.zeros((N_TIMESTEPS - len(seg), n_features), dtype=np.float32)
        seg = np.vstack([seg, pad])

    return seg



def load_csv(path):
<<<<<<< HEAD
    df = pd.read_csv(path, skiprows=1)
    data = df[FEATURE_COLS].values.astype(np.float32)
    N = len(data)

    b, a = signal.butter(3, 0.5, "high", fs=FS)  # gravity removal
=======
    """Load one box CSV → (N_TIMESTEPS, 7) float32 array."""
    df = pd.read_csv(path, skiprows=1)
    data = df[FEATURE_COLS].values.astype(np.float32)

    b, a = signal.butter(3, 0.5, 'high', fs=104)
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i])

    magnitude = np.sqrt((filtered[:, :3] ** 2).sum(axis=1))
<<<<<<< HEAD
    dt = 1.0 / FS
    acc_xy = filtered[:, :2]
    vel_xy = np.cumsum(acc_xy * dt, axis=0)
    b_hp, a_hp = signal.butter(2, 0.3, "high", fs=FS)  # remove noice from integration
    for i in range(2):
        vel_xy[:, i] = (
            signal.filtfilt(b_hp, a_hp, vel_xy[:, i]) if N > 9 else vel_xy[:, i]
        )
    pos_xy = np.cumsum(vel_xy * dt, axis=0)
    for i in range(2):
        pos_xy[:, i] = (
            signal.filtfilt(b_hp, a_hp, pos_xy[:, i]) if N > 9 else pos_xy[:, i]
        )

    features = np.hstack([filtered, magnitude[:, None], vel_xy, pos_xy])
    n_feat = features.shape[1]
    start_idx = N - N_TIMESTEPS
    if start_idx >= 0:
        seg = features[start_idx:N]
    else:
        seg = np.vstack(
            [np.zeros((-start_idx, n_feat), dtype=np.float32), features[:N]]
        )
    return seg.astype(np.float32)


def load_image(path):
    img = Image.open(path).convert("L")
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr  # stroke=1, bg=0
=======
    features  = np.hstack([filtered, magnitude[:, None]])   # (N, 7)
    return extract_active_segment(features, magnitude)


def load_image(path):
    """Load one box PNG → (IMAGE_SIZE, IMAGE_SIZE) float32 in [0,1], strokes=1."""
    img = Image.open(path).convert('L')
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611


def build_img_dir_map():
    mapping = {}
<<<<<<< HEAD
    for d in IMG_DIR.iterdir():
        if d.is_dir():
            mapping[re.sub(r"_\d+$", "", d.name)] = d
    return mapping


def _page_num(stem):
    m = re.search(r"page-(\d+)", stem)
    return int(m.group(1)) if m else -1


=======
    for img_dir in IMG_DIR.iterdir():
        if not img_dir.is_dir():
            continue
        key = re.sub(r'_\d+$', '', img_dir.name)
        mapping[key] = img_dir
    return mapping


>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611
def load_dataset():
    X, y, labels = [], [], []
    missing = 0
    img_dir_map = build_img_dir_map()

    for csv_page_dir in sorted(CSV_DIR.iterdir()):
        if not csv_page_dir.is_dir():
            continue
<<<<<<< HEAD
        stem = csv_page_dir.name
        img_page_dir = img_dir_map.get(stem)
        if img_page_dir is None:
            print(f"  WARNING: no image folder for {stem}, skipping")
            continue

        for csv_path in sorted(csv_page_dir.glob("box_*.csv")):
            box = csv_path.stem
            img_path = img_page_dir / f"{box}.png"
=======
        page_stem    = csv_page_dir.name
        img_page_dir = img_dir_map.get(page_stem)
        if img_page_dir is None:
            print(f'  WARNING: no image folder for {page_stem}, skipping')
            continue

        for csv_path in sorted(csv_page_dir.glob('box_*.csv')):
            box_name = csv_path.stem
            img_path = img_page_dir / f'{box_name}.png'
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611
            if not img_path.exists():
                missing += 1
                continue
            try:
<<<<<<< HEAD
                img_arr = load_image(img_path)
                target_arr = make_stroke_target(img_arr)
                if target_arr.sum() < 5:
                    skipped += 1
                    continue
                X.append(load_csv(csv_path))
                y_target.append(target_arr)
                y_raw.append(img_arr)
                labels.append(f"{stem}/{box}")
            except Exception as e:
                print(f"  ERROR {stem}/{box}: {e}")

    if missing:
        print(f"  Skipped {missing} CSV/PNG pairs (no image)")
    if skipped:
        print(f"  Skipped {skipped} samples (empty stroke target)")

    return (
        np.array(X, dtype=np.float32),
        np.array(y_target, dtype=np.float32),
        np.array(y_raw, dtype=np.float32),
        labels,
    )
=======
                X.append(load_csv(csv_path))
                y.append(load_image(img_path))
                labels.append(f'{page_stem}/{box_name}')
            except Exception as e:
                print(f'  ERROR loading {page_stem}/{box_name}: {e}')

    if missing:
        print(f'  Skipped {missing} CSV files with no matching image')
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), labels
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611


# ── Augmentation ───────────────────────────────────────────────────────────────


def elastic_deform(image, alpha, sigma=4.0, rng=None):
<<<<<<< HEAD
=======
    """
    Apply elastic deformation to a 2D image array.
    alpha controls displacement magnitude, sigma controls smoothness.
    """
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611
    if rng is None:
        rng = np.random.default_rng()
    shape = image.shape
    dx = ndimage.gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    dy = ndimage.gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
<<<<<<< HEAD
    gx, gy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = [np.clip(gy + dy, 0, shape[0] - 1), np.clip(gx + dx, 0, shape[1] - 1)]
    return ndimage.map_coordinates(image, coords, order=1, mode="reflect")


def drift_augment(imu, rng):
    imu = imu.copy()
    n_segs = rng.integers(2, 6)
    seg_len = len(imu) // n_segs
    for s in range(n_segs):
        st = s * seg_len
        en = min(st + seg_len, len(imu))
        dr = rng.uniform(-AUG_DRIFT_MAX, AUG_DRIFT_MAX, imu.shape[1]).astype(np.float32)
        imu[st:en] += np.linspace(0, 1, en - st)[:, None].astype(np.float32) * dr
    return imu

=======
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = [np.clip(y + dy, 0, shape[0]-1),
              np.clip(x + dx, 0, shape[1]-1)]
    return ndimage.map_coordinates(image, coords, order=1, mode='reflect')


def augment_sample(imu, img, rng):
    """
    Apply random augmentation to one (imu, img) pair.
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611

    IMU: amplitude scale, Gaussian noise, time warp.
    Image: small translation, small rotation, elastic deformation.
    NO horizontal flip — asymmetric letters (b/d/p/q) would be mislabeled.
    """
    imu = imu.copy()
<<<<<<< HEAD
    for _ in range(rng.integers(1, 4)):
        dl = int(rng.integers(*AUG_DROP_LEN))
        st = int(rng.integers(1, max(2, len(imu) - dl)))
        imu[st : min(st + dl, len(imu))] = imu[st - 1]
    return imu
=======
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611

    # Amplitude scale
    imu *= rng.uniform(*AUG_SCALE_RANGE)

    # Gaussian noise
    imu += rng.normal(0, AUG_NOISE_STD, imu.shape).astype(np.float32)
<<<<<<< HEAD
    if rng.random() < AUG_DRIFT_PROB:
        imu = drift_augment(imu, rng)
    if rng.random() < AUG_DROP_PROB:
        imu = dropout_augment(imu, rng)
    warp = rng.uniform(*AUG_WARP_RANGE)
    new_len = max(int(len(imu) * warp), 2)
    xs_o = np.linspace(0, 1, len(imu))
    xs_n = np.linspace(0, 1, new_len)
    warped = np.stack(
        [np.interp(xs_n, xs_o, imu[:, c]) for c in range(imu.shape[1])], 1
    ).astype(np.float32)
    xs_out = np.linspace(0, 1, N_TIMESTEPS)
    imu = np.stack(
        [
            np.interp(xs_out, np.linspace(0, 1, new_len), warped[:, c])
            for c in range(warped.shape[1])
        ],
        1,
    ).astype(np.float32)

    # ── Image augmentation — transform raw, then re-derive stroke target ──
    dy = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    dx = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    angle = rng.uniform(-AUG_IMG_ROT_DEG, AUG_IMG_ROT_DEG)
    do_elastic = rng.random() < 0.5

    def transform_img(img):
        img = np.roll(np.roll(img.copy(), dy, 0), dx, 1)
        if dy > 0:
            img[:dy, :] = 0
        elif dy < 0:
            img[dy:, :] = 0
        if dx > 0:
            img[:, :dx] = 0
        elif dx < 0:
            img[:, dx:] = 0
        img = ndimage.rotate(img, angle, reshape=False, mode="constant", cval=0.0)
        if do_elastic:
            img = elastic_deform(img, AUG_ELASTIC_ALPHA, rng=rng)
        return np.clip(img, 0, 1).astype(np.float32)

    aug_raw = transform_img(img_raw)
    aug_target = make_stroke_target(aug_raw)  # re-derive from augmented photo
=======

    # Time warp: resample at different speed then re-sample back to N_TIMESTEPS
    warp    = rng.uniform(*AUG_WARP_RANGE)
    orig_len = imu.shape[0]
    new_len  = max(int(orig_len * warp), 2)
    xs_orig  = np.linspace(0, 1, orig_len)
    xs_new   = np.linspace(0, 1, new_len)
    warped   = np.stack([np.interp(xs_new, xs_orig, imu[:, c])
                         for c in range(imu.shape[1])], axis=1).astype(np.float32)
    xs_out   = np.linspace(0, 1, N_TIMESTEPS)
    xs_w     = np.linspace(0, 1, new_len)
    imu      = np.stack([np.interp(xs_out, xs_w, warped[:, c])
                         for c in range(imu.shape[1])], axis=1).astype(np.float32)

    # ── Image augmentation ────────────────────────────────────────────────────
    img = img.copy()

    # Small translation
    dy = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    dx = int(rng.integers(-AUG_IMG_SHIFT, AUG_IMG_SHIFT + 1))
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy > 0: img[:dy, :] = 0
    elif dy < 0: img[dy:, :] = 0
    if dx > 0: img[:, :dx] = 0
    elif dx < 0: img[:, dx:] = 0

    # Small rotation (scipy rotates around center, reshape=False pads with 0)
    angle = rng.uniform(-AUG_IMG_ROT_DEG, AUG_IMG_ROT_DEG)
    img = ndimage.rotate(img, angle, reshape=False, mode='constant', cval=0.0)

    # Elastic deformation (only sometimes, to keep training stable)
    if rng.random() < 0.5:
        img = elastic_deform(img, alpha=AUG_ELASTIC_ALPHA, rng=rng)
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return imu, img


def augment_dataset(X_train, y_train, seed=0):
    rng = np.random.default_rng(seed)
<<<<<<< HEAD
    Xs, Yrs, Yts = [X_tr], [yr_tr], [yt_tr]
    for k in range(AUGMENT_FACTOR):
        Xc = np.zeros_like(X_tr)
        Yrc = np.zeros_like(yr_tr)
        Ytc = np.zeros_like(yt_tr)
        for i in range(len(X_tr)):
            Xc[i], Yrc[i], Ytc[i] = augment_sample(X_tr[i], yr_tr[i], yt_tr[i], rng)
        Xs.append(Xc)
        Yrs.append(Yrc)
        Yts.append(Ytc)
        print(f"  Augmentation pass {k + 1}/{AUGMENT_FACTOR} done")
    return (np.concatenate(Xs, 0), np.concatenate(Yrs, 0), np.concatenate(Yts, 0))
=======
    X_out = [X_train]
    y_out = [y_train]
    for _ in range(AUGMENT_FACTOR):
        X_copy = np.zeros_like(X_train)
        y_copy = np.zeros_like(y_train)
        for i in range(len(X_train)):
            X_copy[i], y_copy[i] = augment_sample(X_train[i], y_train[i], rng)
        X_out.append(X_copy)
        y_out.append(y_copy)
    return np.concatenate(X_out, axis=0), np.concatenate(y_out, axis=0)
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611


# ── Model ──────────────────────────────────────────────────────────────────────

<<<<<<< HEAD

class ResBlock1D(nn.Module):
    def __init__(self, ch, ks=3, dil=1):
        super().__init__()
        pad = dil * (ks - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, ks, padding=pad, dilation=dil),
            nn.GroupNorm(1, ch),
        )

    def forward(self, x):
        return F.gelu(self.net(x) + x)


class IMUToImage(nn.Module):
    """
    v19: IMU → 64×64 logit map trained against skeleton GT.
    Encoder: 1D ResBlocks → global avg-pool → bottleneck
    Decoder: 4×4 seed → transposed-conv × 4 → 64×64
    """

    def __init__(self, n_feat, hidden=HIDDEN_DIM):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.enc = nn.Sequential(
            ResBlock1D(hidden, dil=1),
            ResBlock1D(hidden, dil=2),
            ResBlock1D(hidden, dil=4),
            ResBlock1D(hidden, dil=8),
            ResBlock1D(hidden, dil=1),
            ResBlock1D(hidden, dil=2),
        )
        self.drop = nn.Dropout(0.3)
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 256 * 4 * 4),
        )
        # 4×4 → 8 → 16 → 32 → 64
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        B = x.shape[0]
        h = self.proj(x).permute(0, 2, 1)
        h = self.enc(h)
        h = self.drop(h)
        h = h.mean(dim=2)
        seed = self.bottleneck(h).view(B, 256, 4, 4)
        logits = self.dec(seed).squeeze(1)  # (B, 64, 64)
        return logits


# ── Loss ───────────────────────────────────────────────────────────────────────


def dice_loss(pred_sigmoid, gt, eps=1e-6):
    """Sørensen–Dice loss averaged over the batch.
=======
def build_model(input_shape, image_size):
    """
    1D-CNN encoder → convolutional upsampling decoder.

    Encoder:  strided Conv1D stack  (150,7) → (19,128)
    Pooling:  learned temporal attention  (19,128) → (128,)
    Seed:     Dense → (8,8,32)
    Decoder:  three ×2 UpSampling2D stages → (64,64,1)
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611

    Total params: ~720k
    """
<<<<<<< HEAD
    inter = (pred_sigmoid * gt).sum(dim=(-2, -1))
    card = pred_sigmoid.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + eps) / (card + eps)).mean()
=======
    inputs = keras.Input(shape=input_shape, name='imu_input')   # (150, 7)

    # ── Encoder: 1D CNN with strided convolutions ─────────────────────────────
    # Each Conv1D(stride=2) halves the time axis
    x = layers.Conv1D(32, 5, strides=1, padding='same', activation='relu')(inputs)   # (150, 32)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 5, strides=2, padding='same', activation='relu')(x)        # (75, 64)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(96, 3, strides=2, padding='same', activation='relu')(x)        # (38, 96)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, strides=2, padding='same', activation='relu')(x)       # (19, 128)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # ── Temporal attention pooling → (batch, 128) ─────────────────────────────
    # Compute a scalar attention score per timestep, softmax over the time axis,
    # then take the weighted sum across time.
    attn_logits = layers.Dense(1)(x)                                      # (T', 1)
    attn_scores = layers.Lambda(
        lambda t: tf.nn.softmax(t, axis=1))(attn_logits)                  # softmax over T'
    x = layers.Multiply()([x, attn_scores])                               # (T', 128)
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)             # (128,)

    # ── Bottleneck: map to spatial seed ───────────────────────────────────────
    # 8×8×32 = 2048 — much smaller than previous 8×8×128 = 8192
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8 * 8 * 32, activation='relu')(x)
    x = layers.Reshape((8, 8, 32))(x)

    # ── Decoder: three ×2 upsampling stages ───────────────────────────────────
    # 8×8 → 16×16
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 16×16 → 32×32
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 32×32 → 64×64
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(16, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(8, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(x)
    outputs = layers.Reshape((image_size, image_size))(outputs)

    return keras.Model(inputs, outputs, name='handwriting_generator_v2')
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611


# ── Loss ───────────────────────────────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.75):
    """
<<<<<<< HEAD
    pred_sig = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight)
    dice = dice_loss(pred_sig, gt)
    return DICE_WEIGHT * dice + (1.0 - DICE_WEIGHT) * bce


# ── Snapshot ──────────────────────────────────────────────────────────────────

SNAPSHOT_EVERY = 10


def save_snapshot(model, X_snap, yr_snap, yt_snap, lbl_snap, device, epoch, tr_loss):
    model.eval()
    n_show = min(6, len(X_snap))
    with torch.no_grad():
        X_t = torch.from_numpy(X_snap[:n_show]).float().to(device)
        logits = model(X_t)
        pred = torch.sigmoid(logits).cpu().numpy()  # raw probabilities

    # 4 columns: GT photo | GT strokes | raw prob (sigmoid) | hard threshold
    fig, axes = plt.subplots(n_show, 4, figsize=(12, 2.5 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    axes[0, 0].set_title("GT (raw photo)", fontsize=7)
    axes[0, 1].set_title("GT (stroke target)", fontsize=7)
    axes[0, 2].set_title("Pred (raw sigmoid)", fontsize=7)
    axes[0, 3].set_title(f"Pred >{SNAP_THRESHOLD}", fontsize=7)
    fig.suptitle(f"Epoch {epoch + 1}  |  train loss={tr_loss:.4f}", fontsize=10)

    for pi in range(n_show):
        p_soft = pred[pi]  # [0,1] continuous
        p_hard = (p_soft >= SNAP_THRESHOLD).astype(np.float32)  # binary
        axes[pi, 0].imshow(yr_snap[pi], cmap="gray", vmin=0, vmax=1)
        axes[pi, 1].imshow(yt_snap[pi], cmap="gray", vmin=0, vmax=1)
        axes[pi, 2].imshow(p_soft, cmap="hot", vmin=0, vmax=1)  # heatmap
        axes[pi, 3].imshow(p_hard, cmap="gray", vmin=0, vmax=1)
        axes[pi, 0].set_ylabel(
            lbl_snap[pi].split("/")[-1], fontsize=5, rotation=0, labelpad=45
        )
        for c in range(4):
            axes[pi, c].set_xticks([])
            axes[pi, c].set_yticks([])

    plt.tight_layout()
    p = MODELS_DIR / f"snapshot_epoch_{epoch + 1:03d}.png"
    plt.savefig(p, dpi=100)
    plt.close()
    print(f"  Snapshot saved: {p.name}")
    model.train()
=======
    Binary focal loss.
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611

    alpha: weight for the positive (stroke) class.  At ~2.5% stroke density,
           alpha=0.75 gives strokes 3× more weight than background.
    gamma: focusing parameter.  gamma=2 strongly down-weights easy negatives
           (background pixels the model already predicts ≈0 correctly).

    This replaces weighted-BCE + MSE.  Focal loss handles extreme class
    imbalance far better because the down-weighting is per-pixel and
    adaptive, not a single scalar applied uniformly.
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Binary cross-entropy per pixel
        bce = -(y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))
        # Probability of correct class
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        # Focal weight
        focal_weight = tf.pow(1.0 - p_t, gamma)
        # Alpha weight
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(alpha_t * focal_weight * bce)
    return loss


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
<<<<<<< HEAD
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Load existing best_side_mount_model.pt and continue training",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading dataset...")
    X, y_target, y_raw, labels = load_dataset()
    print(f"  Loaded {len(X)} samples  (target density: {y_target.mean():.4f})")

    # ── Scale IMU ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    n, t, f = X.shape
    X_sc = scaler.fit_transform(X.reshape(-1, f)).reshape(n, t, f)
    with open(MODELS_DIR / "scaler_side_mount.pkl", "wb") as f_out:
=======
    print('=' * 60)
    print('Side-Mount Handwriting Generation — Training v2')
    print('=' * 60)

    print('\nLoading dataset...')
    X, y, labels = load_dataset()

    if len(X) == 0:
        print('No data found. Check CSV_DIR and IMG_DIR paths.')
        return

    print(f'Loaded {len(X)} samples')
    print(f'X shape: {X.shape}   y shape: {y.shape}')

    padded        = sum(1 for x in X if np.all(x[-1] == 0))
    stroke_density = y.mean()
    print(f'Samples with end-padding: {padded} / {len(X)} ({100*padded/len(X):.1f}%)')
    print(f'Mean stroke density: {stroke_density:.3f} ({100*stroke_density:.1f}% pixels)')

    # Filter out blank images (< 0.5% stroke) — they are uninformative and
    # reinforce the "predict all black" solution
    valid_mask = y.reshape(len(y), -1).mean(axis=1) >= 0.005
    n_removed  = (~valid_mask).sum()
    if n_removed > 0:
        print(f'Removing {n_removed} near-blank images (< 0.5% stroke density)')
        X      = X[valid_mask]
        y      = y[valid_mask]
        labels = [l for l, v in zip(labels, valid_mask) if v]
    print(f'After filtering: {len(X)} samples')

    # Normalize IMU features
    scaler = StandardScaler()
    n, t, f = X.shape
    X_scaled = scaler.fit_transform(X.reshape(-1, f)).reshape(n, t, f)

    X_train, X_test, y_train, y_test, lbl_train, lbl_test = train_test_split(
        X_scaled, y, labels, test_size=TEST_SIZE, random_state=42
    )
    print(f'Train: {len(X_train)}   Test: {len(X_test)}')

    print(f'Augmenting training set (factor={AUGMENT_FACTOR})...')
    X_train, y_train = augment_dataset(X_train, y_train, seed=42)
    print(f'Augmented train size: {len(X_train)}')

    # Save scaler
    scaler_path = MODELS_DIR / 'scaler_side_mount.pkl'
    with open(scaler_path, 'wb') as f_out:
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611
        pickle.dump(scaler, f_out)
    print(f'Scaler saved to {scaler_path}')

    # Build model
    model = build_model(X_train.shape[1:], IMAGE_SIZE)
    model.summary()
    print(f'Trainable params: {model.count_params():,}')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=['mae'],
    )

    model_path = str(MODELS_DIR / 'best_side_mount_model.keras')
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=40,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=20, min_lr=1e-6,
                                          verbose=1),
        keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss',
                                        save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Training history plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Focal Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['mae'],     label='Train')
    axes[1].plot(history.history['val_mae'], label='Val')
    axes[1].set_title('MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

<<<<<<< HEAD
    # ── Snapshot set: first N samples (visualisation only, also in training) ──
    SNAP_N = min(20, n // 4)
    X_snap = X_sc[:SNAP_N]
    yr_snap = y_raw[:SNAP_N]
    yt_snap = y_target[:SNAP_N]
    lbl_snap = labels[:SNAP_N]

    # ── Augment ───────────────────────────────────────────────────────────────
    print(f"Augmenting training set (×{AUGMENT_FACTOR})...")
    X_sc, y_raw, y_target = augment_dataset(X_sc, y_raw, y_target, seed=42)
    print(f"  Augmented size: {len(X_sc)}")

    pos_weight_t = torch.tensor(BCE_POS_WEIGHT, device=device)
    print(f"  pos_weight={BCE_POS_WEIGHT}  render_width={RENDER_WIDTH}px")

    tr_ds = TensorDataset(
        torch.from_numpy(X_sc).float(), torch.from_numpy(y_target).float()
    )
    tr_ld = DataLoader(
        tr_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0
    )

    n_feat = X_sc.shape[2]
    model = IMUToImage(n_feat).to(device)
    print(
        f"\nModel: IMUToImage (v20)  "
        f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    model_pt = str(MODELS_DIR / "best_side_mount_model.pt")

    if args.resume:
        if Path(model_pt).exists():
            model.load_state_dict(
                torch.load(model_pt, weights_only=True, map_location=device)
            )
            print(f"  Resumed from {model_pt}")
        else:
            print(
                f"  WARNING: --resume passed but no saved model found at {model_pt}; starting fresh"
            )

    TARGET_LR = 3e-4
    WARMUP = 10
    PATIENCE = 50
    opt = torch.optim.AdamW(
        model.parameters(), lr=TARGET_LR / WARMUP, weight_decay=1e-4
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=2, eta_min=1e-6
    )

    best_loss = float("inf")
    pat_count = 0
    history = {"loss": []}

    print(f"\nTraining for {EPOCHS} epochs...\n")
    for epoch in range(EPOCHS):
        if epoch < WARMUP:
            for pg in opt.param_groups:
                pg["lr"] = TARGET_LR * (epoch + 1) / WARMUP

        model.train()
        tr_sum = tr_n = 0
        for x_b, yt_b in tr_ld:
            x_b = x_b.to(device)
            yt_b = yt_b.to(device)
            opt.zero_grad()
            loss = image_loss(model(x_b), yt_b, pos_weight_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item() * x_b.size(0)
            tr_n += x_b.size(0)
        tr_loss = tr_sum / tr_n
        history["loss"].append(tr_loss)

        if epoch >= WARMUP:
            sch.step(epoch - WARMUP)
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:3d}/{EPOCHS}  loss={tr_loss:.4f}  lr={lr:.2e}")

        # Save on training loss (all data is training — no val split)
        if epoch >= WARMUP:
            if tr_loss < best_loss:
                best_loss = tr_loss
                pat_count = 0
                torch.save(model.state_dict(), model_pt)
            else:
                pat_count += 1
                if pat_count >= PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        if (epoch + 1) % SNAPSHOT_EVERY == 0 or epoch == 0:
            save_snapshot(
                model, X_snap, yr_snap, yt_snap, lbl_snap, device, epoch, tr_loss
            )

    model.load_state_dict(torch.load(model_pt, weights_only=True))
    model.eval()

    # ── Training history ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["loss"], label="Train BCE")
    ax.set_title("BCE Loss (rendered stroke GT)")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "training_history_side_mount.png", dpi=150)
    plt.close()

    # ── Final predictions on snapshot set ─────────────────────────────────────
    n_show = min(10, SNAP_N)
    with torch.no_grad():
        logits_f = model(torch.from_numpy(X_snap[:n_show]).float().to(device))
        pred_f = torch.sigmoid(logits_f).cpu().numpy()

    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    axes[0, 0].set_title("GT (raw)", fontsize=9)
    axes[0, 1].set_title("GT (stroke)", fontsize=9)
    axes[0, 2].set_title("Predicted", fontsize=9)

    for pi in range(n_show):
        p_hard = (pred_f[pi] >= SNAP_THRESHOLD).astype(np.float32)
        axes[pi, 0].imshow(yr_snap[pi], cmap="gray", vmin=0, vmax=1)
        axes[pi, 1].imshow(yt_snap[pi], cmap="gray", vmin=0, vmax=1)
        axes[pi, 2].imshow(p_hard, cmap="gray", vmin=0, vmax=1)
        axes[pi, 0].set_ylabel(
            lbl_snap[pi].split("/")[-1], fontsize=6, rotation=0, labelpad=55
        )
        for c in range(3):
            axes[pi, c].set_xticks([])
            axes[pi, c].set_yticks([])

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "predictions_side_mount.png", dpi=150)
    plt.close()
    print("Predictions saved.")

    best_ep = int(np.argmin(history["loss"]))
    print(f"\nBest train BCE: {history['loss'][best_ep]:.4f} at epoch {best_ep + 1}")
    print("=" * 60)
    print(f"Done. Model: {model_pt}")
=======
    plt.tight_layout()
    hist_path = MODELS_DIR / 'training_history_side_mount.png'
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f'Training history saved to {hist_path}')

    # ── Prediction grid: GT | raw pred | threshold 0.2 | threshold 0.4 ───────
    preds  = model.predict(X_test)
    n_show = min(8, len(X_test))
    fig, axes = plt.subplots(n_show, 4, figsize=(12, 3 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)
    titles = ['Ground Truth', 'Predicted (raw)', 'Thresh >0.2', 'Thresh >0.4']
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=9)
    for i in range(n_show):
        axes[i, 0].imshow(y_test[i],                          cmap='gray', vmin=0, vmax=1)
        axes[i, 1].imshow(preds[i],                           cmap='gray', vmin=0, vmax=1)
        axes[i, 2].imshow((preds[i] > 0.2).astype(float),    cmap='gray')
        axes[i, 3].imshow((preds[i] > 0.4).astype(float),    cmap='gray')
        axes[i, 0].set_ylabel(lbl_test[i].split('/')[-1], fontsize=6, rotation=0,
                               labelpad=55)
        for col in range(4):
            axes[i, col].axis('off')

    plt.tight_layout()
    pred_path = MODELS_DIR / 'predictions_side_mount.png'
    plt.savefig(pred_path, dpi=150)
    plt.close()
    print(f'Predictions saved to {pred_path}')

    # ── Print peak val stats ──────────────────────────────────────────────────
    best_epoch  = np.argmin(history.history['val_loss'])
    best_val    = history.history['val_loss'][best_epoch]
    best_mae    = history.history['val_mae'][best_epoch]
    pred_max    = preds.max()
    pred_mean   = preds.mean()
    print(f'\nBest val_loss: {best_val:.4f} at epoch {best_epoch}')
    print(f'Best val_mae:  {best_mae:.4f}')
    print(f'Prediction range: mean={pred_mean:.4f}  max={pred_max:.4f}')
    if pred_max < 0.15:
        print('  WARNING: predictions are near-zero — model may not have learned strokes.')
        print('  Try collecting more data or lowering ACTIVITY_THRESHOLD.')

    print('\n' + '=' * 60)
    print(f'Done. Model saved to: {model_path}')
    print('=' * 60)
>>>>>>> a1d4941fc8e3687aef46fe29023e704afda41611


if __name__ == "__main__":
    main()
