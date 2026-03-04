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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy import signal, ndimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT  = SCRIPT_DIR / '..' / '..' / 'Test_Data' / 'side_mount'
CSV_DIR    = DATA_ROOT / 'split_csvs'
IMG_DIR    = DATA_ROOT / 'split_images'
MODELS_DIR = SCRIPT_DIR / '..' / 'models'
MODELS_DIR.mkdir(exist_ok=True)

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


# ── Data loading ───────────────────────────────────────────────────────────────

def extract_active_segment(features, magnitude):
    """
    Find the writing segment within a raw 208-sample box and return a
    (N_TIMESTEPS, n_features) float32 array.
    """
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
    """Load one box CSV → (N_TIMESTEPS, 7) float32 array."""
    df = pd.read_csv(path, skiprows=1)
    data = df[FEATURE_COLS].values.astype(np.float32)

    b, a = signal.butter(3, 0.5, 'high', fs=104)
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, data[:, i])

    magnitude = np.sqrt((filtered[:, :3] ** 2).sum(axis=1))
    features  = np.hstack([filtered, magnitude[:, None]])   # (N, 7)
    return extract_active_segment(features, magnitude)


def load_image(path):
    """Load one box PNG → (IMAGE_SIZE, IMAGE_SIZE) float32 in [0,1], strokes=1."""
    img = Image.open(path).convert('L')
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr


def build_img_dir_map():
    mapping = {}
    for img_dir in IMG_DIR.iterdir():
        if not img_dir.is_dir():
            continue
        key = re.sub(r'_\d+$', '', img_dir.name)
        mapping[key] = img_dir
    return mapping


def load_dataset():
    X, y, labels = [], [], []
    missing = 0
    img_dir_map = build_img_dir_map()

    for csv_page_dir in sorted(CSV_DIR.iterdir()):
        if not csv_page_dir.is_dir():
            continue
        page_stem    = csv_page_dir.name
        img_page_dir = img_dir_map.get(page_stem)
        if img_page_dir is None:
            print(f'  WARNING: no image folder for {page_stem}, skipping')
            continue

        for csv_path in sorted(csv_page_dir.glob('box_*.csv')):
            box_name = csv_path.stem
            img_path = img_page_dir / f'{box_name}.png'
            if not img_path.exists():
                missing += 1
                continue
            try:
                X.append(load_csv(csv_path))
                y.append(load_image(img_path))
                labels.append(f'{page_stem}/{box_name}')
            except Exception as e:
                print(f'  ERROR loading {page_stem}/{box_name}: {e}')

    if missing:
        print(f'  Skipped {missing} CSV files with no matching image')
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), labels


# ── Augmentation ───────────────────────────────────────────────────────────────

def elastic_deform(image, alpha, sigma=4.0, rng=None):
    """
    Apply elastic deformation to a 2D image array.
    alpha controls displacement magnitude, sigma controls smoothness.
    """
    if rng is None:
        rng = np.random.default_rng()
    shape = image.shape
    dx = ndimage.gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    dy = ndimage.gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    coords = [np.clip(y + dy, 0, shape[0]-1),
              np.clip(x + dx, 0, shape[1]-1)]
    return ndimage.map_coordinates(image, coords, order=1, mode='reflect')


def augment_sample(imu, img, rng):
    """
    Apply random augmentation to one (imu, img) pair.

    IMU: amplitude scale, Gaussian noise, time warp.
    Image: small translation, small rotation, elastic deformation.
    NO horizontal flip — asymmetric letters (b/d/p/q) would be mislabeled.
    """
    imu = imu.copy()

    # Amplitude scale
    imu *= rng.uniform(*AUG_SCALE_RANGE)

    # Gaussian noise
    imu += rng.normal(0, AUG_NOISE_STD, imu.shape).astype(np.float32)

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

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return imu, img


def augment_dataset(X_train, y_train, seed=0):
    rng = np.random.default_rng(seed)
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


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(input_shape, image_size):
    """
    1D-CNN encoder → convolutional upsampling decoder.

    Encoder:  strided Conv1D stack  (150,7) → (19,128)
    Pooling:  learned temporal attention  (19,128) → (128,)
    Seed:     Dense → (8,8,32)
    Decoder:  three ×2 UpSampling2D stages → (64,64,1)

    Total params: ~720k
    """
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


# ── Loss ───────────────────────────────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.75):
    """
    Binary focal loss.

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


if __name__ == '__main__':
    main()
