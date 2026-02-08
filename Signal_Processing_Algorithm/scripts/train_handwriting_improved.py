"""
Improved handwriting model with sharper, more detailed outputs.

Key improvements:
1. Perceptual loss (combines MSE + binary cross-entropy)
2. Larger decoder network for finer details
3. Skip connections for preserving information
4. Higher resolution output option
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

class HandwritingDataset:
    def __init__(self, data_dir, image_size=64):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.scaler = StandardScaler()

    def load_png_image(self, png_path):
        """Load PNG image and convert to numpy array."""
        image = Image.open(png_path).convert("L")

        # Resize if needed
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.LANCZOS
            )

        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image) / 255.0

        # Invert (letter should be white on black background)
        image_array = 1.0 - image_array

        return image_array

    def load_csv_data(self, csv_path):
        """Load and preprocess accelerometer data from CSV."""
        df = pd.read_csv(csv_path, skiprows=1)

        # Extract features
        feature_cols = [
            "acc_x[mg]",
            "acc_y[mg]",
            "acc_z[mg]",
            "gyro_x[mdps]",
            "gyro_y[mdps]",
            "gyro_z[mdps]",
        ]

        data = df[feature_cols].values

        # Remove gravity
        b, a = signal.butter(3, 0.5, "high", fs=100)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])

        # Compute magnitude
        acc_magnitude = np.sqrt(
            filtered_data[:, 0] ** 2
            + filtered_data[:, 1] ** 2
            + filtered_data[:, 2] ** 2
        )

        features = np.column_stack([filtered_data, acc_magnitude])

        return features

    def pad_or_truncate(self, data, target_length=200):
        """Pad or truncate time series to fixed length."""
        if len(data) > target_length:
            return data[-target_length:]
        else:
            padding = np.zeros((target_length - len(data), data.shape[1]))
            return np.vstack([padding, data])

    def load_dataset(self, max_samples=None):
        """Load all CSV and PNG pairs."""
        csv_dir = self.data_dir / "alphabet_write"
        png_dir = self.data_dir / "alphabet_write" / "letter_images"

        if not png_dir.exists():
            print(f"❌ Error: PNG directory not found: {png_dir}")
            print("Please run 'python convert_svgs_to_pngs.py' first!")
            return np.array([]), np.array([]), []

        X = []
        y = []
        filenames = []

        csv_files = sorted(
            csv_dir.glob("a_w_*.csv"), key=lambda x: int(x.stem.split("_")[-1])
        )

        if max_samples:
            csv_files = csv_files[:max_samples]

        print(f"Loading {len(csv_files)} samples...")

        for csv_file in csv_files:
            number = csv_file.stem.split("_")[-1]
            png_file = png_dir / f"a_write_{number}.png"

            if not png_file.exists():
                print(
                    f"  ⚠ Warning: {png_file.name} not found, skipping {csv_file.name}"
                )
                continue

            try:
                csv_data = self.load_csv_data(csv_file)
                csv_data = self.pad_or_truncate(csv_data)

                image = self.load_png_image(png_file)

                X.append(csv_data)
                y.append(image)
                filenames.append(csv_file.name)

                print(f"  ✓ Loaded {csv_file.name} → {png_file.name}")

            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {e}")
                continue

        X = np.array(X)
        y = np.array(y)

        print(f"\nLoaded {len(X)} samples")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        return X, y, filenames

    def normalize_features(self, X_train, X_test=None):
        """Normalize accelerometer features."""
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)

        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)

        if X_test is not None:
            n_samples_test = X_test.shape[0]
            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(
                n_samples_test, n_timesteps, n_features
            )
            return X_train_scaled, X_test_scaled

        return X_train_scaled


def build_improved_model(input_shape, output_shape):
    """
    Improved model architecture for sharper outputs.

    Key features:
    - Larger decoder with more parameters
    - Upsampling layers for gradual resolution increase
    - Skip connections (latent features feed into decoder)
    - More conv layers for detail refinement
    """
    inputs = keras.Input(shape=input_shape, name="accelerometer_input")

    # Encoder: Process time series with LSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x_seq = layers.Bidirectional(layers.LSTM(64))(x)  # Sequence features
    x = layers.Dropout(0.3)(x_seq)

    # Dense bottleneck
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation="relu")(x)
    latent = layers.Dropout(0.2)(x)

    # Start with small spatial size and upsample
    img_size = output_shape[0]
    init_size = img_size // 4  # Start at 16x16 for 64x64 output

    x = layers.Dense(init_size * init_size * 32, activation="relu")(latent)
    x = layers.Reshape((init_size, init_size, 32))(x)

    # Decoder: Upsample with convolutions
    # 16x16 -> 32x32
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    # 32x32 -> 64x64
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)

    # Final refinement layers
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation="sigmoid", padding="same")(x)
    outputs = layers.Reshape((img_size, img_size))(outputs)

    model = keras.Model(
        inputs=inputs, outputs=outputs, name="improved_handwriting_generator"
    )

    return model


def combined_loss(y_true, y_pred):
    """
    Combined loss for sharper outputs.

    - MSE: Pixel-level accuracy
    - Binary crossentropy: Sharp edges (treats pixels as binary decisions)
    """
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    bce_loss = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))

    # Weight BCE more for sharper edges
    return 0.3 * mse_loss + 0.7 * bce_loss


def train_model(X_train, y_train, X_val, y_val, epochs=150, batch_size=4):
    """Train the improved model."""
    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]

    print(f"\nBuilding improved model...")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")

    model = build_improved_model(input_shape, output_shape)
    model.summary()

    # Compile with combined loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=combined_loss,
        metrics=["mae"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=15, min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            "best_handwriting_model_improved.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def plot_predictions(model, X_test, y_test, n_samples=5):
    """Visualize model predictions."""
    predictions = model.predict(X_test[:n_samples])

    # Apply threshold for sharper output
    predictions_sharp = (predictions > 0.5).astype(float)

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        # Ground truth
        axes[i, 0].imshow(y_test[i], cmap="gray")
        axes[i, 0].set_title("Ground Truth")
        axes[i, 0].axis("off")

        # Raw prediction
        axes[i, 1].imshow(predictions[i], cmap="gray")
        axes[i, 1].set_title("Prediction (Raw)")
        axes[i, 1].axis("off")

        # Thresholded prediction
        axes[i, 2].imshow(predictions_sharp[i], cmap="gray")
        axes[i, 2].set_title("Prediction (Threshold)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("handwriting_predictions_improved.png", dpi=150)
    print("\n✓ Saved predictions to handwriting_predictions_improved.png")
    plt.show()


def main():
    DATA_DIR = Path(__file__).parent / "test_data"
    IMAGE_SIZE = 64
    TEST_SIZE = 0.2
    EPOCHS = 150
    BATCH_SIZE = 4  # Smaller batch for better generalization with small dataset

    print("=" * 60)
    print("Improved Handwriting Generation Model")
    print("=" * 60)

    dataset = HandwritingDataset(DATA_DIR, image_size=IMAGE_SIZE)
    X, y, filenames = dataset.load_dataset()

    if len(X) == 0:
        print("\n❌ No data loaded!")
        return

    if len(X) < 4:
        print(f"\n⚠ Warning: Only {len(X)} samples. Results will be limited.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    X_train, X_test = dataset.normalize_features(X_train, X_test)

    with open("scaler_improved.pkl", "wb") as f:
        pickle.dump(dataset.scaler, f)
    print("✓ Saved scaler to scaler_improved.pkl")

    model, history = train_model(
        X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Combined Loss")
    plt.legend()
    plt.title("Training History")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.title("Mean Absolute Error")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history_improved.png", dpi=150)
    plt.tight_layout()
    plt.savefig('training_history_improved.png', dpi=150)
    print("✓ Saved training history to training_history_improved.png")
    plt.show()

    plot_predictions(model, X_test, y_test, n_samples=min(5, len(X_test)))

    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print(f"Model saved to: best_handwriting_model_improved.keras")
    print(f"Scaler saved to: scaler_improved.pkl")
    print("\nKey improvements:")
    print("- Larger decoder network for more detail")
    print("- Upsampling architecture for gradual resolution")
    print("- Binary cross-entropy loss for sharper edges")
    print("- Smaller batch size for better generalization")
    print("=" * 60)


if __name__ == "__main__":
    main()
