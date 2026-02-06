"""
CNN-based Letter Classification using Accelerometer Data
Trains a 1D Convolutional Neural Network on individual letter samples from alphabet folder
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Deep learning imports
try:
    from tensorflow import keras
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import to_categorical

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print(
        "WARNING: TensorFlow not installed. Please install with: pip install tensorflow"
    )


class LetterDataLoader:
    """Loads and preprocesses accelerometer data for letter classification"""

    def __init__(self, data_dir, target_length=200, sampling_rate=104):
        """
        Args:
            data_dir: Directory containing CSV files
            target_length: Fixed sequence length for CNN input (padding/truncation)
            sampling_rate: Sensor sampling rate in Hz
        """
        self.data_dir = data_dir
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.label_encoder = LabelEncoder()

    def read_csv_file(self, filepath):
        """Read CSV file and extract accelerometer data"""
        try:
            # Read CSV, skipping comment lines starting with #
            df = pd.read_csv(filepath, comment="#")

            # Check if file has enough data (filter out corrupted single-row files)
            if len(df) <= 1:
                return None, "Insufficient data (single row)"

            # Extract acceleration columns (convert from milligravity to m/s^2)
            if "acc_x[mg]" in df.columns:
                acc_x = df["acc_x[mg]"].values / 1000.0 * 9.81  # mg to m/s^2
                acc_y = df["acc_y[mg]"].values / 1000.0 * 9.81
                acc_z = df["acc_z[mg]"].values / 1000.0 * 9.81
            else:
                return None, "Missing acceleration columns"

            return np.stack([acc_x, acc_y, acc_z], axis=1), None

        except Exception as e:
            return None, f"Error reading file: {str(e)}"

    def preprocess_signal(self, acc_data):
        """Apply preprocessing: filtering, gravity removal, normalization"""
        if acc_data is None or len(acc_data) == 0:
            return None

        try:
            # 1. Remove gravity using high-pass Butterworth filter (cutoff 0.5 Hz)
            nyquist = self.sampling_rate / 2
            cutoff = 0.5 / nyquist
            b, a = signal.butter(3, cutoff, "high")

            filtered = np.zeros_like(acc_data)
            for i in range(3):  # Filter each axis
                filtered[:, i] = signal.filtfilt(b, a, acc_data[:, i])

            # 2. Pad or truncate to fixed length
            if len(filtered) < self.target_length:
                # Pad with zeros at the end
                padding = np.zeros((self.target_length - len(filtered), 3))
                filtered = np.vstack([filtered, padding])
            else:
                # Truncate to target length
                filtered = filtered[: self.target_length, :]

            # 3. Normalize using StandardScaler (per-axis)
            scaler = StandardScaler()
            normalized = scaler.fit_transform(filtered)

            return normalized

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def extract_label_from_filename(self, filename):
        """Extract letter label from filename (e.g., 'a1.csv' -> 'a')"""
        # Remove extension and any numbers/special chars to get the letter
        letter = filename[0].lower()
        if letter.isalpha():
            return letter
        return None

    def load_dataset(self):
        """Load all CSV files and create dataset"""
        X_data = []
        y_labels = []
        skipped_files = []

        print(f"Loading data from: {self.data_dir}")
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        print(f"Found {len(csv_files)} CSV files")

        for i, filename in enumerate(csv_files):
            if (i + 1) % 50 == 0:
                print(f"Processing file {i + 1}/{len(csv_files)}...")

            filepath = os.path.join(self.data_dir, filename)

            # Extract label
            label = self.extract_label_from_filename(filename)
            if label is None:
                skipped_files.append((filename, "Invalid filename format"))
                continue

            # Read and preprocess
            raw_data, error = self.read_csv_file(filepath)
            if raw_data is None:
                skipped_files.append((filename, error))
                continue

            processed_data = self.preprocess_signal(raw_data)
            if processed_data is None:
                skipped_files.append((filename, "Preprocessing failed"))
                continue

            X_data.append(processed_data)
            y_labels.append(label)

        print(f"\nSuccessfully loaded {len(X_data)} samples")
        print(f"Skipped {len(skipped_files)} files")

        if skipped_files:
            print("\nSkipped files:")
            for fname, reason in skipped_files[:10]:  # Show first 10
                print(f"  - {fname}: {reason}")
            if len(skipped_files) > 10:
                print(f"  ... and {len(skipped_files) - 10} more")

        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_labels)

        # Encode labels (a-z -> 0-25)
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"\nDataset shape: {X.shape}")
        print(f"Labels shape: {y_encoded.shape}")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")

        # Print class distribution
        unique, counts = np.unique(y_encoded, return_counts=True)
        print("\nClass distribution:")
        for cls, count in zip(self.label_encoder.classes_, counts):
            print(f"  {cls.upper()}: {count} samples")

        return X, y_encoded, y


class LetterCNN:
    """1D CNN for letter classification from accelerometer data"""

    def __init__(self, input_shape, num_classes=26):
        """
        Args:
            input_shape: (timesteps, features) e.g., (200, 3)
            num_classes: Number of letter classes (26 for A-Z)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self):
        """Build 1D CNN architecture"""
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is required. Install with: pip install tensorflow"
            )

        model = Sequential(
            [
                # First convolutional block
                Conv1D(
                    64, kernel_size=5, activation="relu", input_shape=self.input_shape
                ),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.25),
                # Second convolutional block
                Conv1D(128, kernel_size=5, activation="relu"),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.25),
                # Third convolutional block
                Conv1D(256, kernel_size=3, activation="relu"),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Dropout(0.3),
                # Dense layers
                Flatten(),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation="relu"),
                Dropout(0.4),
                Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        self.model = model
        print("\nModel Architecture:")
        model.summary()

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the CNN model"""
        if self.model is None:
            self.build_model()

        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)

        # Compute class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight

        class_weights_array = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = dict(enumerate(class_weights_array))

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
            ),
            ModelCheckpoint(
                "best_letter_cnn_model.keras",
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        print("\nStarting training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        self.history = self.model.fit(
            X_train,
            y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        return self.history

    def evaluate(self, X_test, y_test, label_encoder):
        """Evaluate model and generate metrics"""
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)

        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calculate metrics
        test_loss, test_acc = self.model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Classification report
        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=[c.upper() for c in label_encoder.classes_],
                zero_division=0,
            )
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        return y_pred, cm, test_acc

    def plot_training_history(self, save_path="training_history.png"):
        """Plot training and validation accuracy/loss"""
        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy plot
        ax1.plot(self.history.history["accuracy"], label="Train Accuracy")
        ax1.plot(self.history.history["val_accuracy"], label="Val Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        # Loss plot
        ax2.plot(self.history.history["loss"], label="Train Loss")
        ax2.plot(self.history.history["val_loss"], label="Val Loss")
        ax2.set_title("Model Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to: {save_path}")
        plt.close()

    def plot_confusion_matrix(
        self, cm, label_encoder, save_path="confusion_matrix.png"
    ):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[c.upper() for c in label_encoder.classes_],
            yticklabels=[c.upper() for c in label_encoder.classes_],
            cbar_kws={"label": "Count"},
        )
        plt.title("Confusion Matrix - Letter Classification")
        plt.xlabel("Predicted Letter")
        plt.ylabel("True Letter")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("CNN Letter Classification from Accelerometer Data")
    print("=" * 70)

    # Configuration
    DATA_DIR = r"C:\Users\sekan\Documents\Senior_Design\Test_Data\alphabet"
    TARGET_LENGTH = 200  # Fixed sequence length
    SAMPLING_RATE = 104  # Hz
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2  # 20% of training data for validation
    RANDOM_SEED = 42

    # 1. Load and preprocess data
    print("\n" + "=" * 70)
    print("Step 1: Loading and Preprocessing Data")
    print("=" * 70)

    loader = LetterDataLoader(DATA_DIR, TARGET_LENGTH, SAMPLING_RATE)
    X, y_encoded, y_raw = loader.load_dataset()

    if len(X) == 0:
        print("ERROR: No valid data loaded. Exiting.")
        return

    # Filter out classes with too few samples (need at least 4 for stratified split)
    print("\n" + "=" * 70)
    print("Step 1.5: Filtering Classes with Insufficient Samples")
    print("=" * 70)

    MIN_SAMPLES_PER_CLASS = 4
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    valid_classes = unique_classes[class_counts >= MIN_SAMPLES_PER_CLASS]

    # Filter data to keep only valid classes
    valid_mask = np.isin(y_encoded, valid_classes)
    X_filtered = X[valid_mask]
    y_filtered = y_encoded[valid_mask]
    y_raw_filtered = y_raw[valid_mask]

    removed_classes = [
        loader.label_encoder.classes_[c]
        for c in unique_classes
        if c not in valid_classes
    ]
    if removed_classes:
        print(
            f"Removed {len(removed_classes)} classes with < {MIN_SAMPLES_PER_CLASS} samples: {removed_classes}"
        )

    # Re-encode labels to be contiguous (0, 1, 2, ...)
    label_encoder_filtered = LabelEncoder()
    y_encoded_filtered = label_encoder_filtered.fit_transform(y_raw_filtered)

    print(
        f"Filtered dataset: {X_filtered.shape[0]} samples across {len(label_encoder_filtered.classes_)} classes"
    )
    print(f"Remaining classes: {label_encoder_filtered.classes_}")

    # Update loader's label encoder for later use
    loader.label_encoder = label_encoder_filtered

    # 2. Split data (stratified by letter)
    print("\n" + "=" * 70)
    print("Step 2: Splitting Data")
    print("=" * 70)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_filtered,
        y_encoded_filtered,
        test_size=TEST_SIZE,
        stratify=y_encoded_filtered,
        random_state=RANDOM_SEED,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        stratify=y_train_full,
        random_state=RANDOM_SEED,
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 3. Build and train CNN
    print("\n" + "=" * 70)
    print("Step 3: Building and Training CNN")
    print("=" * 70)

    if not HAS_TENSORFLOW:
        print("ERROR: TensorFlow not installed. Cannot train CNN.")
        print("Install with: pip install tensorflow")
        return

    input_shape = (TARGET_LENGTH, 3)  # (timesteps, features)
    num_classes = len(loader.label_encoder.classes_)

    cnn = LetterCNN(input_shape, num_classes)
    cnn.build_model()

    history = cnn.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

    # 4. Evaluate on test set
    print("\n" + "=" * 70)
    print("Step 4: Evaluating on Test Set")
    print("=" * 70)

    y_pred, cm, test_acc = cnn.evaluate(X_test, y_test, loader.label_encoder)

    # 5. Generate visualizations
    print("\n" + "=" * 70)
    print("Step 5: Generating Visualizations")
    print("=" * 70)

    output_dir = (
        r"C:\Users\sekan\Documents\Senior_Design\Signal_Processing_Algorithm\output"
    )
    os.makedirs(output_dir, exist_ok=True)

    cnn.plot_training_history(os.path.join(output_dir, "cnn_training_history.png"))
    cnn.plot_confusion_matrix(
        cm, loader.label_encoder, os.path.join(output_dir, "cnn_confusion_matrix.png")
    )

    # 6. Save model
    print("\n" + "=" * 70)
    print("Step 6: Saving Model")
    print("=" * 70)

    model_path = os.path.join(output_dir, "final_letter_cnn_model.keras")
    cnn.model.save(model_path)
    print(f"Model saved to: {model_path}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Total samples trained on: {len(X_train)}")
    print(f"Model and visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
