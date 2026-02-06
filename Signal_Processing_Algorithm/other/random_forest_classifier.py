"""
Random Forest Letter Classification with Engineered Features
Simpler model better suited for small datasets
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class FeatureExtractor:
    """Extract statistical features from accelerometer time series"""

    def __init__(self, sampling_rate=104):
        self.sampling_rate = sampling_rate

    def extract_features(self, acc_data):
        """Extract comprehensive feature set from acceleration data"""
        features = []

        # Process each axis
        for axis in range(3):
            axis_data = acc_data[:, axis]

            # Time domain features
            features.extend(
                [
                    np.mean(axis_data),
                    np.std(axis_data),
                    np.min(axis_data),
                    np.max(axis_data),
                    np.median(axis_data),
                    np.percentile(axis_data, 25),
                    np.percentile(axis_data, 75),
                    kurtosis(axis_data),
                    skew(axis_data),
                    np.ptp(axis_data),  # peak-to-peak
                ]
            )

            # Zero crossings
            zero_crossings = np.sum(np.diff(np.sign(axis_data)) != 0)
            features.append(zero_crossings)

            # Peak detection
            peaks, _ = signal.find_peaks(axis_data)
            features.append(len(peaks))

            # Energy
            features.append(np.sum(axis_data**2))

            # Frequency domain features (FFT)
            fft_vals = np.fft.fft(axis_data)
            fft_mag = np.abs(fft_vals[: len(fft_vals) // 2])
            features.extend(
                [
                    np.mean(fft_mag),
                    np.std(fft_mag),
                    np.max(fft_mag),
                ]
            )

        # Cross-axis features
        magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
        features.extend(
            [
                np.mean(magnitude),
                np.std(magnitude),
                np.max(magnitude),
                np.min(magnitude),
            ]
        )

        # Duration
        features.append(len(acc_data))

        return np.array(features)


class LetterDataLoader:
    """Load and preprocess accelerometer data"""

    def __init__(self, data_dir, sampling_rate=104):
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FeatureExtractor(sampling_rate)

    def read_csv_file(self, filepath):
        """Read CSV file and extract accelerometer data"""
        try:
            df = pd.read_csv(filepath, comment="#")

            if len(df) <= 1:
                return None, "Insufficient data (single row)"

            if "acc_x[mg]" in df.columns:
                acc_x = df["acc_x[mg]"].values / 1000.0 * 9.81
                acc_y = df["acc_y[mg]"].values / 1000.0 * 9.81
                acc_z = df["acc_z[mg]"].values / 1000.0 * 9.81
            else:
                return None, "Missing acceleration columns"

            return np.stack([acc_x, acc_y, acc_z], axis=1), None

        except Exception as e:
            return None, f"Error reading file: {str(e)}"

    def preprocess_signal(self, acc_data):
        """Apply high-pass filter to remove gravity"""
        if acc_data is None or len(acc_data) == 0:
            return None

        try:
            nyquist = self.sampling_rate / 2
            cutoff = 0.5 / nyquist
            b, a = signal.butter(3, cutoff, "high")

            filtered = np.zeros_like(acc_data)
            for i in range(3):
                filtered[:, i] = signal.filtfilt(b, a, acc_data[:, i])

            return filtered

        except Exception as e:
            return None

    def extract_label_from_filename(self, filename):
        """Extract letter label from filename"""
        letter = filename[0].lower()
        if letter.isalpha():
            return letter
        return None

    def load_dataset(self):
        """Load all CSV files and extract features"""
        X_features = []
        y_labels = []
        skipped_files = []

        print(f"Loading data from: {self.data_dir}")
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        print(f"Found {len(csv_files)} CSV files")

        for i, filename in enumerate(csv_files):
            if (i + 1) % 50 == 0:
                print(f"Processing file {i + 1}/{len(csv_files)}...")

            filepath = os.path.join(self.data_dir, filename)

            label = self.extract_label_from_filename(filename)
            if label is None:
                skipped_files.append((filename, "Invalid filename format"))
                continue

            raw_data, error = self.read_csv_file(filepath)
            if raw_data is None:
                skipped_files.append((filename, error))
                continue

            processed_data = self.preprocess_signal(raw_data)
            if processed_data is None:
                skipped_files.append((filename, "Preprocessing failed"))
                continue

            # Extract features
            features = self.feature_extractor.extract_features(processed_data)

            X_features.append(features)
            y_labels.append(label)

        print(f"\nSuccessfully loaded {len(X_features)} samples")
        print(f"Skipped {len(skipped_files)} files")

        X = np.array(X_features)
        y = np.array(y_labels)

        y_encoded = self.label_encoder.fit_transform(y)

        print(f"\nDataset shape: {X.shape}")
        print(f"Feature vector size: {X.shape[1]}")
        print(f"Classes: {self.label_encoder.classes_}")

        return X, y_encoded, y


def main():
    print("=" * 70)
    print("Random Forest Letter Classification")
    print("=" * 70)

    DATA_DIR = r"C:\Users\sekan\Documents\Senior_Design\Test_Data\alphabet"
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    # Load data
    print("\nStep 1: Loading and Extracting Features")
    print("=" * 70)

    loader = LetterDataLoader(DATA_DIR)
    X, y_encoded, y_raw = loader.load_dataset()

    if len(X) == 0:
        print("ERROR: No valid data loaded.")
        return

    # Filter classes with too few samples
    print("\nStep 2: Filtering Classes")
    print("=" * 70)

    MIN_SAMPLES_PER_CLASS = 4
    unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
    valid_classes = unique_classes[class_counts >= MIN_SAMPLES_PER_CLASS]

    valid_mask = np.isin(y_encoded, valid_classes)
    X_filtered = X[valid_mask]
    y_filtered = y_encoded[valid_mask]
    y_raw_filtered = y_raw[valid_mask]

    label_encoder_filtered = LabelEncoder()
    y_encoded_filtered = label_encoder_filtered.fit_transform(y_raw_filtered)

    print(
        f"Filtered dataset: {X_filtered.shape[0]} samples, {len(label_encoder_filtered.classes_)} classes"
    )

    # Split data
    print("\nStep 3: Splitting Data")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered,
        y_encoded_filtered,
        test_size=TEST_SIZE,
        stratify=y_encoded_filtered,
        random_state=RANDOM_SEED,
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train Random Forest
    print("\nStep 4: Training Random Forest")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )

    rf.fit(X_train, y_train)

    # Evaluate
    print("\nStep 5: Evaluating Model")
    print("=" * 70)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[c.upper() for c in label_encoder_filtered.classes_],
            zero_division=0,
        )
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.upper() for c in label_encoder_filtered.classes_],
        yticklabels=[c.upper() for c in label_encoder_filtered.classes_],
    )
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    output_dir = (
        r"C:\Users\sekan\Documents\Senior_Design\Signal_Processing_Algorithm\output"
    )
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "rf_confusion_matrix.png"), dpi=300)
    print(f"\nConfusion matrix saved to: {output_dir}/rf_confusion_matrix.png")
    plt.close()

    # Feature importance
    feature_names = []
    for axis in ["X", "Y", "Z"]:
        feature_names.extend(
            [
                f"{axis}_mean",
                f"{axis}_std",
                f"{axis}_min",
                f"{axis}_max",
                f"{axis}_median",
                f"{axis}_q25",
                f"{axis}_q75",
                f"{axis}_kurtosis",
                f"{axis}_skew",
                f"{axis}_range",
                f"{axis}_zero_cross",
                f"{axis}_peaks",
                f"{axis}_energy",
                f"{axis}_fft_mean",
                f"{axis}_fft_std",
                f"{axis}_fft_max",
            ]
        )
    feature_names.extend(["mag_mean", "mag_std", "mag_max", "mag_min", "duration"])

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.barh(range(20), importances[indices])
    plt.yticks(range(20), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Most Important Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rf_feature_importance.png"), dpi=300)
    print(f"Feature importance saved to: {output_dir}/rf_feature_importance.png")
    plt.close()

    # Save model
    model_path = os.path.join(output_dir, "random_forest_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": rf,
                "label_encoder": label_encoder_filtered,
                "feature_extractor": loader.feature_extractor,
            },
            f,
        )
    print(f"\nModel saved to: {model_path}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
