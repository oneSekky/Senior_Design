"""
Diagnose what the model is actually outputting.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
from tensorflow import keras


def load_and_preprocess_csv(csv_path, scaler, target_length=200):
    """Load and preprocess a CSV file."""
    df = pd.read_csv(csv_path, skiprows=1)

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
        filtered_data[:, 0] ** 2 + filtered_data[:, 1] ** 2 + filtered_data[:, 2] ** 2
    )

    features = np.column_stack([filtered_data, acc_magnitude])

    # Pad or truncate
    if len(features) > target_length:
        features = features[-target_length:]
    else:
        padding = np.zeros((target_length - len(features), features.shape[1]))
        features = np.vstack([padding, features])

    # Normalize
    features_reshaped = features.reshape(-1, features.shape[1])
    features_normalized = scaler.transform(features_reshaped)
    features_normalized = features_normalized.reshape(
        1, target_length, features.shape[1]
    )

    return features_normalized


def main():
    print("=" * 60)
    print("Diagnosing Model Predictions")
    print("=" * 60)

    # Try improved model first
    model_path = "best_handwriting_model_improved.keras"
    scaler_path = "scaler_improved.pkl"

    if not Path(model_path).exists():
        print(f"Improved model not found, using original...")
        model_path = "best_handwriting_model.keras"
        scaler_path = "scaler.pkl"

    print(f"\nLoading: {model_path}")

    # Load with compile=False to avoid custom loss issues
    try:
        model = keras.models.load_model(model_path, compile=False)
    except:
        print("Error loading model with compile=False, trying with custom objects...")

        # Define custom loss if needed
        def combined_loss(y_true, y_pred):
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            bce_loss = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
            return 0.3 * mse_loss + 0.7 * bce_loss

        model = keras.models.load_model(
            model_path, custom_objects={"combined_loss": combined_loss}
        )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print("✓ Model loaded\n")

    # Find a test CSV
    base_dir = Path(__file__).parent

    # Try alphabet_write first (has ground truth)
    test_csv = None
    csv_dir = base_dir / "test_data" / "alphabet_write"
    if csv_dir.exists():
        csv_files = sorted(csv_dir.glob("a_w_*.csv"))
        if csv_files:
            test_csv = csv_files[0]

    # Fall back to alphabet folder
    if test_csv is None:
        csv_dir = base_dir / "test_data" / "alphabet"
        if csv_dir.exists():
            csv_files = sorted(csv_dir.glob("a*.csv"))
            if csv_files:
                test_csv = csv_files[0]

    if test_csv is None:
        print("❌ No CSV files found!")
        return

    print(f"Testing on: {test_csv.name}\n")

    # Preprocess and predict
    X = load_and_preprocess_csv(test_csv, scaler)
    prediction = model.predict(X, verbose=1)[0]

    # Analyze prediction
    print(f"\n{'=' * 60}")
    print("PREDICTION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Shape: {prediction.shape}")
    print(f"Min value: {prediction.min():.6f}")
    print(f"Max value: {prediction.max():.6f}")
    print(f"Mean value: {prediction.mean():.6f}")
    print(f"Std dev: {prediction.std():.6f}")
    print(f"Values > 0.5: {(prediction > 0.5).sum()} / {prediction.size}")
    print(f"Values > 0.1: {(prediction > 0.1).sum()} / {prediction.size}")
    print(f"Values > 0.01: {(prediction > 0.01).sum()} / {prediction.size}")
    print(f"{'=' * 60}\n")

    # Show different visualizations
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Raw prediction
    axes[0, 0].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(
        f"Raw Prediction\n(min={prediction.min():.4f}, max={prediction.max():.4f})"
    )
    axes[0, 0].axis("off")

    # Rescaled to use full range
    pred_rescaled = (prediction - prediction.min()) / (
        prediction.max() - prediction.min() + 1e-8
    )
    axes[0, 1].imshow(pred_rescaled, cmap="gray")
    axes[0, 1].set_title("Rescaled (stretched to 0-1)")
    axes[0, 1].axis("off")

    # Threshold at 0.5
    pred_thresh_05 = (prediction > 0.5).astype(float)
    axes[0, 2].imshow(pred_thresh_05, cmap="gray")
    axes[0, 2].set_title(f"Threshold > 0.5\n({(prediction > 0.5).sum()} pixels)")
    axes[0, 2].axis("off")

    # Threshold at 0.1
    pred_thresh_01 = (prediction > 0.1).astype(float)
    axes[1, 0].imshow(pred_thresh_01, cmap="gray")
    axes[1, 0].set_title(f"Threshold > 0.1\n({(prediction > 0.1).sum()} pixels)")
    axes[1, 0].axis("off")

    # Threshold at 0.01
    pred_thresh_001 = (prediction > 0.01).astype(float)
    axes[1, 1].imshow(pred_thresh_001, cmap="gray")
    axes[1, 1].set_title(f"Threshold > 0.01\n({(prediction > 0.01).sum()} pixels)")
    axes[1, 1].axis("off")

    # Histogram
    axes[1, 2].hist(prediction.flatten(), bins=50, color="blue", alpha=0.7)
    axes[1, 2].set_title("Histogram of Pixel Values")
    axes[1, 2].set_xlabel("Pixel Value")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("prediction_diagnosis.png", dpi=150, bbox_inches="tight")
    print("✓ Saved diagnosis to prediction_diagnosis.png")
    plt.show()

    # Recommendations
    print("\nRECOMMENDATIONS:")
    if prediction.max() < 0.1:
        print("⚠ Model outputs are very low (max < 0.1)")
        print("  → Model may not have trained properly")
        print(
            "  → Try retraining with original architecture (train_handwriting_simple.py)"
        )
    elif prediction.max() < 0.5:
        print("⚠ Model outputs are low (max < 0.5)")
        print("  → Try using lower threshold (e.g., 0.1 instead of 0.5)")
        print("  → Or use rescaled output")
    else:
        print("✓ Model output range looks reasonable")
        print("  → Threshold at 0.5 should work")


if __name__ == "__main__":
    main()
