"""
Test the model on all 'a' files with adjustable threshold.
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


def predict_all(model, scaler, csv_files, threshold=0.2):
    """Predict on all CSV files."""
    predictions = []
    filenames = []

    print(f"Generating predictions for {len(csv_files)} files...")
    print(f"Using threshold: {threshold}\n")

    for i, csv_file in enumerate(csv_files, 1):
        try:
            X = load_and_preprocess_csv(csv_file, scaler)
            pred = model.predict(X, verbose=0)[0]

            # Apply threshold
            pred_binary = (pred > threshold).astype(float)

            predictions.append(pred_binary)
            filenames.append(csv_file.stem)

            if i % 10 == 0:
                print(f"  Processed {i}/{len(csv_files)}...")

        except Exception as e:
            print(f"  ✗ Error with {csv_file.name}: {e}")
            continue

    print(f"✓ Generated {len(predictions)} predictions\n")

    return predictions, filenames


def show_all_predictions_grid(
    predictions, filenames, threshold, save_name="all_predictions_grid.png"
):
    """Display all predictions in a grid."""
    n_predictions = len(predictions)

    if n_predictions == 0:
        print("No predictions to show!")
        return

    # Calculate grid size
    cols = 5
    rows = (n_predictions + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle(
        f"All Predictions ({n_predictions} letters) - Threshold: {threshold}",
        fontsize=16,
        fontweight="bold",
    )

    # Handle single row/col cases
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (pred, filename) in enumerate(zip(predictions, filenames)):
        row = idx // cols
        col = idx % cols

        axes[row, col].imshow(pred, cmap="gray")
        axes[row, col].set_title(filename, fontsize=8)
        axes[row, col].axis("off")

    # Hide empty subplots
    for idx in range(len(predictions), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_name, dpi=200, bbox_inches="tight")
    print(f"✓ Saved grid to {save_name}")
    plt.show()


def main():
    print("=" * 60)
    print("Test Model on All Alphabet 'a' Files")
    print("=" * 60)

    # Check for model
    model_path = "best_handwriting_model_improved.keras"
    scaler_path = "scaler_improved.pkl"

    if not Path(model_path).exists():
        print(f"Improved model not found, using original...")
        model_path = "best_handwriting_model.keras"
        scaler_path = "scaler.pkl"

        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return

    if not Path(scaler_path).exists():
        print(f"❌ Scaler not found: {scaler_path}")
        return

    # Load model
    print(f"\nLoading model: {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
    except:
        # Try with custom loss
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

    # Find alphabet folder
    base_dir = Path(__file__).parent
    alphabet_dir = base_dir / "test_data" / "alphabet"

    if not alphabet_dir.exists():
        print(f"❌ Alphabet folder not found: {alphabet_dir}")
        return

    # Get all 'a' CSV files
    csv_files = sorted(alphabet_dir.glob("a*.csv"))

    if not csv_files:
        print(f"❌ No CSV files starting with 'a' found")
        return

    print(f"Found {len(csv_files)} CSV files for letter 'a'")

    # Ask for threshold
    print("\nChoose threshold:")
    print("  1. 0.1 (more permissive, thicker lines)")
    print("  2. 0.2 (balanced) [recommended]")
    print("  3. 0.3 (stricter, thinner lines)")
    print("  4. 0.4 (very strict)")

    choice = input("\nEnter choice (1-4) or press Enter for default [2]: ").strip()

    threshold_map = {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4}
    threshold = threshold_map.get(choice, 0.2)

    print()

    # Generate predictions
    predictions, filenames = predict_all(model, scaler, csv_files, threshold=threshold)

    # Show grid
    show_all_predictions_grid(predictions, filenames, threshold)

    print("\n" + "=" * 60)
    print("✓ Complete!")
    print(f"Displayed {len(predictions)} predictions with threshold {threshold}")
    print("=" * 60)

    print(f"Displayed {len(predictions)} predictions with threshold {threshold}")
    print("=" * 60)

if __name__ == "__main__":
    main()
