"""
Letter Prediction Script - Use the trained CNN to classify new letter samples

Usage:
    python predict_letter.py path/to/letter.csv
    python predict_letter.py path/to/folder/  # Process all CSVs in folder
"""

import os
import sys

import numpy as np
from letter_cnn_classifier import LetterDataLoader
from tensorflow import keras

# Letter mapping (classes kept after filtering)
LETTERS = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "j",
    "l",
    "n",
    "o",
    "p",
    "q",
    "t",
    "v",
    "y",
    "z",
]


def load_model(model_path="output/final_letter_cnn_model.keras"):
    """Load the trained CNN model"""
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print(
            "Please train the model first by running: python letter_cnn_classifier.py"
        )
        return None

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!\n")
    return model


def preprocess_csv(filepath, target_length=200, sampling_rate=104):
    """Preprocess a single CSV file for prediction"""
    loader = LetterDataLoader(".", target_length, sampling_rate)

    # Read and preprocess
    raw_data, error = loader.read_csv_file(filepath)
    if raw_data is None:
        return None, error

    processed_data = loader.preprocess_signal(raw_data)
    if processed_data is None:
        return None, "Preprocessing failed"

    # Add batch dimension: (200, 3) -> (1, 200, 3)
    return np.expand_dims(processed_data, axis=0), None


def predict_letter(model, preprocessed_data, top_k=3):
    """
    Predict the letter from preprocessed data

    Returns:
        predictions: List of (letter, confidence) tuples for top-k predictions
    """
    # Get prediction probabilities
    probs = model.predict(preprocessed_data, verbose=0)[0]

    # Get top-k predictions
    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = [(LETTERS[idx], float(probs[idx])) for idx in top_indices]

    return predictions


def predict_single_file(model, filepath):
    """Predict letter for a single CSV file"""
    print(f"Processing: {os.path.basename(filepath)}")

    # Preprocess
    data, error = preprocess_csv(filepath)
    if data is None:
        print(f"  ERROR: {error}\n")
        return

    # Predict
    predictions = predict_letter(model, data, top_k=3)

    # Display results
    print(f"  Top 3 predictions:")
    for i, (letter, confidence) in enumerate(predictions, 1):
        print(f"    {i}. {letter.upper()}: {confidence * 100:.2f}%")
    print(
        f"  Most likely: {predictions[0][0].upper()} ({predictions[0][1] * 100:.2f}% confident)\n"
    )


def predict_folder(model, folder_path):
    """Predict letters for all CSV files in a folder"""
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    print(f"Found {len(csv_files)} CSV files\n")

    results = []
    for filename in sorted(csv_files):
        filepath = os.path.join(folder_path, filename)

        # Preprocess
        data, error = preprocess_csv(filepath)
        if data is None:
            print(f"{filename}: ERROR - {error}")
            continue

        # Predict
        predictions = predict_letter(model, data, top_k=1)
        predicted_letter = predictions[0][0]
        confidence = predictions[0][1]

        results.append((filename, predicted_letter, confidence))
        print(f"{filename}: {predicted_letter.upper()} ({confidence * 100:.2f}%)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Prediction Summary")
    print(f"{'=' * 60}")
    print(f"Total files processed: {len(results)}")
    print(f"Average confidence: {np.mean([c for _, _, c in results]) * 100:.2f}%")

    # Group by predicted letter
    from collections import Counter

    letter_counts = Counter([letter for _, letter, _ in results])
    print(f"\nPredicted letter distribution:")
    for letter, count in sorted(letter_counts.items()):
        print(f"  {letter.upper()}: {count} files")


def main():
    """Main prediction pipeline"""
    print("=" * 70)
    print("CNN Letter Prediction from Accelerometer Data")
    print("=" * 70)
    print()

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict_letter.py <path_to_csv_file>")
        print("  python predict_letter.py <path_to_folder>")
        print()
        print("Example:")
        print("  python predict_letter.py ../Test_Data/alphabet/a1_c+1.csv")
        print("  python predict_letter.py ../Test_Data/alphabet/")
        return

    path = sys.argv[1]

    if not os.path.exists(path):
        print(f"ERROR: Path not found: {path}")
        return

    # Load model
    model = load_model()
    if model is None:
        return

    # Process file or folder
    if os.path.isfile(path):
        predict_single_file(model, path)
    elif os.path.isdir(path):
        predict_folder(model, path)
    else:
        print(f"ERROR: Invalid path: {path}")


if __name__ == "__main__":
    main()
