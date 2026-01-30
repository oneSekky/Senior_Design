"""
Complete Letter Recognition Pipeline
Processes accelerometer data and outputs recognized letters
"""

import numpy as np
import pandas as pd
from classifier import SimpleLetterClassifier, SimpleThresholdClassifier
from data_reader import read_accelerometer_csv
from feature_extraction import create_feature_vector, extract_stroke_features
from stroke_segmentation import segment_strokes, visualize_segmentation


class LetterRecognitionPipeline:
    """
    Complete pipeline from CSV to recognized text
    """

    def __init__(self, use_dtw=True):
        """
        Initialize pipeline

        Parameters:
        -----------
        use_dtw : bool
            If True, use DTW classifier, else use simpler threshold classifier
        """
        self.use_dtw = use_dtw
        if use_dtw:
            self.classifier = SimpleLetterClassifier()
        else:
            self.classifier = SimpleThresholdClassifier()
        self.trained = False

    def train(self, csv_file, labels, verbose=True):
        """
        Train the classifier from a labeled CSV file

        Parameters:
        -----------
        csv_file : str
            Path to training CSV file
        labels : list of str
            Labels for each stroke in order (e.g., ['A', 'A', 'A'])
        verbose : bool
            Print progress info
        """
        # Read data
        df = read_accelerometer_csv(csv_file)

        if verbose:
            print(f"Training from {csv_file}")
            print(f"Loaded {len(df)} samples")

        # Segment into strokes
        strokes = segment_strokes(df, threshold=30, min_samples=15)

        if verbose:
            visualize_segmentation(df, strokes)

        # Check if we have enough strokes
        if len(strokes) < len(labels):
            print(
                f"Warning: Found {len(strokes)} strokes but expected {len(labels)} labels"
            )
            labels = labels[: len(strokes)]
        elif len(strokes) > len(labels):
            print(
                f"Warning: Found {len(strokes)} strokes but only {len(labels)} labels"
            )
            strokes = strokes[: len(labels)]

        # Train classifier
        if self.use_dtw:
            self.classifier.train_from_strokes([s["data"] for s in strokes], labels)
        else:
            # Extract features and train
            feature_vectors = []
            for stroke in strokes:
                features = extract_stroke_features(stroke["data"])
                fv = create_feature_vector(features)
                feature_vectors.append(fv)
            self.classifier.train(feature_vectors, labels)

        self.trained = True

        if verbose:
            print(f"Training complete! Learned {len(set(labels))} unique letters")

    def recognize(self, csv_file, verbose=True):
        """
        Recognize letters from a CSV file

        Parameters:
        -----------
        csv_file : str
            Path to CSV file to recognize
        verbose : bool
            Print detailed info

        Returns:
        --------
        str
            Recognized text
        list
            List of (letter, confidence) tuples
        """
        if not self.trained:
            raise ValueError("Classifier not trained! Call train() first")

        # Read data
        df = read_accelerometer_csv(csv_file)

        if verbose:
            print(f"\nRecognizing from {csv_file}")
            print(f"Loaded {len(df)} samples")

        # Segment into strokes
        strokes = segment_strokes(df, threshold=30, min_samples=15)

        if verbose:
            visualize_segmentation(df, strokes)

        # Classify each stroke
        results = []
        recognized_text = ""

        for i, stroke in enumerate(strokes):
            if self.use_dtw:
                letter, confidence = self.classifier.classify_stroke(stroke["data"])
            else:
                features = extract_stroke_features(stroke["data"])
                fv = create_feature_vector(features)
                letter, confidence = self.classifier.classify(fv)

            results.append((letter, confidence))
            recognized_text += letter

            if verbose:
                print(f"  Stroke {i + 1}: '{letter}' (confidence: {confidence:.2f})")

        return recognized_text, results

    def quick_recognize(self, csv_file):
        """
        Quick recognition with minimal output

        Parameters:
        -----------
        csv_file : str
            Path to CSV file

        Returns:
        --------
        str
            Recognized text
        """
        text, _ = self.recognize(csv_file, verbose=False)
        return text


def demo_pipeline():
    """
    Demonstration of the pipeline with test data
    """
    print("=" * 60)
    print("Letter Recognition Pipeline Demo")
    print("=" * 60)

    # Create pipeline
    pipeline = LetterRecognitionPipeline(use_dtw=True)

    # Train on Test3 (multiple A's)
    print("\n### TRAINING PHASE ###")
    # Assuming Test3 has multiple A's - adjust number based on actual data
    training_labels = ["A"] * 10  # Adjust this based on how many A's are in Test3
    pipeline.train("../Test_Data/Test3_same_strokes.csv", training_labels)

    print("\n### RECOGNITION PHASE ###")
    # Try to recognize from Test2
    text, results = pipeline.recognize("../Test_Data/Test2.csv")

    print("\n" + "=" * 60)
    print(f"FINAL RESULT: '{text}'")
    print("=" * 60)
    print("\nNote: Recognition quality depends on:")
    print("  1. Proper segmentation (threshold tuning)")
    print("  2. Similar writing style between training and test")
    print("  3. Having enough training samples")
    print("\nTo improve results:")
    print("  - Collect training data for all letters (A-Z)")
    print("  - Multiple samples per letter")
    print("  - Tune segmentation threshold parameter")


if __name__ == "__main__":
    demo_pipeline()
