"""
Simple Classifier - Uses Dynamic Time Warping for template matching
Matches unknown strokes to known letter templates
"""

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class SimpleLetterClassifier:
    """
    Simple template-based classifier using DTW distance
    """

    def __init__(self):
        self.templates = {}  # Dictionary: letter -> list of template signals

    def add_template(self, letter, signal):
        """
        Add a template for a letter

        Parameters:
        -----------
        letter : str
            The letter this template represents
        signal : np.array
            2D array of shape (n_samples, n_features) - typically acceleration data
        """
        if letter not in self.templates:
            self.templates[letter] = []
        self.templates[letter].append(signal)

    def train_from_strokes(self, strokes, labels):
        """
        Train classifier from labeled stroke data

        Parameters:
        -----------
        strokes : list of pd.DataFrame
            List of stroke data
        labels : list of str
            Corresponding labels for each stroke
        """
        for stroke_data, label in zip(strokes, labels):
            # Use x, y acceleration as signal
            signal = np.column_stack(
                [stroke_data["acc_x[mg]"].values, stroke_data["acc_y[mg]"].values]
            )
            self.add_template(label, signal)

    def classify(self, signal):
        """
        Classify an unknown signal

        Parameters:
        -----------
        signal : np.array
            2D array of shape (n_samples, n_features)

        Returns:
        --------
        str
            Predicted letter
        float
            Confidence score (lower is better - it's a distance)
        """
        if not self.templates:
            return "?", float("inf")

        best_letter = None
        best_distance = float("inf")

        # Compare to all templates
        for letter, templates in self.templates.items():
            for template in templates:
                try:
                    distance, _ = fastdtw(signal, template, dist=euclidean)

                    if distance < best_distance:
                        best_distance = distance
                        best_letter = letter
                except:
                    # If DTW fails, skip this template
                    continue

        return best_letter if best_letter else "?", best_distance

    def classify_stroke(self, stroke_data):
        """
        Classify a stroke DataFrame directly

        Parameters:
        -----------
        stroke_data : pd.DataFrame
            Stroke data with acceleration columns

        Returns:
        --------
        str
            Predicted letter
        float
            Confidence score
        """
        signal = np.column_stack(
            [stroke_data["acc_x[mg]"].values, stroke_data["acc_y[mg]"].values]
        )
        return self.classify(signal)


class SimpleThresholdClassifier:
    """
    Even simpler classifier based on feature thresholds
    Fallback if DTW doesn't work well
    """

    def __init__(self):
        self.letter_features = {}  # letter -> mean feature vector

    def train(self, feature_vectors, labels):
        """
        Train by averaging features for each letter

        Parameters:
        -----------
        feature_vectors : list of np.array
            Feature vectors
        labels : list of str
            Corresponding labels
        """
        for fv, label in zip(feature_vectors, labels):
            if label not in self.letter_features:
                self.letter_features[label] = []
            self.letter_features[label].append(fv)

        # Average features for each letter
        for letter in self.letter_features:
            self.letter_features[letter] = np.mean(
                np.vstack(self.letter_features[letter]), axis=0
            )

    def classify(self, feature_vector):
        """
        Classify by finding nearest mean feature vector

        Parameters:
        -----------
        feature_vector : np.array
            Feature vector to classify

        Returns:
        --------
        str
            Predicted letter
        float
            Distance to nearest template
        """
        if not self.letter_features:
            return "?", float("inf")

        best_letter = None
        best_distance = float("inf")

        for letter, template in self.letter_features.items():
            distance = np.linalg.norm(feature_vector - template)
            if distance < best_distance:
                best_distance = distance
                best_letter = letter

        return best_letter, best_distance
