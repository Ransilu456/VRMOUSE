import numpy as np
import json
import os

class GestureAI:
    def __init__(self, data_file='gestures.json'):
        self.data_file = data_file
        self.prototypes = {} # {gesture_name: average_feature_vector}
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    # Convert list back to numpy arrays
                    self.prototypes = {k: np.array(v) for k, v in data.items()}
                print(f"AI loaded {len(self.prototypes)} gestures.")
            except Exception as e:
                print(f"Error loading AI data: {e}. Starting fresh.")
                self.prototypes = {}
        else:
            self.prototypes = {}

    def save_data(self):
        try:
            # Convert numpy arrays to list for JSON
            serializable_data = {k: v.tolist() for k, v in self.prototypes.items()}
            with open(self.data_file, 'w') as f:
                json.dump(serializable_data, f)
            print("AI data saved.")
        except Exception as e:
            print(f"Error saving AI data: {e}")

    def train_gesture(self, name, samples):
        """Train a gesture using a list of feature vectors"""
        if not samples:
            return
        # Calculate mean vector as the prototype
        avg_vector = np.mean(samples, axis=0)
        self.prototypes[name] = avg_vector
        self.save_data()

    def predict(self, feature_vector):
        if not self.prototypes:
            return "none", 0.0

        best_gesture = "none"
        min_dist = float('inf')

        # Simple Euclidean Distance
        for name, proto in self.prototypes.items():
            dist = np.linalg.norm(feature_vector - proto)
            if dist < min_dist:
                min_dist = dist
                best_gesture = name

        # Confidence based on distance (heuristic)
        # With max-distance normalization, the max possible distance is around 3.16.
        # A threshold of 0.8 is usually good for rejecting "none"
        if min_dist > 0.8:
            return "none", 0.0

        confidence = max(0, 1 - (min_dist / 0.8))
        return best_gesture, confidence

def normalize_features(features, max_dist):
    """Normalize the raw feature distance vector by max distance"""
    if max_dist == 0: return features
    return np.array(features) / max_dist
