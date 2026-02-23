import numpy as np
import math

class FeatureExtractor:
    """Extracts advanced features from MediaPipe landmarks for V2."""
    
    def __init__(self):
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_mcp = [1, 5, 9, 13, 17]
        
    def get_angle(self, p1, p2, p3):
        """Calculates the angle between three points (degrees)."""
        a = np.array(p1)
        b = np.array(p2) # Vertex
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def extract(self, landmarks):
        """
        Extracts a feature vector:
        - Extension states (5)
        - Joint angles (5 major joints)
        - Relative distances (normalized)
        - Palm orientation
        """
        features = {}
        h, w = 1.0, 1.0 # Normalized landmarks already
        
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # 1. Finger Extension States (Binary-ish)
        # Check if tip is further from wrist than MCP
        wrist = pts[0]
        extensions = []
        for tip, mcp in zip(self.finger_tips, self.finger_mcp):
            dist_tip = np.linalg.norm(pts[tip] - wrist)
            dist_mcp = np.linalg.norm(pts[mcp] - wrist)
            extensions.append(1.0 if dist_tip > dist_mcp else 0.0)
        features["extensions"] = extensions

        # 2. Joint Angles (Proximal joints)
        angles = []
        # Index angle (0-5-8)
        angles.append(self.get_angle(pts[0], pts[5], pts[8]))
        # Middle angle (0-9-12)
        angles.append(self.get_angle(pts[0], pts[9], pts[12]))
        # Ring angle (0-13-16)
        angles.append(self.get_angle(pts[0], pts[13], pts[16]))
        # Pinky angle (0-17-20)
        angles.append(self.get_angle(pts[0], pts[17], pts[20]))
        # Thumb angle (1-2-4)
        angles.append(self.get_angle(pts[1], pts[2], pts[4]))
        features["angles"] = angles

        # 3. Palm Orientation (Simplified)
        # Vector from wrist to middle finger MCP
        palm_v = pts[9] - pts[0]
        features["palm_vec"] = palm_v[:2].tolist() # X, Y direction

        # 4. Critical Distances (Pinch detection)
        # Thumb to Index
        d_ti = np.linalg.norm(pts[4] - pts[8])
        # Thumb to Middle
        d_tm = np.linalg.norm(pts[4] - pts[12])
        features["pinch_distances"] = [d_ti, d_tm]

        return features

    def get_flattened_vector(self, landmarks):
        """Returns features as a flat list for ML models."""
        f = self.extract(landmarks)
        return f["extensions"] + f["angles"] + f["palm_vec"] + f["pinch_distances"]
