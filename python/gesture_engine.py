import cv2
import mediapipe as mp
import math
import numpy as np
import json
import os
import joblib

class MediaPipeEngine:
    def __init__(self, config_path="config.json"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config['mediapipe']['max_num_hands'],
            min_detection_confidence=self.config['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=self.config['mediapipe']['min_tracking_confidence']
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Fingertip landmark IDs
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
        self.prev_y = 0 # For scroll tracking
        
        # Model loading
        self.model = None
        self.model_path = os.path.join(os.path.dirname(__file__), 'gesture_model.pkl')
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("Gesture model loaded successfully.")
            except Exception as e:
                print(f"Error loading gesture model: {e}")

    def _load_config(self, path):
        abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
        if os.path.exists(abs_path):
            try:
                with open(abs_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading config: {e}")
        
        # Fallback defaults
        return {
            "mediapipe": {"min_detection_confidence": 0.7, "min_tracking_confidence": 0.7, "max_num_hands": 1},
            "thresholds": {"click_distance": 30, "right_click_distance": 40, "scroll_distance": 15}
        }

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two (x,y) tuples"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_fingers_up(self, lm_list):
        """Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Pinky]"""
        fingers = []
        if len(lm_list) != 21:
            return [False]*5

        # Thumb: compare tip (4) x with IP (3) x (Assuming Right Hand for now)
        if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(True)
        else:
            fingers.append(False)

        # 4 Fingers: compare tip y with PIP y (landmarks 6, 10, 14, 18)
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(True)
            else:
                fingers.append(False)
                
        return fingers

    def _create_better_mask(self, hand_landmarks, h, w):
        """Creates a smooth skeleton-based mask for the hand"""
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Gather all landmark points
        points = []
        for lm in hand_landmarks.landmark:
            points.append((int(lm.x * w), int(lm.y * h)))
            
        # 2. Draw thick lines connecting landmarks (skeleton)
        # Use MediaPipe's hand connections to define the skeleton
        for connection in self.mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            p1 = points[start_idx]
            p2 = points[end_idx]
            cv2.line(mask, p1, p2, 255, 15) # Thick lines
            
        # 3. Draw circles at joints to fill gaps
        for p in points:
            cv2.circle(mask, p, 10, 255, -1)
            
        # 4. Create a convex hull of the palm area (landmarks 0, 1, 5, 9, 13, 17) to fill the palm
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_points = np.array([points[i] for i in palm_indices], dtype=np.int32)
        cv2.fillConvexPoly(mask, palm_points, 255)
        
        # 5. Apply dilation and blur for a smoother, more "organic" look
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Threshold to get sharp edges back but smoother
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask

    def process(self, frame):
        """
        Process the frame and return action, cursor_pos, and debug imagery.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        action = "none"
        cursor_pos = None
        h_frame, w_frame, _ = frame.shape
        mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Create improved mask
                mask = self._create_better_mask(hand_landmarks, h_frame, w_frame)
                
                # 2. Draw Landmarks for debugging
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 3. Extract landmark coordinates
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w_frame), int(lm.y * h_frame)
                    lm_list.append([id, cx, cy])
                
                # Use Index Finger tip for cursor positioning
                cursor_pos = (lm_list[8][1], lm_list[8][2]) 

                # 4. Detect Action
                if self.model:
                    # ML-based action detection (to be implemented)
                    features = []
                    # Normalize landmarks relative to wrist (index 0)
                    wrist_x, wrist_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
                    for lm in hand_landmarks.landmark:
                        features.extend([lm.x - wrist_x, lm.y - wrist_y])
                    
                    prediction = self.model.predict([features])[0]
                    # We only take the action if it's not 'background' or 'none'
                    if prediction != "none":
                        action = prediction
                else:
                    # FALLBACK: Heuristic Logic (Old flowchart style but using config)
                    fingers = self.get_fingers_up(lm_list)
                    dist_thumb_index = self.calculate_distance((lm_list[4][1], lm_list[4][2]), (lm_list[8][1], lm_list[8][2]))
                    dist_index_middle = self.calculate_distance((lm_list[8][1], lm_list[8][2]), (lm_list[12][1], lm_list[12][2]))

                    thresh = self.config['thresholds']
                    
                    if all(fingers): # All 5 up
                        action = "none"
                    elif fingers[0] and fingers[1] and dist_thumb_index < thresh['click_distance']:
                        action = "left_click"
                        cv2.circle(frame, (lm_list[8][1], lm_list[8][2]), 15, (0, 255, 0), cv2.FILLED)
                    elif fingers[1] and fingers[2] and dist_index_middle < thresh['right_click_distance']:
                        action = "right_click"
                        cv2.circle(frame, (lm_list[8][1], lm_list[8][2]), 15, (0, 0, 255), cv2.FILLED)
                    elif (fingers[1] and not fingers[2]) or (fingers[1] and fingers[2]):
                        action = "move"
                        if fingers[1] and fingers[2]:
                            current_y = lm_list[8][2]
                            if self.prev_y != 0:
                                if self.prev_y - current_y > thresh['scroll_distance']: 
                                    action = "scroll_up"
                                elif current_y - self.prev_y > thresh['scroll_distance']:
                                    action = "scroll_down"
                            self.prev_y = current_y
                        else:
                            self.prev_y = 0

        return action, cursor_pos, frame, mask
