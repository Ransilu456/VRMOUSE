import cv2
import mediapipe as mp
import math
import numpy as np
import json
import os
import joblib

class MediaPipeEngine:
    def __init__(self, config_path="config.json"):
        # 1. Resolve absolute paths for reliable loading
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.script_dir)
        
        # Load configuration
        abs_config_path = os.path.join(self.root_dir, config_path)
        self.config = self._load_config(abs_config_path)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config['mediapipe']['max_num_hands'],
            min_detection_confidence=self.config['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=self.config['mediapipe']['min_tracking_confidence']
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.tip_ids = [4, 8, 12, 16, 20] 
        self.prev_y = 0 
        
        # 2. Robust Model loading
        self.model = None
        self.model_path = os.path.join(self.script_dir, 'gesture_model.pkl')
        
        print(f"[Engine] Searching for AI model at: {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(">>> [Engine] AI GESTURE MODEL LOADED. AI CONTROL ENABLED.")
            except Exception as e:
                print(f"!!! [Engine] Model loading failed: {e}")
        else:
            print(">>> [Engine] No model found. Falling back to Heuristic control.")

    def _load_config(self, abs_path):
        if os.path.exists(abs_path):
            try:
                with open(abs_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"!!! [Engine] Config load error: {e}")
        
        # Safe defaults
        print(">>> [Engine] Using default configuration.")
        return {
            "mediapipe": {"min_detection_confidence": 0.7, "min_tracking_confidence": 0.7, "max_num_hands": 1},
            "thresholds": {"click_distance": 30, "right_click_distance": 40, "scroll_distance": 15}
        }

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_fingers_up(self, lm_list):
        fingers = []
        if len(lm_list) != 21: return [False]*5
        # Thumb
        if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]: fingers.append(True)
        else: fingers.append(False)
        # 4 Fingers
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]: fingers.append(True)
            else: fingers.append(False)
        return fingers

    def _create_better_mask(self, hand_landmarks, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
        
        for connection in self.mp_hands.HAND_CONNECTIONS:
            p1, p2 = points[connection[0]], points[connection[1]]
            cv2.line(mask, p1, p2, 255, 15)
        for p in points: cv2.circle(mask, p, 10, 255, -1)
        
        palm_points = np.array([points[i] for i in [0, 1, 5, 9, 13, 17]], dtype=np.int32)
        cv2.fillConvexPoly(mask, palm_points, 255)
        
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def process(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        action = "none"
        cursor_pos = None
        h_frame, w_frame, _ = frame.shape
        mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mask = self._create_better_mask(hand_landmarks, h_frame, w_frame)
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                lm_list = [[id, int(lm.x * w_frame), int(lm.y * h_frame)] 
                           for id, lm in enumerate(hand_landmarks.landmark)]
                cursor_pos = (lm_list[8][1], lm_list[8][2]) 

                # AI DECISION LOGIC
                if self.model:
                    try:
                        features = []
                        w_x, w_y = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y
                        for lm in hand_landmarks.landmark:
                            features.extend([lm.x - w_x, lm.y - w_y])
                        
                        # AI control - Exclusive usage if model is active
                        action = self.model.predict([features])[0]
                    except Exception as e:
                        print(f"!!! [Engine] ML Prediction Error: {e}")
                else:
                    # HEURISTIC CONTROL (Only if NO model is loaded)
                    fingers = self.get_fingers_up(lm_list)
                    d_ti = self.calculate_distance((lm_list[4][1], lm_list[4][2]), (lm_list[8][1], lm_list[8][2]))
                    d_im = self.calculate_distance((lm_list[8][1], lm_list[8][2]), (lm_list[12][1], lm_list[12][2]))
                    thresh = self.config['thresholds']
                    
                    if all(fingers): action = "none"
                    elif fingers[0] and fingers[1] and d_ti < thresh['click_distance']: action = "left_click"
                    elif fingers[1] and fingers[2] and d_im < thresh['right_click_distance']: action = "right_click"
                    elif fingers[1]:
                        action = "move"
                        if fingers[2]: # Scroll mode
                            current_y = lm_list[8][2]
                            if self.prev_y != 0:
                                if self.prev_y - current_y > thresh['scroll_distance']: action = "scroll_up"
                                elif current_y - self.prev_y > thresh['scroll_distance']: action = "scroll_down"
                            self.prev_y = current_y
                        else: self.prev_y = 0

        return action, cursor_pos, frame, mask
