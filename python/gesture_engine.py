import cv2
import mediapipe as mp
import math
import numpy as np
import json
import os
import joblib
import time

try:
    from feature_extractor import FeatureExtractor
    from state_machine import GlobalStateMachine, GestureState
    from conflict_resolver import ConflictResolver
except ImportError:
    from .feature_extractor import FeatureExtractor
    from .state_machine import GlobalStateMachine, GestureState
    from .conflict_resolver import ConflictResolver

class MediaPipeEngine:
    def __init__(self, config_path="config.json"):
        # 1. Resolve absolute paths for reliable loading
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.dirname(self.script_dir)
        
        # Load configuration
        abs_config_path = os.path.join(self.root_dir, config_path)
        self.config = self._load_config(abs_config_path)
        
        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config['mediapipe']['max_num_hands'],
            min_detection_confidence=self.config['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=self.config['mediapipe']['min_tracking_confidence']
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # V2 Modules
        self.extractor = FeatureExtractor()
        self.resolver = ConflictResolver()
        
        gesture_names = ["move", "left_click", "right_click", "drag", "scroll", "double_click"]
        self.state_machine = GlobalStateMachine(gesture_names)
        
        # Timing for Double Click / Drag
        self.pinch_start_time = 0
        
        # Model loading
        self.model = None
        self.model_path = os.path.join(self.script_dir, 'gesture_model.pkl')
        
        print(f"[Engine V2] Searching for AI model at: {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(">>> [Engine V2] AI GESTURE MODEL LOADED.")
            except Exception as e:
                print(f"!!! [Engine V2] Model loading failed: {e}")
        else:
            print(">>> [Engine V2] No model found. Using Heuristic logic + State Machine.")

    def _load_config(self, abs_path):
        if os.path.exists(abs_path):
            try:
                with open(abs_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"!!! [Engine] Config load error: {e}")
        
        return {
            "mediapipe": {"min_detection_confidence": 0.7, "min_tracking_confidence": 0.7, "max_num_hands": 1},
            "thresholds": {"pinch": 0.05, "scroll": 0.05}
        }

    def _create_v2_mask(self, hand_landmarks, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
        
        # Draw connections with thick lines (solid hand shape)
        for connection in self.mp_hands.HAND_CONNECTIONS:
            p1, p2 = points[connection[0]], points[connection[1]]
            cv2.line(mask, p1, p2, 255, 12)
        
        # Fill in palm
        palm_indices = [0, 1, 5, 9, 13, 17]
        palm_points = np.array([points[i] for i in palm_indices], dtype=np.int32)
        cv2.fillConvexPoly(mask, palm_points, 255)
        
        # Soften and threshold
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        return mask

    def process(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        detections = {g: False for g in ["move", "left_click", "right_click", "drag", "scroll", "double_click"]}
        cursor_pos = None
        h_frame, w_frame, _ = frame.shape
        mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
        hand_landmarks = None
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mask = self._create_v2_mask(hand_landmarks, h_frame, w_frame)
            
            # 1. Feature Extraction
            features = self.extractor.extract(hand_landmarks)
            cursor_pos = (int(hand_landmarks.landmark[8].x * w_frame), 
                          int(hand_landmarks.landmark[8].y * h_frame))

            # 2. Raw Gesture Detection (Heuristic or AI)
            if self.model:
                try:
                    flat_f = self.extractor.get_flattened_vector(hand_landmarks)
                    ai_gesture = self.model.predict([flat_f])[0]
                    if ai_gesture in detections: detections[ai_gesture] = True
                except: pass
            else:
                # V2 MANDATORY HEURISTIC LOGIC
                ext = features["extensions"]
                pinch_dist = features["pinch_distances"] # [Thumb-Index, Thumb-Middle]
                p_thresh = self.config['thresholds']['pinch']
                
                # Move: Index extended
                if ext[1] and not ext[2] and not ext[3] and not ext[4]:
                    detections["move"] = True
                
                # Scroll: Index and Middle extended
                if ext[1] and ext[2] and not ext[3] and not ext[4]:
                    detections["scroll"] = True
                
                # Clicks / Drag (Pinch based)
                is_pinch_ti = pinch_dist[0] < p_thresh
                is_pinch_tm = pinch_dist[1] < p_thresh
                
                if is_pinch_ti:
                    now = time.time()
                    if self.pinch_start_time == 0:
                        self.pinch_start_time = now
                    
                    elapsed = now - self.pinch_start_time
                    if elapsed > 0.4: # Hold > 400ms = Drag
                        detections["drag"] = True
                    else:
                        detections["left_click"] = True
                else:
                    self.pinch_start_time = 0
                
                if is_pinch_tm:
                    detections["right_click"] = True

        # 3. State Machine Update
        states = self.state_machine.update_all(detections)
        
        # 4. Conflict Resolution
        active_gestures = [name for name, state in states.items() if state == GestureState.ACTIVE]
        final_action = self.resolver.resolve(active_gestures)

        return {
            "action": final_action,
            "cursor_pos": cursor_pos,
            "states": states,
            "mask": mask,
            "landmarks": hand_landmarks,
            "results": results,
            "frame": frame # Main app expects frame back
        }
