import cv2
import mediapipe as mp
import math
import numpy as np

class MediaPipeEngine:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Fingertip landmark IDs
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
        self.prev_y = 0 # For scroll tracking

    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two (x,y) tuples"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_fingers_up(self, lm_list):
        """Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Pinky]"""
        fingers = []
        if len(lm_list) != 21:
            return [False]*5

        # Thumb: compare tip (4) x with IP (3) x (Assuming Right Hand for now)
        # To make it robust for both, we can check relative to wrist or another point
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

    def process(self, frame):
        """
        Process the frame and return:
        - action (string: 'move', 'left_click', 'right_click', 'scroll_up', 'scroll_down', 'none')
        - cursor_pos (tuple: (x, y) coordinates of the index finger)
        - frame (the image with landmarks drawn)
        """
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        action = "none"
        cursor_pos = None
        
        # Create a black mask to simulate the old skin mask
        h_frame, w_frame, _ = frame.shape
        mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Draw Landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 2. Extract landmark coordinates
                h, w, c = frame.shape
                lm_list = []
                x_list, y_list = [], []
                
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
                    x_list.append(cx)
                    y_list.append(cy)
                
                # 3. Draw Bounding Box (Flowchart requirement)
                # xmin, xmax = min(x_list), max(x_list)
                # ymin, ymax = min(y_list), max(y_list)
                # cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
                
                # Draw hand hull to simulate a mask
                hull_points = np.array([[cx, cy] for cx, cy in zip(x_list, y_list)], dtype=np.int32)
                if len(hull_points) > 0:
                    hull = cv2.convexHull(hull_points)
                    cv2.fillConvexPoly(mask, hull, 255)
                
                # Use Index Finger tip for cursor positioning
                cursor_pos = (lm_list[8][1], lm_list[8][2]) 

                # 4. Detect which fingers are UP
                fingers = self.get_fingers_up(lm_list)
                # print(fingers) # [Thumb, Index, Middle, Ring, Pinky]
                
                # Calculate distances between specific fingertips
                # Thumb (4) and Index (8)
                dist_thumb_index = self.calculate_distance((lm_list[4][1], lm_list[4][2]), (lm_list[8][1], lm_list[8][2]))
                # Index (8) and Middle (12)
                dist_index_middle = self.calculate_distance((lm_list[8][1], lm_list[8][2]), (lm_list[12][1], lm_list[12][2]))

                # --- FLOWCHART LOGIC IMPLEMENTATION ---
                
                # RULE: If all Five Fingers are up -> No Action
                if fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                    action = "none"
                    
                # RULE: If both Thumb and Index Fingers are up AND length between them is below 30px
                elif fingers[0] and fingers[1] and dist_thumb_index < 30:
                    action = "left_click"
                    cv2.circle(frame, (lm_list[8][1], lm_list[8][2]), 15, (0, 255, 0), cv2.FILLED)
                    
                # RULE: If both Index and Middle Fingers are up AND length between them is below 40px
                elif fingers[1] and fingers[2] and dist_index_middle < 40:
                    action = "right_click"
                    cv2.circle(frame, (lm_list[8][1], lm_list[8][2]), 15, (0, 0, 255), cv2.FILLED)
                    
                # RULE: If index Finger is up OR if both Index and middle Fingers are up -> Move Mouse
                # Note: The "move" state is the default fallback if no smaller distance clicks trigger
                elif (fingers[1] and not fingers[2]) or (fingers[1] and fingers[2]):
                    action = "move"
                    # Add Scrolling sub-logic
                    # If both Index and middle are up and moved Towards up/down
                    if fingers[1] and fingers[2]:
                        current_y = lm_list[8][2]
                        if self.prev_y != 0:
                            # If y value decreases significantly, hand is moving UP screen
                            if self.prev_y - current_y > 15: 
                                action = "scroll_up"
                            # If y value increases significantly, hand is moving DOWN screen
                            elif current_y - self.prev_y > 15:
                                action = "scroll_down"
                        self.prev_y = current_y
                    else:
                        # Reset scroll reference when not in that 2-finger mode
                        self.prev_y = 0
                        

        return action, cursor_pos, frame, mask
