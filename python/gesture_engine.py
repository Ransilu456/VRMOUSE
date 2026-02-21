import cv2
import numpy as np
import math

class GestureEngine:
    def __init__(self):
        # MOG2 for background isolation
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.kernel = np.ones((5, 5), np.uint8)
        
        # Skin color ranges (calibrated for better robustness)
        self.lower_hsv = np.array([0, 20, 80], dtype=np.uint8)
        self.upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        
        # Debounce state
        self.gesture_history = []
        self.history_size = 5

    def get_skin_mask(self, frame):
        # 1. Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # 2. Skin thresholding (HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 15, 60]), np.array([30, 255, 255]))
        
        # 3. YCrCb skin detection
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        
        # Union of skin masks, AND with foreground
        combined_skin = cv2.bitwise_or(mask_hsv, mask_ycrcb)
        mask = cv2.bitwise_and(combined_skin, fg_mask)
        
        # Clean up noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        return mask

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 given points p1, p2, p3"""
        a = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2)
        b = math.sqrt((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2)
        c = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        try:
            angle = math.acos((a**2 + c**2 - b**2) / (2*a*c)) * 180 / math.pi
        except:
            angle = 180
        return angle

    def process(self, frame):
        mask = self.get_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, "none", mask

        # Largest contour is the hand
        hand_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand_contour) < 1500:
            return None, None, "none", mask

        # Smooth contour
        epsilon = 0.01 * cv2.arcLength(hand_contour, True)
        hand_contour = cv2.approxPolyDP(hand_contour, epsilon, True)

        # 1. Find Palm Center
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        palm_center = max_loc
        palm_radius = int(max_val)

        # 2. Extract Features (Distance of extreme points from palm center)
        # We'll take 10 points around the hull to create a signature
        hull_pts = cv2.convexHull(hand_contour)
        
        # Calculate distances from palm center to all hull points
        distances = []
        for pt in hull_pts:
            d = np.sqrt((pt[0][0] - palm_center[0])**2 + (pt[0][1] - palm_center[1])**2)
            distances.append(d)
        
        # Normalize and sample fixed number of features for AI
        distances.sort(reverse=True)
        # Top 10 furthest points (fingertips and major knuckle points)
        raw_features = distances[:10]
        if len(raw_features) < 10:
            raw_features.extend([0] * (10 - len(raw_features)))
        
        # Normalize by the max feature distance (longest fingertip)
        # This is much more stable than the palm radius which fluctuates with mask holes
        max_dist = raw_features[0] if raw_features[0] > 0 else 1
        normalized_features = np.array(raw_features) / max_dist

        # 3. Visualization
        cv2.circle(frame, palm_center, palm_radius, (255, 0, 0), 2)
        cv2.circle(frame, palm_center, 5, (0, 0, 255), -1)
        
        # Draw some of the hull points
        for i, pt in enumerate(hull_pts[:10]):
            cv2.circle(frame, tuple(pt[0]), 5, (0, 255, 0), -1)

        # Cursor Position (Stable fingertip tracking)
        # Instead of just the raw min pixels, average the top 3 extreme points 
        # that are furthest from palm to stay on the main fingertip
        hull_list = [tuple(pt[0]) for pt in hull_pts]
        # Sort by distance from palm descending
        hull_list.sort(key=lambda p: np.sqrt((p[0]-palm_center[0])**2 + (p[1]-palm_center[1])**2), reverse=True)
        
        # Filter points that are actually in the upper half of the hand relative to palm
        tops = [p for p in hull_list[:3] if p[1] < palm_center[1]]
        if not tops: tops = [hull_list[0]] # fallback
        
        # Average the top points and pick the highest one for cursor
        avg_x = sum(p[0] for p in tops) / len(tops)
        avg_y = sum(p[1] for p in tops) / len(tops)
        
        cursor_pos = (int(avg_x), int(avg_y))

        return cursor_pos, normalized_features, palm_center, mask
