import cv2
import numpy as np

class MaskRenderer:
    """Handles rendering of the dedicated Black & White Hand Mask window."""
    
    def __init__(self, window_name="VR Mouse - Hand Mask"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
    def render(self, frame, mask, landmarks=None, gesture_name="none", state="idle"):
        """
        Creates a high-quality mask view.
        - Black background
        - White segmentation
        - Debug info if landmarks provided
        """
        h_frame, w_frame, _ = frame.shape
        
        # 1. Base Mask (B&W)
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 2. Add Contrast & Glow effect to the mask (Optional Aesthetic)
        # display = cv2.GaussianBlur(display, (5, 5), 0)
        
        # 3. Debug Overlays
        if landmarks:
            # Draw landmarks in a subtle color on the mask
            import mediapipe as mp
            mp_draw = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            mp_draw.draw_landmarks(
                display, 
                landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
            )
            
        # 4. Info Panel
        cv2.rectangle(display, (0, 0), (w_frame, 40), (20, 20, 20), -1)
        cv2.putText(display, f"STATE: {state.upper()}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        color = (0, 255, 0) if gesture_name != "none" else (150, 150, 150)
        cv2.putText(display, f"GESTURE: {gesture_name.upper()}", (w_frame - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.imshow(self.window_name, display)

    def close(self):
        cv2.destroyWindow(self.window_name)
