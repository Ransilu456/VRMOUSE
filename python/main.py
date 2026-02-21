import cv2
import numpy as np
import time
import os
import ctypes

# Load the C++ DLL via ctypes
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mouse_control.dll'))
try:
    mouse_dll = ctypes.CDLL(dll_path)
    # Define argument types for the DLL functions
    mouse_dll.move_mouse.argtypes = [ctypes.c_int, ctypes.c_int]
    mouse_dll.click_mouse.argtypes = []
    mouse_dll.right_click_mouse.argtypes = []
    mouse_dll.double_click_mouse.argtypes = []
    print(f"DLL loaded successfully from {dll_path}")
except Exception as e:
    print(f"Could not load DLL at {dll_path}: {e}")
    mouse_dll = None

from gesture_engine import GestureEngine
from smoothing import Smoother
from gesture_ai import GestureAI

def draw_info(frame, gesture, confidence, training_mode, wizard_step, sample_count):
    color = (255, 255, 255)
    if gesture == "click": color = (0, 255, 0)
    elif gesture == "right_click": color = (0, 0, 255)
    
    cv2.putText(frame, f"AI Gesture: {gesture.upper()} ({int(confidence*100)}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if training_mode:
        if wizard_step:
            cv2.putText(frame, f"WIZARD: {wizard_step}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Samples: {sample_count}/20 (Hold 'S' to Save)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "TRAINING MODE - Press 'T' to Start Mouse", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press '2'-'4' to train specific gestures:", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "2: POINTER (Move), 3: V-SHAPE (Click), 4: OPEN (Right Click)", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'R' to Reset BG, 'M' for Mask View", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    else:
        cv2.putText(frame, "MOUSE ACTIVE - Press 'T' for Training", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)
    engine = GestureEngine()
    # 1Euro Filter parameters: min_cutoff for jitter, beta for lag/speed
    smooth = Smoother(min_cutoff=0.5, beta=0.05)
    ai = GestureAI()

    # Adjust to your screen resolution
    screen_w, screen_h = 1920, 1080 
    
    # We'll use a smaller region of the webcam frame for movement to make it easier
    move_box_w, move_box_h = 400, 300
    move_box_x, move_box_y = 100, 100
    
    last_action_time = 0
    cooldown = 0.5 # seconds between actions
    training_mode = True
    show_mask = False
    
    wizard_step = None
    collecting_samples = []

    print("AI-Powered Virtual Mouse Started.")
    print("Commands: '1'-'4' (Train), 'S' (Save Sample), 'T' (Toggle Control), 'R' (Reset BG), 'M' (Mask)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        # Draw movement area
        cv2.rectangle(frame, (move_box_x, move_box_y), (move_box_x+move_box_w, move_box_y+move_box_h), (255, 0, 0), 2)

        hand_pos, features, palm_center, mask = engine.process(frame)
        
        gesture = "none"
        confidence = 0
        if features is not None:
            gesture, confidence = ai.predict(features)

        draw_info(frame, gesture, confidence, training_mode, wizard_step, len(collecting_samples))

        if show_mask:
            cv2.imshow("Mask View", mask)

        if hand_pos:
            hx, hy = hand_pos
            
            # Map move_box coordinates to screen coordinates
            screen_x = np.interp(hx, [move_box_x, move_box_x+move_box_w], [0, screen_w])
            screen_y = np.interp(hy, [move_box_y, move_box_y+move_box_h], [0, screen_h])

            target_x, target_y = smooth.smooth(screen_x, screen_y)
            
            if not training_mode and mouse_dll:
                if gesture == "pointer" or gesture == "move":
                    mouse_dll.move_mouse(target_x, target_y)

                # Handle Actions
                current_time = time.time()
                if current_time - last_action_time > cooldown:
                    if gesture == "click":
                        mouse_dll.click_mouse()
                        last_action_time = current_time
                    elif gesture == "right_click":
                        mouse_dll.right_click_mouse()
                        last_action_time = current_time
            
            # Draw cursor position on frame for debug
            cv2.circle(frame, (hx, hy), 5, (0, 255, 255), -1)

        cv2.imshow("Virtual Mouse AI", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27: # ESC
            break
        elif key == ord('m') or key == ord('M'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Mask View")
            print(f"Mask View: {'ON' if show_mask else 'OFF'}")
        elif key == ord('t') or key == ord('T'):
            training_mode = not training_mode
            wizard_step = None
            print(f"Mode Switched: {'TRAINING' if training_mode else 'MOUSE ACTIVE'}")
        elif key == ord('r') or key == ord('R'):
            engine.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
            print("Background Reset.")
        
        # Training Logic
        if training_mode:
            if key == ord('2'): wizard_step = "pointer"
            elif key == ord('3'): wizard_step = "click"
            elif key == ord('4'): wizard_step = "right_click"
            
            if wizard_step and features is not None:
                if key == ord('s') or key == ord('S'):
                    collecting_samples.append(features)
                    if len(collecting_samples) >= 20:
                        ai.train_gesture(wizard_step, collecting_samples)
                        print(f"Trained gesture: {wizard_step}")
                        wizard_step = None
                        collecting_samples = []

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
