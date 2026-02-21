import cv2
import numpy as np
import time
import os
import ctypes
import json

# Load config
def load_config(path='config.json'):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    if os.path.exists(abs_path):
        with open(abs_path, 'r') as f:
            return json.load(f)
    return None

config = load_config()

# Load the C++ DLL via ctypes
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mouse_control.dll'))
try:
    mouse_dll = ctypes.CDLL(dll_path)
    # Define argument types for the DLL functions
    mouse_dll.move_mouse.argtypes = [ctypes.c_int, ctypes.c_int]
    mouse_dll.click_mouse.argtypes = []
    mouse_dll.right_click_mouse.argtypes = []
    mouse_dll.scroll_up.argtypes = []
    mouse_dll.scroll_down.argtypes = []
    print(f"DLL loaded successfully from {dll_path}")
except Exception as e:
    print(f"Could not load DLL at {dll_path}: {e}")
    mouse_dll = None

from gesture_engine import MediaPipeEngine
from smoothing import Smoother

def main():
    cap = cv2.VideoCapture(0)
    engine = MediaPipeEngine()
    
    # Use config values
    s_cfg = config['smoothing'] if config else {"min_cutoff": 0.5, "beta": 0.05}
    smooth = Smoother(min_cutoff=s_cfg['min_cutoff'], beta=s_cfg['beta'])

    screen_w = config['screen']['width'] if config else 1920
    screen_h = config['screen']['height'] if config else 1080 
    
    mbox = config['tracking_box'] if config else {"x": 100, "y": 100, "w": 400, "h": 300}
    move_box_x, move_box_y = mbox['x'], mbox['y']
    move_box_w, move_box_h = mbox['w'], mbox['h']
    
    cd = config['cooldowns'] if config else {"click": 0.5, "scroll": 0.1}
    click_cooldown = cd['click']
    scroll_cooldown = cd['scroll']
    
    last_action_time = 0
    mouse_active = False
    show_mask = False
    
    print("MediaPipe Virtual Mouse Started.")
    print("Commands: 'T' (Toggle Control), 'M' (Toggle Mask), 'ESC' (Quit)")

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, (move_box_x, move_box_y), (move_box_x+move_box_w, move_box_y+move_box_h), (255, 0, 0), 2)

        # Get action from the engine
        action, cursor_pos, frame, mask = engine.process(frame)
        
        # UI Text
        status_text = "ACTIVE" if mouse_active else "PAUSED (Press 'T')"
        color = (0, 255, 0) if mouse_active else (0, 0, 255)
        cv2.putText(frame, f"Mouse: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Action: {action.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if show_mask:
            cv2.imshow("Mask View", mask)

        if cursor_pos:
            hx, hy = cursor_pos
            
            # Map move_box to screen
            screen_x = np.interp(hx, [move_box_x, move_box_x+move_box_w], [0, screen_w])
            screen_y = np.interp(hy, [move_box_y, move_box_y+move_box_h], [0, screen_h])
            target_x, target_y = smooth.smooth(screen_x, screen_y)
            
            cv2.circle(frame, (hx, hy), 5, (0, 255, 255), -1)

            if mouse_active and mouse_dll:
                if action in ["move", "left_click", "right_click", "scroll_up", "scroll_down"]:
                    mouse_dll.move_mouse(target_x, target_y)

                current_time = time.time()
                
                # Handling clicks
                if action in ["left_click", "right_click"]:
                    if current_time - last_action_time > click_cooldown:
                        if action == "left_click":
                            mouse_dll.click_mouse()
                        elif action == "right_click":
                            mouse_dll.right_click_mouse()
                        last_action_time = current_time
                        
                # Handling scrolls
                elif action in ["scroll_up", "scroll_down"]:
                    if current_time - last_action_time > scroll_cooldown:
                        if action == "scroll_up":
                            mouse_dll.scroll_up()
                        elif action == "scroll_down":
                            mouse_dll.scroll_down()
                        last_action_time = current_time

        cv2.imshow("MediaPipe Tracker", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('m') or key == ord('M'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Mask View")
        elif key == ord('t') or key == ord('T'):
            mouse_active = not mouse_active

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
