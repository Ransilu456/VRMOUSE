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
        try:
            with open(abs_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    return None

def draw_glass_panel(img, x, y, w, h, opacity=0.4):
    """Draws a semi-transparent 'glass' panel for the HUD"""
    sub_img = img[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, 255, dtype=np.uint8) # White background
    res = cv2.addWeighted(sub_img, 1-opacity, rect, opacity, 0)
    img[y:y+h, x:x+w] = res
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)

def main():
    config = load_config()
    cap = cv2.VideoCapture(0)
    
    from gesture_engine import MediaPipeEngine
    from smoothing import Smoother

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
    
    # DLL loading
    dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mouse_control.dll'))
    mouse_dll = None
    try:
        mouse_dll = ctypes.CDLL(dll_path)
        mouse_dll.move_mouse.argtypes = [ctypes.c_int, ctypes.c_int]
        mouse_dll.click_mouse.argtypes = []
        mouse_dll.right_click_mouse.argtypes = []
        mouse_dll.scroll_up.argtypes = []
        mouse_dll.scroll_down.argtypes = []
        print(f"DLL loaded successfully from {dll_path}")
    except Exception as e:
        print(f"Could not load DLL at {dll_path}: {e}")

    last_action_time = 0
    mouse_active = False
    show_mask = False
    show_settings = False
    
    # Ripple Animation State: [(x, y, radius, opacity), ...]
    ripples = []
    
    print("MediaPipe Virtual Mouse Started.")
    print("Commands: 'T' (Toggle Control), 'M' (Toggle Mask), 'S' (Toggle Settings), 'ESC' (Quit)")

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        
        # 1. Background Logic & Engine Processing
        action, cursor_pos, frame, mask = engine.process(frame)
        
        # 2. Draw Tracking Box (Modern thin lines)
        cv2.rectangle(frame, (move_box_x, move_box_y), (move_box_x+move_box_w, move_box_y+move_box_h), (255, 200, 0), 1)
        cv2.putText(frame, "TRACKING AREA", (move_box_x, move_box_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # 3. HUD - Top Left (Status)
        draw_glass_panel(frame, 10, 10, 240, 100)
        status_text = "ACTIVE" if mouse_active else "PAUSED"
        status_color = (0, 200, 0) if mouse_active else (0, 0, 200)
        cv2.putText(frame, f"MOUSE: {status_text}", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)
        
        model_type = "AI (MODEL)" if engine.model else "HEURISTIC"
        cv2.putText(frame, f"ENGINE: {model_type}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        
        action_color = (255, 100, 0) if action != "none" else (150, 150, 150)
        cv2.putText(frame, f"ACTION: {action.upper()}", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6, action_color, 1)

        # 4. Settings Overlay (Top Right)
        if show_settings and config:
            draw_glass_panel(frame, w_frame - 220, 10, 210, 160, opacity=0.6)
            cv2.putText(frame, "SETTINGS (S)", (w_frame - 210, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
            y_off = 55
            for k, v in config['thresholds'].items():
                cv2.putText(frame, f"{k}: {v}", (w_frame - 210, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
                y_off += 20
            for k, v in config['cooldowns'].items():
                cv2.putText(frame, f"CD {k}: {v}", (w_frame - 210, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
                y_off += 20

        if show_mask:
            cv2.imshow("Mask View", mask)

        if cursor_pos:
            hx, hy = cursor_pos
            
            # Map move_box to screen
            screen_x = np.interp(hx, [move_box_x, move_box_x+move_box_w], [0, screen_w])
            screen_y = np.interp(hy, [move_box_y, move_box_y+move_box_h], [0, screen_h])
            target_x, target_y = smooth.smooth(screen_x, screen_y)
            
            # Cursor HUD element
            cv2.circle(frame, (hx, hy), 6, (255, 255, 255), 1)
            cv2.circle(frame, (hx, hy), 2, (0, 255, 255), -1)

            if mouse_active and mouse_dll:
                if action in ["move", "left_click", "right_click", "scroll_up", "scroll_down"]:
                    mouse_dll.move_mouse(target_x, target_y)

                current_time = time.time()
                
                # Action Logic
                triggered = False
                if action in ["left_click", "right_click"]:
                    if current_time - last_action_time > click_cooldown:
                        if action == "left_click":
                            mouse_dll.click_mouse()
                            ripples.append([hx, hy, 5, 1.0, (0, 255, 0)]) # Green ripple
                        elif action == "right_click":
                            mouse_dll.right_click_mouse()
                            ripples.append([hx, hy, 5, 1.0, (0, 0, 255)]) # Red ripple
                        triggered = True
                        last_action_time = current_time
                        
                elif action in ["scroll_up", "scroll_down"]:
                    if current_time - last_action_time > scroll_cooldown:
                        if action == "scroll_up":
                            mouse_dll.scroll_up()
                        elif action == "scroll_down":
                            mouse_dll.scroll_down()
                        triggered = True
                        last_action_time = current_time

        # 5. Update & Draw Ripples
        new_ripples = []
        for r in ripples:
            rh_x, rh_y, radius, alpha, color = r
            # Draw ripple
            overlay = frame.copy()
            cv2.circle(overlay, (rh_x, rh_y), int(radius), color, 2)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Update
            radius += 4
            alpha -= 0.1
            if alpha > 0:
                new_ripples.append([rh_x, rh_y, radius, alpha, color])
        ripples = new_ripples

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
        elif key == ord('s') or key == ord('S'):
            show_settings = not show_settings

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
