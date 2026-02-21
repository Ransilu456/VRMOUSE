import cv2
import numpy as np
import os
import ctypes
import time
from gesture_engine import MediaPipeEngine
from smoothing import Smoother

def load_config(path='config.json'):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    if os.path.exists(abs_path):
        try:
            with open(abs_path, 'r') as f:
                import json
                return json.load(f)
        except Exception as e:
            print(f"[Main] Error loading config: {e}")
    return None

def draw_glass_panel(img, x, y, w, h, opacity=0.4):
    """Draws a semi-transparent 'glass' panel for the HUD"""
    sub_img = img[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, 255, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1-opacity, rect, opacity, 0)
    img[y:y+h, x:x+w] = res
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)

def main():
    config = load_config()
    cap = cv2.VideoCapture(0)
    
    engine = MediaPipeEngine()
    
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
    
    # DLL loading - Absolute Path
    dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mouse_control.dll'))
    mouse_dll = None
    try:
        mouse_dll = ctypes.CDLL(dll_path)
        mouse_dll.move_mouse.argtypes = [ctypes.c_int, ctypes.c_int]
        mouse_dll.click_mouse.argtypes = []
        mouse_dll.right_click_mouse.argtypes = []
        mouse_dll.scroll_up.argtypes = []
        mouse_dll.scroll_down.argtypes = []
        print(f"[Main] DLL loaded: {dll_path}")
    except Exception as e:
        print(f"!!! [Main] DLL LOAD FAILED: {e}")

    last_action_time = 0
    mouse_active = False
    show_mask = False
    show_settings = False
    ripples = []
    
    print("--- MediaPipe Virtual Mouse Pro ---")
    print("Keys: [T] Toggle Control, [M] Toggle Mask, [S] Toggle Settings, [ESC] Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        
        # 1. Engine Processing
        action, cursor_pos, frame, mask = engine.process(frame)
        
        # 2. Ripple Logic
        new_ripples = []
        for r_x, r_y, r_rad, r_op, r_col in ripples:
            if r_op > 0.1:
                cv2.circle(frame, (r_x, r_y), int(r_rad), r_col, 2)
                new_ripples.append((r_x, r_y, r_rad + 4, r_op * 0.9, r_col))
        ripples = new_ripples

        # 3. Cursor Mapping & Smoothing
        if cursor_pos and mouse_active:
            cx, cy = cursor_pos
            if move_box_x < cx < move_box_x + move_box_w and move_box_y < cy < move_box_y + move_box_h:
                target_x = int((cx - move_box_x) * (screen_w / move_box_w))
                target_y = int((cy - move_box_y) * (screen_h / move_box_h))
                sx, sy = smooth.smooth(target_x, target_y)
                if mouse_dll: mouse_dll.move_mouse(int(sx), int(sy))

                # 4. Action Execution
                now = time.time()
                if action == "left_click" and now - last_action_time > click_cooldown:
                    if mouse_dll: mouse_dll.click_mouse()
                    ripples.append((cx, cy, 10, 1.0, (0, 255, 0)))
                    last_action_time = now
                elif action == "right_click" and now - last_action_time > click_cooldown:
                    if mouse_dll: mouse_dll.right_click_mouse()
                    ripples.append((cx, cy, 10, 1.0, (0, 0, 255)))
                    last_action_time = now
                elif action == "scroll_up" and now - last_action_time > scroll_cooldown:
                    if mouse_dll: mouse_dll.scroll_up()
                    last_action_time = now
                elif action == "scroll_down" and now - last_action_time > scroll_cooldown:
                    if mouse_dll: mouse_dll.scroll_down()
                    last_action_time = now

        # 5. UI Overlays
        if show_mask:
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0)

        # HUD
        draw_glass_panel(frame, 10, 10, 260, 110)
        status_text = "ACTIVE" if mouse_active else "PAUSED"
        status_color = (0, 200, 0) if mouse_active else (0, 0, 200)
        cv2.putText(frame, f"MOUSE: {status_text}", (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 1)
        
        mode_label = "AI MODEL" if engine.model else "HEURISTIC"
        cv2.putText(frame, f"ENGINE: {mode_label}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        
        act_col = (255, 100, 0) if action != "none" else (150, 150, 150)
        cv2.putText(frame, f"ACTION: {action.upper()}", (25, 95), cv2.FONT_HERSHEY_DUPLEX, 0.6, act_col, 1)

        cv2.rectangle(frame, (move_box_x, move_box_y), (move_box_x+move_box_w, move_box_y+move_box_h), (255, 200, 0), 1)
        
        cv2.imshow("VR Mouse Pro", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord('t'): mouse_active = not mouse_active
        elif key == ord('m'): show_mask = not show_mask
        elif key == ord('s'): show_settings = not show_settings

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
