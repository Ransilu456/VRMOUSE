import cv2
import numpy as np
import os
import ctypes
import time
import json
from gesture_engine import MediaPipeEngine
from smoothing_engine import AdaptiveSmoother
from mask_renderer import MaskRenderer
from state_machine import GestureState

def load_config(path='config.json'):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    if os.path.exists(abs_path):
        try:
            with open(abs_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Main] Error loading config: {e}")
    return None


def draw_panel(img, x, y, w, h, bg_color=(20, 20, 20), alpha=0.7):
    """Draw a semi-transparent dark panel."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)


def draw_hand_skeleton(frame, landmarks, mp_hands, mp_draw, hand_detected):
    """Draw landmarks and connections with color indicating confidence."""
    if hand_detected and landmarks:
        conn_color = (0, 255, 150)  # green connections
        dot_color = (0, 200, 255)   # cyan dots
        mp_draw.draw_landmarks(
            frame, landmarks, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=dot_color, thickness=2, circle_radius=4),
            mp_draw.DrawingSpec(color=conn_color, thickness=2)
        )


def main():
    config = load_config()
    cap = cv2.VideoCapture(0)

    engine = MediaPipeEngine()

    # Expose mediapipe drawing utils
    import mediapipe as mp
    mp_hands_mod = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    screen_w = config['screen']['width'] if config else 1920
    screen_h = config['screen']['height'] if config else 1080

    smooth = AdaptiveSmoother(screen_w=screen_w, screen_h=screen_h)
    mask_view = MaskRenderer()

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
        print(f"[Main V2] DLL loaded: {dll_path}")
    except Exception as e:
        print(f"!!! [Main V2] DLL LOAD FAILED: {e}")

    mouse_active = False
    prev_time = time.time()
    action_flash_timer = 0
    last_action = "none"
    ripples = []

    print("--- VRMOUSE V2 (Production Grade) ---")
    print("Keys: [T] Toggle Control, [ESC] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape

        # 1. Engine Processing
        out = engine.process(frame)
        action     = out["action"]
        cursor_pos = out["cursor_pos"]
        mask       = out["mask"]
        landmarks  = out["landmarks"]
        states     = out["states"]
        hand_detected = landmarks is not None

        # 2. Draw Hand Skeleton on main frame
        draw_hand_skeleton(frame, landmarks, mp_hands_mod, mp_draw, hand_detected)

        # Index finger tip position indicator
        if hand_detected and cursor_pos:
            cx, cy = cursor_pos
            # Outer ring
            cv2.circle(frame, (cx, cy), 18, (0, 255, 150), 2)
            # Inner dot
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            # Crosshair lines
            cv2.line(frame, (cx - 22, cy), (cx - 12, cy), (0, 255, 150), 1)
            cv2.line(frame, (cx + 12, cy), (cx + 22, cy), (0, 255, 150), 1)
            cv2.line(frame, (cx, cy - 22), (cx, cy - 12), (0, 255, 150), 1)
            cv2.line(frame, (cx, cy + 12), (cx, cy + 22), (0, 255, 150), 1)

        # 3. Ripple effect on action
        if action != "none" and action != last_action and hand_detected and cursor_pos:
            cx, cy = cursor_pos
            col = (0, 255, 80) if "click" in action else (0, 200, 255) if "scroll" in action else (255, 200, 0)
            ripples.append([cx, cy, 5, 1.0, col])
            action_flash_timer = 8  # frames to flash
        last_action = action

        # Draw ripples
        new_ripples = []
        for r in ripples:
            rx, ry, rrad, rop, rcol = r
            if rop > 0.05:
                cv2.circle(frame, (rx, ry), int(rrad), rcol, 2)
                r[2] += 6
                r[3] *= 0.80
                new_ripples.append(r)
        ripples = new_ripples

        # 4. Mask Window
        mask_view.render(frame, mask, landmarks, action,
                         states.get(action, GestureState.IDLE) if action != "none" else "idle")

        # 5. Cursor Mapping & Action Execution
        if cursor_pos and mouse_active:
            cx, cy = cursor_pos
            sx, sy = smooth.smooth(cx, cy)
            target_x = int(sx * (screen_w / w_frame))
            target_y = int(sy * (screen_h / h_frame))
            if mouse_dll:
                mouse_dll.move_mouse(target_x, target_y)

            if action == "left_click":
                if mouse_dll: mouse_dll.click_mouse()
            elif action == "right_click":
                if mouse_dll: mouse_dll.right_click_mouse()
            elif action == "scroll_up":
                if mouse_dll: mouse_dll.scroll_up()
            elif action == "scroll_down":
                if mouse_dll: mouse_dll.scroll_down()

        # 6. HUD ─ TOP LEFT PANEL
        fps = 1 / max(time.time() - prev_time, 0.001)
        prev_time = time.time()

        panel_w, panel_h = 260, 180
        draw_panel(frame, 10, 10, panel_w, panel_h)

        # Hand detection badge
        if hand_detected:
            cv2.circle(frame, (35, 45), 10, (0, 220, 60), -1)
            cv2.putText(frame, "HAND DETECTED", (52, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 60), 1)
        else:
            cv2.circle(frame, (35, 45), 10, (0, 60, 220), -1)
            cv2.putText(frame, "NO HAND", (52, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1)

        # Mouse control status
        ctrl_col = (0, 255, 100) if mouse_active else (80, 80, 255)
        ctrl_txt = "MOUSE: ON  [T] to toggle" if mouse_active else "MOUSE: OFF [T] to toggle"
        cv2.putText(frame, ctrl_txt, (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ctrl_col, 1)

        # Engine type
        eng_txt = "ENGINE: AI MODEL" if engine.model else "ENGINE: HEURISTIC"
        cv2.putText(frame, eng_txt, (20, 107), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Current gesture state
        gesture_label = action.upper().replace("_", " ") if action != "none" else "IDLE"
        if action_flash_timer > 0:
            g_color = (0, 255, 80) if "click" in action else (0, 200, 255) if "scroll" in action else (255, 200, 0)
            action_flash_timer -= 1
        else:
            g_color = (160, 160, 160)
        cv2.putText(frame, f"GESTURE: {gesture_label}", (20, 135), cv2.FONT_HERSHEY_DUPLEX, 0.55, g_color, 1)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        # 7. BOTTOM RIGHT ─ Finger extension mini-display
        if hand_detected and landmarks:
            try:
                features = engine.extractor.extract(landmarks)
                ext = features["extensions"]
                finger_names = ["TH", "IN", "MD", "RG", "PK"]
                bx, by = w_frame - 160, h_frame - 70
                draw_panel(frame, bx - 10, by - 30, 155, 60)
                cv2.putText(frame, "FINGERS:", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                for fi, (name, up) in enumerate(zip(finger_names, ext)):
                    fx = bx + fi * 28
                    col = (0, 255, 100) if up else (60, 60, 60)
                    cv2.rectangle(frame, (fx, by + 5), (fx + 20, by + 30), col, -1)
                    cv2.putText(frame, name, (fx + 1, by + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            except:
                pass

        cv2.imshow("VR Mouse V2", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('t'):
            mouse_active = not mouse_active

    cap.release()
    mask_view.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
