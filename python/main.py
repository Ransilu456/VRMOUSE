import cv2
import numpy as np
import os
import ctypes
import time
import json
import mediapipe as mp
from gesture_engine import MediaPipeEngine
from smoothing_engine import AdaptiveSmoother
from mask_renderer import MaskRenderer, GESTURE_CONFIG
from state_machine import GestureState

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_config(path='config.json'):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    if os.path.exists(abs_path):
        try:
            with open(abs_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Main] Error loading config: {e}")
    return None


def draw_panel(img, x, y, w, h, color=(18, 18, 18), alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)


def draw_gesture_card(img, gesture_name, h_frame, w_frame):
    """
    Draws a large gesture mode card on the right side of the frame.
    Shows current gesture in big text with a colored accent bar.
    """
    gcfg  = GESTURE_CONFIG.get(gesture_name, GESTURE_CONFIG["none"])
    label = gcfg[2]
    color = gcfg[1]
    is_active = gesture_name != "none"

    cx = w_frame - 175
    cy = int(h_frame * 0.5)
    cw, ch = 165, 90

    # Panel background
    overlay = img.copy()
    cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)

    # Left accent bar (colored = active gesture)
    bar_col = tuple(color) if is_active else (40, 40, 40)
    cv2.rectangle(img, (cx, cy), (cx + 4, cy + ch), bar_col, -1)

    # Border
    cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), (70, 70, 70), 1)

    # Label text: "MODE" subheading
    cv2.putText(img, "GESTURE MODE", (cx + 10, cy + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # Big gesture label
    font_scale = 0.55 if len(label) > 8 else 0.7
    txt_col = tuple(color) if is_active else (60, 60, 60)
    cv2.putText(img, label, (cx + 10, cy + 52),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, txt_col, 1)

    # State dot
    state_text = "ACTIVE" if is_active else "IDLE"
    dot_col = tuple(color) if is_active else (50, 50, 50)
    cv2.circle(img, (cx + 12, cy + 74), 5, dot_col, -1)
    cv2.putText(img, state_text, (cx + 22, cy + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                tuple(color) if is_active else (70, 70, 70), 1)


def draw_finger_bar(img, landmarks, engine, w_frame, h_frame):
    """Mini finger state bar at bottom center."""
    if not landmarks:
        return
    try:
        features = engine.extractor.extract(landmarks)
        ext = features["extensions"]
    except:
        return

    names  = ["TH", "IN", "MD", "RG", "PK"]
    bw     = 32
    total  = bw * 5 + 4 * 6
    bx     = (w_frame - total) // 2
    by     = h_frame - 52

    draw_panel(img, bx - 8, by - 24, total + 16, 60)
    cv2.putText(img, "FINGER STATE", (bx, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    for i, (name, up) in enumerate(zip(names, ext)):
        fx = bx + i * (bw + 6)
        col = (0, 220, 80) if up else (35, 35, 35)
        # Filled rect
        cv2.rectangle(img, (fx, by), (fx + bw, by + 28), col, -1)
        # Label
        cv2.putText(img, name, (fx + 5, by + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 0) if up else (90, 90, 90), 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    cap    = cv2.VideoCapture(0)

    engine = MediaPipeEngine()
    mp_draw   = mp.solutions.drawing_utils
    mp_hands_ = mp.solutions.hands

    screen_w = config['screen']['width']  if config else 1920
    screen_h = config['screen']['height'] if config else 1080

    smooth    = AdaptiveSmoother(screen_w=screen_w, screen_h=screen_h)
    mask_view = MaskRenderer()

    # DLL
    dll_path  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mouse_control.dll'))
    mouse_dll = None
    try:
        mouse_dll = ctypes.CDLL(dll_path)
        mouse_dll.move_mouse.argtypes        = [ctypes.c_int, ctypes.c_int]
        mouse_dll.click_mouse.argtypes       = []
        mouse_dll.right_click_mouse.argtypes = []
        mouse_dll.scroll_up.argtypes         = []
        mouse_dll.scroll_down.argtypes       = []
        print(f"[Main V2] DLL loaded: {dll_path}")
    except Exception as e:
        print(f"!!! [Main V2] DLL LOAD FAILED: {e}")

    mouse_active    = False
    prev_time       = time.time()
    ripples         = []
    prev_action     = "none"

    print("--- VRMOUSE V2 ---  [T] Toggle Control  [ESC] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape

        # ── Engine ──────────────────────────────────────────────────────────
        out           = engine.process(frame)
        action        = out["action"]
        cursor_pos    = out["cursor_pos"]
        mask          = out["mask"]
        landmarks     = out["landmarks"]
        states        = out["states"]
        hand_detected = landmarks is not None

        # ── Skeleton on frame ────────────────────────────────────────────────
        if hand_detected:
            mp_draw.draw_landmarks(
                frame, landmarks, mp_hands_.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 255, 140), thickness=2)
            )

        # ── Index finger crosshair ───────────────────────────────────────────
        if hand_detected and cursor_pos:
            cx, cy = cursor_pos
            gcfg   = GESTURE_CONFIG.get(action, GESTURE_CONFIG["none"])
            c_col  = tuple(gcfg[1])
            cv2.circle(frame, (cx, cy), 16, c_col, 2)
            cv2.circle(frame, (cx, cy), 4,  (255, 255, 255), -1)
            for dx, dy in [(-24, 0), (24, 0), (0, -24), (0, 24)]:
                cv2.line(frame, (cx + dx // 2, cy + dy // 2),
                                (cx + dx, cy + dy), c_col, 1)

        # ── Ripple on gesture change ─────────────────────────────────────────
        if action != prev_action and action != "none" and cursor_pos:
            gcfg = GESTURE_CONFIG.get(action, GESTURE_CONFIG["none"])
            ripples.append([cursor_pos[0], cursor_pos[1], 8, 1.0, tuple(gcfg[1])])
        prev_action = action

        new_ripples = []
        for r in ripples:
            rx, ry, rrad, rop, rcol = r
            if rop > 0.04:
                cv2.circle(frame, (rx, ry), int(rrad), rcol, 2)
                r[2] += 8; r[3] *= 0.78
                new_ripples.append(r)
        ripples = new_ripples

        # ── Mask window ──────────────────────────────────────────────────────
        state_val = states.get(action, GestureState.IDLE) if action != "none" else GestureState.IDLE
        mask_view.render(frame, mask, landmarks, action, str(state_val))

        # ── Mouse control ────────────────────────────────────────────────────
        if cursor_pos and mouse_active:
            cx, cy = cursor_pos
            sx, sy = smooth.smooth(cx, cy)
            tx     = int(sx * (screen_w / w_frame))
            ty     = int(sy * (screen_h / h_frame))
            if mouse_dll: mouse_dll.move_mouse(tx, ty)

            if action == "left_click"  and mouse_dll: mouse_dll.click_mouse()
            elif action == "right_click" and mouse_dll: mouse_dll.right_click_mouse()
            elif action == "scroll_up"   and mouse_dll: mouse_dll.scroll_up()
            elif action == "scroll_down" and mouse_dll: mouse_dll.scroll_down()

        # ── FPS ──────────────────────────────────────────────────────────────
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 0.001)
        prev_time = now

        # ════════════════════════════════════════════════════════════════════
        #  HUD ─ TOP LEFT: System Status panel
        # ════════════════════════════════════════════════════════════════════
        draw_panel(frame, 10, 10, 250, 115)

        # Title
        cv2.putText(frame, "VRMOUSE  V2", (20, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 220), 1)
        cv2.line(frame, (16, 44), (254, 44), (55, 55, 55), 1)

        # Hand detection row
        dot_col = (0, 215, 60) if hand_detected else (50, 50, 200)
        det_txt = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
        det_col = (0, 215, 60)  if hand_detected else (80, 80, 220)
        cv2.circle(frame, (26, 62), 6, dot_col, -1)
        cv2.putText(frame, det_txt, (38, 67),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, det_col, 1)

        # Mouse control row
        mc_col = (0, 240, 100) if mouse_active else (80, 80, 220)
        mc_txt = "Mouse: ON  — [T] OFF" if mouse_active else "Mouse: OFF — [T] ON"
        cv2.circle(frame, (26, 86), 6, mc_col, -1)
        cv2.putText(frame, mc_txt, (38, 91),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, mc_col, 1)

        # Engine + FPS row
        eng_txt = f"Engine: {'AI' if engine.model else 'Heuristic'}    FPS:{fps:.0f}"
        cv2.putText(frame, eng_txt, (20, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        # ════════════════════════════════════════════════════════════════════
        #  HUD ─ RIGHT: Big Gesture Mode Card
        # ════════════════════════════════════════════════════════════════════
        draw_gesture_card(frame, action, h_frame, w_frame)

        # ════════════════════════════════════════════════════════════════════
        #  HUD ─ BOTTOM: Finger State Bar
        # ════════════════════════════════════════════════════════════════════
        draw_finger_bar(frame, landmarks, engine, w_frame, h_frame)

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
