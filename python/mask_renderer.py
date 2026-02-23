import cv2
import numpy as np
import mediapipe as mp

# Gesture display config: icon text + color
GESTURE_CONFIG = {
    "none":        ("--",   (80,  80,  80),  "IDLE"),
    "move":        ("✦",    (0,   220, 255), "MOVE"),
    "left_click":  ("●",    (0,   255, 100), "LEFT CLICK"),
    "right_click": ("●",    (255, 120, 0),   "RIGHT CLICK"),
    "scroll":      ("⇅",    (200, 100, 255), "SCROLL"),
    "double_click":("◉",    (0,   200, 255), "DOUBLE CLICK"),
    "drag":        ("✥",    (255, 200, 0),   "DRAG"),
}


class MaskRenderer:
    """Renders a premium Black & White Hand Mask window with gesture HUD."""

    def __init__(self, window_name="VR Mouse — Hand Mask"):
        self.window_name = window_name
        self.mp_draw  = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 320)

    def _draw_bar(self, img, x, y, w_bar, h_bar, label, color):
        """Draws a colored labelled bar."""
        cv2.rectangle(img, (x, y), (x + w_bar, y + h_bar), color, -1)
        cv2.putText(img, label, (x + 4, y + h_bar - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    def render(self, frame, mask, landmarks=None,
               gesture_name="none", state="idle"):

        h, w = frame.shape[:2]

        # ── Base: Pure black canvas ─────────────────────────────────────────
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # ── White hand silhouette ────────────────────────────────────────────
        if mask is not None and mask.any():
            white_hand = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # Tint based on gesture
            gcfg = GESTURE_CONFIG.get(gesture_name, GESTURE_CONFIG["none"])
            tint_color = np.array(gcfg[1], dtype=np.uint8)
            tinted = np.where(white_hand > 0, tint_color, 0).astype(np.uint8)
            canvas = cv2.addWeighted(canvas, 1.0, tinted, 0.6, 0)
            canvas = cv2.addWeighted(canvas, 1.0, white_hand, 0.4, 0)

        # ── Skeleton overlay ─────────────────────────────────────────────────
        if landmarks:
            self.mp_draw.draw_landmarks(
                canvas, landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(50, 255, 200), thickness=2, circle_radius=3),
                self.mp_draw.DrawingSpec(color=(200, 200, 200), thickness=1)
            )

        # ── Top bar ──────────────────────────────────────────────────────────
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, 44), (w, 44), (60, 60, 60), 1)

        cv2.putText(canvas, "HAND MASK", (10, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (210, 210, 210), 1)

        # Detection dot
        dot_col = (0, 220, 60) if landmarks else (50, 50, 200)
        cv2.circle(canvas, (w - 20, 22), 8, dot_col, -1)

        # ── State pill ───────────────────────────────────────────────────────
        state_str = state.upper() if isinstance(state, str) else str(state).upper()
        state_col = (0, 180, 60) if state_str == "ACTIVE" else \
                    (255, 180, 0) if state_str in ("CONFIRMED", "DETECTED") else \
                    (80, 80, 80)
        pill_txt = f"  {state_str}  "
        (tw, th), _ = cv2.getTextSize(pill_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        px = w - tw - 14
        cv2.rectangle(canvas, (px - 4, h - 36), (px + tw + 4, h - 36 + th + 10), state_col, -1)
        cv2.putText(canvas, pill_txt, (px, h - 36 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # ── Bottom gesture banner ─────────────────────────────────────────────
        gcfg = GESTURE_CONFIG.get(gesture_name, GESTURE_CONFIG["none"])
        g_label = gcfg[2]
        g_color = gcfg[1]

        bh = 48
        banner_overlay = canvas.copy()
        cv2.rectangle(banner_overlay, (0, h - bh), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(banner_overlay, 0.8, canvas, 0.2, 0, canvas)

        # Accent line
        if gesture_name != "none":
            cv2.line(canvas, (0, h - bh), (w, h - bh), tuple(g_color), 2)

        (tw2, th2), _ = cv2.getTextSize(g_label, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
        tx = (w - tw2) // 2
        cv2.putText(canvas, g_label, (tx, h - bh + 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75,
                    tuple(g_color) if gesture_name != "none" else (70, 70, 70), 1)

        cv2.imshow(self.window_name, canvas)

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
