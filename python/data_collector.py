import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
from train_model import train
from gesture_engine import MediaPipeEngine

def draw_glass_panel(img, x, y, w, h, opacity=0.4):
    """Draws a semi-transparent 'glass' panel for the HUD"""
    sub_img = img[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, 255, dtype=np.uint8) # White background
    res = cv2.addWeighted(sub_img, 1-opacity, rect, opacity, 0)
    img[y:y+h, x:x+w] = res
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)

def get_sample_counts(data_file):
    counts = {}
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    label = row[0]
                    counts[label] = counts.get(label, 0) + 1
    return counts

def main():
    cap = cv2.VideoCapture(0)
    engine = MediaPipeEngine()
    
    labels = ["none", "move", "left_click", "right_click", "scroll_up", "scroll_down"]
    current_label_idx = 0
    
    # Use absolute paths for reliability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "gesture_data.csv")
    
    recording = False
    status_msg = f"Ready (CSV: {os.path.basename(data_file)})"
    status_color = (100, 100, 100)
    
    print(f"--- Gesture Studio Started ---")
    print(f"Saving data to: {data_file}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        
        counts = get_sample_counts(data_file)
        
        # 1. UI - Header Panel
        draw_glass_panel(frame, 10, 10, 350, 100)
        cv2.putText(frame, "GESTURE STUDIO", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(frame, f"STATUS: {status_msg[:40]}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        total_samples = sum(counts.values())
        cv2.putText(frame, f"TOTAL SAMPLES: {total_samples}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

        # Mask Overlay Toggle
        show_mask = True # Always on in data collector for transparency

        # 2. UI - Side Menu
        draw_glass_panel(frame, w_frame - 200, 10, 190, 220, opacity=0.6)
        cv2.putText(frame, "SELECT ACTION", (w_frame - 190, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1)
        for i, label in enumerate(labels):
            y_pos = 60 + (i * 25)
            prefix = "> " if i == current_label_idx else "  "
            color = (0, 0, 0) if i == current_label_idx else (100, 100, 100)
            count = counts.get(label, 0)
            cv2.putText(frame, f"{prefix}{i+1}. {label} ({count})", (w_frame - 185, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 3. UI - Controls Guide
        draw_glass_panel(frame, 10, h_frame - 80, 400, 70)
        cv2.putText(frame, "[1-6] Select Action  |  [R] Toggle Record", (20, h_frame - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
        cv2.putText(frame, "[T] Train Model      |  [ESC] Quit", (20, h_frame - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

        # 4. Processing Hand & Recording using the Engine
        out = engine.process(frame)
        frame = out["frame"]
        mask = out["mask"]
        landmarks = out["landmarks"]
        
        if landmarks:
            if show_mask:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                frame = cv2.addWeighted(frame, 0.6, mask_rgb, 0.4, 0)
            
            if recording:
                wrist = landmarks.landmark[0]
                features = engine.extract_features(landmarks)
                row = [labels[current_label_idx]] + features
                
                with open(data_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

        cv2.imshow("Gesture Studio", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('r') or key == ord('R'):
            recording = not recording
            status_msg = "Recording..." if recording else "Ready to record"
        elif ord('1') <= key <= ord('6'):
            current_label_idx = key - ord('1')
            recording = False
            status_msg = f"Switched to {labels[current_label_idx]}"
        elif key == ord('t') or key == ord('T'):
            recording = False
            status_msg = "Training..."
            cv2.imshow("Gesture Studio", frame)
            cv2.waitKey(100) # Give UI a moment to show "Training..."
            
            success, msg = train()
            status_msg = msg
            status_color = (0, 150, 0) if success else (0, 0, 200)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
