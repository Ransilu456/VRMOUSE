import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np
from gesture_engine import MediaPipeEngine

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
    
    labels = ["none", "move", "left_click", "right_click", "scroll", "double_click"]
    current_label_idx = 0
    
    # Use absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "gesture_data_v2.csv") # New file for V2
    
    recording = False
    status_msg = f"V2 Studio Ready ({os.path.basename(data_file)})"
    
    print(f"--- VRMOUSE V2 Gesture Studio ---")
    print(f"Saving V2 features to: {data_file}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape
        
        counts = get_sample_counts(data_file)
        
        # UI Overlays
        cv2.rectangle(frame, (10, 10), (350, 120), (30, 30, 30), -1)
        cv2.putText(frame, "V2 GESTURE STUDIO", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(frame, f"STATUS: {status_msg}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Side Menu
        for i, label in enumerate(labels):
            color = (0, 255, 0) if i == current_label_idx else (180, 180, 180)
            prefix = "> " if i == current_label_idx else "  "
            count = counts.get(label, 0)
            cv2.putText(frame, f"{prefix}{i+1}. {label} ({count})", (w_frame - 200, 40 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Engine Process
        out = engine.process(frame)
        mask = out["mask"]
        landmarks = out["landmarks"]
        
        if landmarks and recording:
            # V2 Feature Engineering
            features = engine.extractor.get_flattened_vector(landmarks)
            row = [labels[current_label_idx]] + features
            
            with open(data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        if recording:
            cv2.circle(frame, (30, h_frame - 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING", (50, h_frame - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow("V2 Gesture Studio", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord('r'):
            recording = not recording
            status_msg = "Recording..." if recording else "Paused"
        elif ord('1') <= key <= ord('6'):
            current_label_idx = key - ord('1')
            recording = False
            status_msg = f"Ready: {labels[current_label_idx]}"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
