import cv2
import mediapipe as mp
import csv
import os
import time

def main():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    # Define labels to collect
    labels = ["none", "move", "left_click", "right_click", "scroll_up", "scroll_down"]
    current_label_idx = 0
    
    data_file = "gesture_data.csv"
    file_exists = os.path.isfile(data_file)
    
    recording = False
    
    print("--- Gesture Data Collector ---")
    print(f"Current Label: {labels[current_label_idx]}")
    print("Commands:")
    print("  'R' : Toggle Recording")
    print("  'N' : Next Label")
    print("  'ESC' : Quit")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        label = labels[current_label_idx]
        status = "RECORDING" if recording else "IDLE"
        color = (0, 0, 255) if recording else (255, 0, 0)
        
        cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Status: {status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if recording:
                    # Collect relative landmarks (normalized by wrist)
                    wrist = hand_landmarks.landmark[0]
                    row = [label]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x - wrist.x, lm.y - wrist.y])
                    
                    with open(data_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

        cv2.imshow("Data Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('r') or key == ord('R'):
            recording = not recording
            print(f"{'Started' if recording else 'Stopped'} recording for {label}")
        elif key == ord('n') or key == ord('N'):
            current_label_idx = (current_label_idx + 1) % len(labels)
            recording = False
            print(f"Switched to label: {labels[current_label_idx]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
