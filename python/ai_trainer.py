import cv2
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def draw_glass_panel(img, x, y, w, h, opacity=0.4):
    sub_img = img[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, 255, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1-opacity, rect, opacity, 0)
    img[y:y+h, x:x+w] = res
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "gesture_data.csv")
    model_path = os.path.join(script_dir, "gesture_model.pkl")
    
    cv2.namedWindow("AI Trainer")
    
    status_msg = "Ready to train"
    last_acc = 0
    training = False
    
    while True:
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        # Gradient background
        for i in range(600):
            color = int(40 + (i/600)*30)
            img[i, :] = (color, color, color)
            
        draw_glass_panel(img, 50, 50, 700, 500, opacity=0.2)
        cv2.putText(img, "AI GESTURE TRAINER", (70, 90), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        
        # Load Data Stats
        if os.path.exists(data_file):
            try:
                data = pd.read_csv(data_file, header=None)
                counts = data[0].value_counts()
                total = len(data)
                
                cv2.putText(img, f"Data Path: ...{data_file[-40:]}", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(img, f"Total Samples: {total}", (70, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw bars for each class
                y_off = 200
                labels = ["none", "move", "left_click", "right_click", "scroll_up", "scroll_down"]
                for label in labels:
                    count = counts.get(label, 0)
                    cv2.putText(img, f"{label:12}: {count}", (70, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # Bar
                    bar_w = int((count / max(counts.values() if not counts.empty else [1])) * 300)
                    cv2.rectangle(img, (250, y_off - 15), (250 + bar_w, y_off + 5), (0, 200, 0), -1)
                    y_off += 30
            except Exception as e:
                cv2.putText(img, f"Error reading data: {e}", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(img, "No data found! Run Gesture Studio first.", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        # Status & Results
        draw_glass_panel(img, 50, 480, 700, 70, opacity=0.3)
        cv2.putText(img, f"STATUS: {status_msg}", (70, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        if last_acc > 0:
            cv2.putText(img, f"MODEL ACCURACY: {last_acc:.2%}", (70, 535), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

        cv2.putText(img, "[T] Train Model  |  [Q] Quit", (300, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("AI Trainer", img)
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t') and not training:
            training = True
            status_msg = "TRAINING... PLEASE WAIT"
            cv2.imshow("AI Trainer", img)
            cv2.waitKey(10)
            
            if os.path.exists(data_file):
                try:
                    data = pd.read_csv(data_file, header=None)
                    
                    # Data Cleaning: Hand has 21 landmarks (x,y) = 42 features + 1 label = 43 columns
                    initial_len = len(data)
                    data = data[data.apply(lambda x: len(x) == 43, axis=1)]
                    cleaned_len = len(data)
                    
                    if initial_len != cleaned_len:
                        print(f"[Trainer] Cleaned {initial_len - cleaned_len} malformed rows.")

                    if len(data) < 50:
                        status_msg = "ERROR: Need at least 50 valid samples!"
                    elif len(data[0].unique()) < 2:
                        status_msg = "ERROR: Need at least 2 different gesture types!"
                    else:
                        X = data.iloc[:, 1:].values
                        y = data.iloc[:, 0].values
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        
                        y_pred = model.predict(X_test)
                        last_acc = accuracy_score(y_test, y_pred)
                        
                        # Ensure absolute path for saving
                        joblib.dump(model, model_path)
                        status_msg = f"SUCCESS! Accuracy: {last_acc:.2f}. Model saved."
                except Exception as e:
                    status_msg = f"TRAIN FAILED: {e}"
            training = False

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
