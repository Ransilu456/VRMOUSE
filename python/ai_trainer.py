import cv2
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "gesture_data_v2.csv")
    model_path = os.path.join(script_dir, "gesture_model.pkl")
    
    cv2.namedWindow("AI Trainer V2")
    
    status_msg = "V2 Trainer Ready"
    last_acc = 0
    training = False
    
    # Expected features: 5(ext) + 5(angle) + 2(palm) + 2(pinch) = 14
    # Total columns: 1 (label) + 14 (features) = 15
    EXPECTED_COLS = 15

    while True:
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(img, "VRMOUSE V2 AI TRAINER", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        
        if os.path.exists(data_file):
            try:
                data = pd.read_csv(data_file, header=None)
                total = len(data)
                cv2.putText(img, f"Total Samples: {total}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                counts = data[0].value_counts()
                y_off = 150
                for label, count in counts.items():
                    cv2.putText(img, f"{label}: {count}", (50, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_off += 30
            except:
                cv2.putText(img, "Error reading data_v2.csv", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        else:
            cv2.putText(img, "No V2 data found! Run Data Collector first.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.putText(img, f"STATUS: {status_msg}", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if last_acc > 0:
            cv2.putText(img, f"ACCURACY: {last_acc:.2%}", (50, 530), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

        cv2.putText(img, "[T] Train  |  [Q] Quit", (300, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        cv2.imshow("AI Trainer V2", img)
        key = cv2.waitKey(100) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t') and not training:
            training = True
            status_msg = "TRAINING..."
            cv2.imshow("AI Trainer V2", img)
            cv2.waitKey(1)
            
            if os.path.exists(data_file):
                try:
                    data = pd.read_csv(data_file, header=None)
                    # Cleaning
                    data = data[data.apply(lambda x: len(x) == EXPECTED_COLS, axis=1)]
                    
                    if len(data) < 20:
                        status_msg = "Error: Too few samples (min 20)"
                    else:
                        X = data.iloc[:, 1:].values
                        y = data.iloc[:, 0].values
                        
                        # V2 specifies high accuracy and stability
                        # Using Random Forest as requested in spec model options (MLP/RandomForest/XGBoost)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train)
                        
                        y_pred = model.predict(X_test)
                        last_acc = accuracy_score(y_test, y_pred)
                        
                        joblib.dump(model, model_path)
                        status_msg = "SUCCESS: Model saved as gesture_model.pkl"
                except Exception as e:
                    status_msg = f"FAILED: {e}"
            training = False

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
