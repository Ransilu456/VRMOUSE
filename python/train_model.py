import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train():
    # Use absolute paths for reliability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "gesture_data.csv")
    
    if not os.path.exists(data_file):
        return False, f"Error: Data file not found at {data_file}"

    try:
        # Load and validate data
        data = pd.read_csv(data_file, header=None)
        
        if len(data) < 20:
            return False, "Not enough data! Collect at least 20 samples total across all gestures."

        X = data.iloc[:, 1:].values # Features
        y = data.iloc[:, 0].values  # Labels
        
        # Check if we have at least 2 classes
        if len(set(y)) < 2:
            return False, "Not enough variety! Record at least 2 different gestures (e.g. 'none' and 'move')."

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training Random Forest Classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        model_path = os.path.join(script_dir, "gesture_model.pkl")
        joblib.dump(model, model_path)
        
        return True, f"Success! Accuracy: {acc:.2f}. Model saved to {os.path.basename(model_path)}"
    except Exception as e:
        return False, f"Training failed: {e}"

if __name__ == "__main__":
    success, msg = train()
    print(msg)
