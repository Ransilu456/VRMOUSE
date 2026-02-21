import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def main():
    data_file = "gesture_data.csv"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Run data_collector.py first.")
        return

    print("Loading data...")
    # CSV has no header, first column is label, rest are coordinates
    data = pd.read_csv(data_file, header=None)
    
    X = data.iloc[:, 1:].values # Features
    y = data.iloc[:, 0].values  # Labels
    
    print(f"Dataset size: {len(data)} samples")
    print(f"Classes: {set(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nTraining Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_path = "python/gesture_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
