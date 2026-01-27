import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os


def train_model(csv_path, feature_cols, model_path):
    # -----------------------------
    # Load data
    # -----------------------------
    data = pd.read_csv(csv_path)

    X = data[feature_cols]
    y = data["label"]

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # ML Pipeline (IMPORTANT)
    # -----------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    # -----------------------------
    # Train model
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # Save model
    # -----------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model trained and saved at: {model_path}")


# -----------------------------
# Exercise selection
# -----------------------------
exercise = input("Choose exercise (squat / pushup / bicep): ").lower()

if exercise == "squat":
    train_model(
        "data/squat.csv",
        ["knee_angle", "hip_angle", "back_angle"],
        "models/squat_model.pkl"
    )

elif exercise == "pushup":
    train_model(
        "data/pushup.csv",
        ["elbow_angle", "body_angle"],
        "models/pushup_model.pkl"
    )

elif exercise == "bicep":
    train_model(
        "data/bicep.csv",
        ["elbow_angle", "shoulder_angle"],
        "models/bicep_model.pkl"
    )

else:
    print("Invalid exercise selected")
