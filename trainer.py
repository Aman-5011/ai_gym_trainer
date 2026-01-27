import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# -----------------------------
# Exercise selection
# -----------------------------
exercise = input("Choose exercise (squat / pushup / bicep): ").lower()

VIDEO_PATHS = {
    "squat": r"C:\Users\Acer\.vscode\AI_MLproject\videos\squat_vdo.mp4",
    "pushup": r"C:\Users\Acer\Desktop\AI_Gym_Trainer\videos\pushup_video.mp4",
    "bicep": r"C:\Users\Acer\Desktop\AI_Gym_Trainer\videos\bicep_video.mp4"
}

MODEL_PATHS = {
    "squat": "models/squat_model.pkl",
    "pushup": "models/pushup_model.pkl",
    "bicep": "models/bicep_model.pkl"
}

FEATURES = {
    "squat": ["knee_angle", "hip_angle", "back_angle"],
    "pushup": ["elbow_angle", "body_angle"],
    "bicep": ["elbow_angle", "shoulder_angle"]
}

# -----------------------------
# Load trained model (Pipeline)
# -----------------------------
model = joblib.load(MODEL_PATHS[exercise])

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Angle calculation
# -----------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])

    angle = abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# -----------------------------
# Video input
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATHS[exercise])

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

cv2.namedWindow("AI Fitness Trainer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Fitness Trainer", 450, 600)

# -----------------------------
# Prediction smoothing buffer
# -----------------------------
prediction_buffer = deque(maxlen=10)

total_frames = 0
correct_frames = 0

# -----------------------------
# Video loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # -----------------------------
        # Angle calculations
        # -----------------------------
        feature_map = {
            "knee_angle": calculate_angle(hip, knee, ankle),
            "hip_angle": calculate_angle(shoulder, hip, knee),
            "back_angle": calculate_angle(shoulder, hip, [hip[0], hip[1] - 0.1]),
            "elbow_angle": calculate_angle(shoulder, elbow, wrist),
            "body_angle": calculate_angle(shoulder, hip, ankle),
            "shoulder_angle": calculate_angle(elbow, shoulder, hip)
        }

        # -----------------------------
        # Prepare features (DataFrame)
        # -----------------------------
        features = [[feature_map[f] for f in FEATURES[exercise]]]
        features_df = pd.DataFrame(features, columns=FEATURES[exercise])

        # -----------------------------
        # Probability-based prediction
        # -----------------------------
        prob_correct = model.predict_proba(features_df)[0][1]
        prediction_buffer.append(prob_correct)

        avg_prob = np.mean(prediction_buffer)

        # Threshold (tunable)
        is_correct = avg_prob >= 0.6

        total_frames += 1
        if is_correct:
            correct_frames += 1
            status = "Correct"
            color = (0, 255, 0)
        else:
            status = "Incorrect"
            color = (0, 0, 255)

        accuracy = (correct_frames / total_frames) * 100

        # -----------------------------
        # Display
        # -----------------------------
        cv2.putText(image, f"Posture: {status}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(image, f"Accuracy: {accuracy:.2f}%", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        mp_draw.draw_landmarks(image, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

    cv2.imshow("AI Fitness Trainer", image)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
