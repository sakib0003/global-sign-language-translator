
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from collections import deque, Counter
from landmark_utils import normalize_landmarks


# =====================================
# 1. LOAD NORMALIZED DIGIT MODEL
# =====================================

MODEL_PATH = "models/bdsl47_digits_normalized.joblib"

print("Loading Digit Model...")

data = joblib.load(MODEL_PATH)

model = data["model"]
FEATURES = data["feature_columns"]

if not data.get("normalized", False):
    raise SystemExit("Model is not normalized!")

print("Digit Model loaded successfully!")


# =====================================
# 2. SETTINGS
# =====================================

CONFIDENCE_THRESHOLD = 0.40

predictions = deque(maxlen=15)

current_prediction = "Waiting..."
current_confidence = 0.0


# =====================================
# 3. MEDIAPIPE
# =====================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# =====================================
# 4. WEBCAM
# =====================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    hands.close()
    raise SystemExit("Webcam could not be opened!")

print("\nDIGIT CALIBRATION TEST")
print("----------------------")
print("Hold one sign steadily.")
print("Press R to reset.")
print("Press Q to quit.")


# =====================================
# 5. MAIN LOOP
# =====================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(rgb)

        display_prediction = "Show your hand"
        confidence_text = ""
        stability_text = "Stability: 0/15"

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            values = []

            for landmark in hand.landmark:

                values.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            try:

                normalized = normalize_landmarks(values)

                features = pd.DataFrame(
                    [normalized],
                    columns=FEATURES,
                    dtype=np.float32
                )

                probabilities = model.predict_proba(
                    features
                )[0]

                best_index = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[best_index]
                )

                prediction = model.classes_[best_index]

                if confidence >= CONFIDENCE_THRESHOLD:

                    predictions.append(prediction)

                else:

                    predictions.append("Uncertain")

                valid = [
                    p for p in predictions
                    if p != "Uncertain"
                ]

                if valid:

                    counts = Counter(valid)

                    stable_prediction, votes = (
                        counts.most_common(1)[0]
                    )

                    display_prediction = (
                        stable_prediction.split("__")[-1]
                    )

                    stability_text = (
                        f"Stability: {votes}/15"
                    )

                    confidence_text = (
                        f"Current confidence: {confidence * 100:.1f}%"
                    )

                else:

                    display_prediction = "Uncertain"

                current_confidence = confidence

            except ValueError:

                predictions.clear()
                display_prediction = "Invalid landmarks"

        else:

            predictions.clear()

        # =================================
        # 6. DISPLAY
        # =================================

        cv2.rectangle(
            frame,
            (10, 10),
            (frame.shape[1] - 10, 155),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "DIGIT CALIBRATION",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Prediction: {display_prediction}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            confidence_text,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            stability_text,
            (20, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "BdSL Digit Calibration",
            frame
        )

        # =================================
        # 7. KEYBOARD
        # =================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):

            predictions.clear()

            print("Calibration history reset.")

finally:

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("Calibration test closed.")
