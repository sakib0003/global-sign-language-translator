
import cv2
import joblib
import numpy as np
import mediapipe as mp

from collections import deque, Counter
from landmark_utils import normalize_landmarks

# ==========================================
# 1. LOAD SVM MODEL
# ==========================================

MODEL_PATH = "models/bdsl47_alphabet_svm.joblib"

print("Loading SVM model...")

data = joblib.load(MODEL_PATH)

model = data["model"]
FEATURE_COLUMNS = data["feature_columns"]

print("SVM model loaded successfully!")
print("Total classes:", len(model.classes_))

# ==========================================
# 2. SETTINGS
# ==========================================

CONFIDENCE_THRESHOLD = 0.50

HISTORY_SIZE = 15
REQUIRED_VOTES = 10

predictions = deque(maxlen=HISTORY_SIZE)

# ==========================================
# 3. MEDIAPIPE SETUP
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================
# 4. CAMERA SETUP
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera could not be opened!")

print("\nBdSL SVM Alphabet Recognition Started")
print("Press Q to quit")
print("Press R to reset prediction")

# ==========================================
# 5. MAIN LOOP
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame error!")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        current_label = "Show your hand"
        stable_label = "Waiting..."
        confidence = 0.0

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            values = []

            for point in hand_landmarks.landmark:

                values.extend([
                    point.x,
                    point.y,
                    point.z
                ])

            try:

                normalized = normalize_landmarks(values)

                features = np.array(
                    [normalized],
                    dtype=np.float32
                )

                # SVM prediction
                probabilities = model.predict_proba(features)[0]

                best_index = np.argmax(probabilities)

                confidence = float(
                    probabilities[best_index]
                )

                raw_label = model.classes_[best_index]

                current_label = str(
                    raw_label
                ).split("__")[-1]

                if confidence >= CONFIDENCE_THRESHOLD:

                    predictions.append(current_label)

                else:

                    predictions.clear()

                if predictions:

                    label, votes = Counter(
                        predictions
                    ).most_common(1)[0]

                    if votes >= REQUIRED_VOTES:

                        stable_label = label

            except ValueError:

                predictions.clear()

                current_label = "Invalid landmarks"

        else:

            predictions.clear()

        # ==================================
        # 6. DISPLAY INFORMATION
        # ==================================

        cv2.rectangle(
            frame,
            (10, 10),
            (550, 170),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "BdSL Alphabet - SVM RBF",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Prediction: {current_label}",
            (25, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.1f}%",
            (25, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Stable: {stable_label}",
            (25, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "BdSL Alphabet Recognition - SVM",
            frame
        )

        # ==================================
        # 7. KEYBOARD CONTROLS
        # ==================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        elif key == ord("r"):

            predictions.clear()

            print("Prediction history reset!")

finally:

    camera.release()

    hands.close()

    cv2.destroyAllWindows()

    print("Camera closed.")
