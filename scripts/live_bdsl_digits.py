
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp

from PIL import Image, ImageDraw, ImageFont
from collections import deque, Counter
from pathlib import Path

from landmark_utils import normalize_landmarks


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "bdsl47_digits_normalized.joblib"

FONT_PATH = (
    Path.home()
    / "AppData/Local/Microsoft/Windows/Fonts/"
    / "NotoSansBengali-VariableFont_wdth,wght.ttf"
)

WINDOW_NAME = "Bangla Sign Language Digit Translator"

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

CONFIDENCE_THRESHOLD = 0.40
REQUIRED_VOTES = 10
HISTORY_SIZE = 15


BANGLA_DIGITS = {
    "sign00": "০",
    "sign01": "১",
    "sign02": "২",
    "sign03": "৩",
    "sign04": "৪",
    "sign05": "৫",
    "sign06": "৬",
    "sign07": "৭",
    "sign08": "৮",
    "sign09": "৯"
}


# ==========================================
# 2. LOAD BENGALI FONT
# ==========================================

print("Loading Bengali font...")

if not FONT_PATH.exists():
    raise SystemExit(f"Font not found: {FONT_PATH}")

bengali_font = ImageFont.truetype(
    str(FONT_PATH), 48
)

small_bengali_font = ImageFont.truetype(
    str(FONT_PATH), 27
)


# ==========================================
# 3. LOAD MODEL
# ==========================================

print("Loading normalized Digit Model...")

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found: {MODEL_PATH}")

model_data = joblib.load(MODEL_PATH)

if not model_data.get("normalized", False):
    raise SystemExit("ERROR: Model is not normalized!")

model = model_data["model"]
FEATURES = model_data["feature_columns"]

model_labels = {
    str(label).split("__")[-1]
    for label in model.classes_
}

missing_labels = set(BANGLA_DIGITS) - model_labels

if missing_labels:
    raise SystemExit(
        f"Missing model labels: {missing_labels}"
    )

print("Digit Model loaded successfully!")
print("Total classes:", len(model.classes_))


# ==========================================
# 4. PREDICTION HISTORY
# ==========================================

predictions = deque(maxlen=HISTORY_SIZE)


# ==========================================
# 5. MEDIAPIPE
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
# 6. START CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    hands.close()
    raise SystemExit("ERROR: Webcam could not be opened!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# Create the correct window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Enable Full Screen
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print("\n======================================")
print("BANGLA SIGN LANGUAGE DIGIT TRANSLATOR")
print("======================================")
print("Show a digit sign with your hand.")
print("R = Reset prediction")
print("Q = Quit")
print("======================================")


# ==========================================
# 7. DRAW BENGALI TEXT
# ==========================================

def draw_bengali_text(
    frame,
    text,
    position,
    font,
    color=(0, 255, 0)
):
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(image)

    draw.text(
        position,
        text,
        font=font,
        fill=color,
        stroke_width=1,
        stroke_fill=(0, 0, 0)
    )

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


# ==========================================
# 8. MAIN WEBCAM LOOP
# ==========================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("Camera frame could not be read.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(rgb)

        display_digit = "হাত দেখান"
        confidence_text = "Confidence: --"
        stability_text = "Stability: 0/15"

        # ==================================
        # 9. HAND DETECTION
        # ==================================

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

                # Model prediction
                probabilities = model.predict_proba(
                    features
                )[0]

                best_index = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[best_index]
                )

                prediction = str(
                    model.classes_[best_index]
                )

                sign_label = prediction.split("__")[-1]

                # Confidence filter
                if confidence >= CONFIDENCE_THRESHOLD:
                    predictions.append(
                        (sign_label, confidence)
                    )
                else:
                    predictions.append(None)

                valid_predictions = [
                    item for item in predictions
                    if item is not None
                ]

                # Stable prediction
                if valid_predictions:

                    counts = Counter(
                        item[0]
                        for item in valid_predictions
                    )

                    stable_sign, votes = (
                        counts.most_common(1)[0]
                    )

                    matching_confidences = [
                        item[1]
                        for item in valid_predictions
                        if item[0] == stable_sign
                    ]

                    average_confidence = (
                        sum(matching_confidences)
                        / len(matching_confidences)
                    )

                    stability_text = (
                        f"Stability: {votes}/{HISTORY_SIZE}"
                    )

                    confidence_text = (
                        f"Confidence: "
                        f"{average_confidence * 100:.1f}%"
                    )

                    if votes >= REQUIRED_VOTES:

                        display_digit = BANGLA_DIGITS.get(
                            stable_sign,
                            "অজানা"
                        )

                    else:
                        display_digit = "অপেক্ষা করুন"

                else:
                    display_digit = "আবার চেষ্টা করুন"

            except (ValueError, KeyError) as error:

                predictions.clear()
                display_digit = "হাত ঠিকভাবে দেখান"
                print("Prediction error:", error)

        else:
            predictions.clear()

        # ==================================
        # 10. DISPLAY INFORMATION
        # ==================================

        height, width = frame.shape[:2]

        # Compact panel - does not cover full camera
        cv2.rectangle(
            frame,
            (10, 10),
            (520, 165),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "BANGLA DIGIT RECOGNITION",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            stability_text,
            (25, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            confidence_text,
            (25, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        # Bengali prediction
        frame = draw_bengali_text(
            frame,
            f"Prediction: {display_digit}",
            (25, 120),
            small_bengali_font,
            (0, 255, 0)
        )

        # Large Bengali digit
        if display_digit in BANGLA_DIGITS.values():

            frame = draw_bengali_text(
                frame,
                display_digit,
                (width - 150, 40),
                bengali_font,
                (0, 255, 0)
            )

        # Keyboard instructions
        cv2.putText(
            frame,
            "Q: Quit     R: Reset",
            (25, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ==================================
        # 11. FULL SCREEN DISPLAY
        # ==================================

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):
            predictions.clear()
            print("Prediction history reset.")

finally:

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("Webcam closed.")
