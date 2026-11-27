
import cv2
import joblib
import json
import numpy as np
import pandas as pd
import mediapipe as mp

from ultralytics import YOLO
from collections import deque, Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from landmark_utils import normalize_landmarks


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ALPHABET_PATH = BASE_DIR / "models" / "bdsl47_alphabet_svm.joblib"
DIGIT_PATH = BASE_DIR / "models" / "bdsl47_digits_normalized.joblib"

WORD_PATH = (
    BASE_DIR / "runs" / "classify" / "train-3"
    / "weights" / "best.pt"
)

MAPPING_PATH = (
    BASE_DIR / "datasets" / "processed"
    / "bdsl126_bengali_mapping.json"
)

FONT_PATH = (
    Path.home()
    / "AppData/Local/Microsoft/Windows/Fonts/"
    / "NotoSansBengali-VariableFont_wdth,wght.ttf"
)


# ==================================================
# 2. SETTINGS
# ==================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

HISTORY_SIZE = 15
REQUIRED_VOTES = 10

ALPHABET_THRESHOLD = 0.50
DIGIT_THRESHOLD = 0.40
WORD_THRESHOLD = 0.25

WORD_HISTORY_SIZE = 7
WORD_REQUIRED_VOTES = 4

WORD_INFERENCE_INTERVAL = 5

PANEL_WIDTH = 680
WORD_PANEL_HEIGHT = 330
NORMAL_PANEL_HEIGHT = 250


# ==================================================
# 3. BANGLA DIGITS
# ==================================================

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


# ==================================================
# 4. CHECK REQUIRED FILES
# ==================================================

required_files = [
    ALPHABET_PATH,
    DIGIT_PATH,
    WORD_PATH,
    MAPPING_PATH,
    FONT_PATH
]

for path in required_files:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")


# ==================================================
# 5. LOAD ALPHABET MODEL
# ==================================================

print("Loading Alphabet Model...")

alphabet_data = joblib.load(ALPHABET_PATH)

alphabet_model = alphabet_data["model"]
alphabet_features = alphabet_data["feature_columns"]


# ==================================================
# 6. LOAD DIGIT MODEL
# ==================================================

print("Loading Digit Model...")

digit_data = joblib.load(DIGIT_PATH)

if not digit_data.get("normalized", False):
    raise SystemExit("Digit model normalization check failed!")

digit_model = digit_data["model"]
digit_features = digit_data["feature_columns"]


# ==================================================
# 7. LOAD WORD MODEL
# ==================================================

print("Loading Word Model...")

word_model = YOLO(str(WORD_PATH))


# ==================================================
# 8. LOAD BENGALI MAPPING
# ==================================================

print("Loading Bengali Mapping...")

with open(MAPPING_PATH, "r", encoding="utf-8") as file:
    bengali_mapping = json.load(file)

if len(bengali_mapping) != len(word_model.names):
    raise SystemExit(
        "Word mapping and model class count do not match!"
    )

print("All models loaded successfully!")


# ==================================================
# 9. BENGALI FONTS
# ==================================================

bengali_font = ImageFont.truetype(
    str(FONT_PATH),
    55
)

small_font = ImageFont.truetype(
    str(FONT_PATH),
    23
)


# ==================================================
# 10. MEDIAPIPE SETUP
# ==================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==================================================
# 11. PREDICTION HISTORY
# ==================================================

alphabet_history = deque(maxlen=HISTORY_SIZE)
digit_history = deque(maxlen=HISTORY_SIZE)
word_history = deque(maxlen=WORD_HISTORY_SIZE)

mode = 1
word_frame_count = 0

current_prediction = "Show your hand"
stable_prediction = "Waiting..."
confidence = 0.0

last_word_prediction = "হাত দেখান"
last_word_confidence = 0.0

word_stable_prediction = "অপেক্ষা করুন"
word_top5 = []


# ==================================================
# 12. DRAW BENGALI TEXT
# ==================================================

def draw_bengali(
    frame,
    text,
    position,
    font,
    color=(0, 255, 0)
):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(rgb)

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


# ==================================================
# 13. STABLE PREDICTION
# ==================================================

def get_stable_prediction(history, required_votes):

    valid = [
        item for item in history
        if item is not None
    ]

    if not valid:
        return "Waiting...", 0

    label, votes = Counter(valid).most_common(1)[0]

    if votes >= required_votes:
        return label, votes

    return "Unstable", votes


# ==================================================
# 14. RESET FUNCTION
# ==================================================

def reset_predictions():

    global word_frame_count
    global current_prediction
    global stable_prediction
    global confidence

    global last_word_prediction
    global last_word_confidence

    global word_stable_prediction
    global word_top5

    alphabet_history.clear()
    digit_history.clear()
    word_history.clear()

    word_frame_count = 0

    current_prediction = "Show your hand"
    stable_prediction = "Waiting..."
    confidence = 0.0

    last_word_prediction = "হাত দেখান"
    last_word_confidence = 0.0

    word_stable_prediction = "অপেক্ষা করুন"
    word_top5 = []


# ==================================================
# 15. CAMERA SETUP
# ==================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    camera.release()
    camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():

    hands.close()
    raise SystemExit("Webcam could not be opened!")

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

WINDOW_NAME = "Global Sign Language Translator"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


# ==================================================
# 16. START MESSAGE
# ==================================================

print("\n===================================")
print("GLOBAL SIGN LANGUAGE TRANSLATOR")
print("===================================")

print("1 = Alphabet Mode")
print("2 = Digit Mode")
print("3 = Word Mode")

print("R = Reset")
print("Q = Quit")

print("===================================")


# ==================================================
# 17. MAIN LOOP
# ==================================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame error!")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Clean frame for Word Model
        # This frame has no drawn landmarks or display panel
        clean_frame = frame.copy()

        height, width = frame.shape[:2]

        # ==========================================
        # MEDIAPIPE HAND DETECTION
        # ==========================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        hand_result = hands.process(rgb)

        # Draw landmarks in every mode
        if hand_result.multi_hand_landmarks:

            for hand_landmarks in hand_result.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,

                    mp_draw.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=2,
                        circle_radius=4
                    ),

                    mp_draw.DrawingSpec(
                        color=(255, 0, 255),
                        thickness=2
                    )
                )

        # ==========================================
        # ALPHABET MODE
        # ==========================================

        if mode == 1:

            current_prediction = "Show your hand"
            confidence = 0.0

            if hand_result.multi_hand_landmarks:

                hand = hand_result.multi_hand_landmarks[0]

                values = []

                for point in hand.landmark:

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

                    probabilities = alphabet_model.predict_proba(
                        features
                    )[0]

                    best_index = int(
                        np.argmax(probabilities)
                    )

                    confidence = float(
                        probabilities[best_index]
                    )

                    raw_label = alphabet_model.classes_[best_index]

                    current_prediction = str(
                        raw_label
                    ).split("__")[-1]

                    if confidence >= ALPHABET_THRESHOLD:
                        alphabet_history.append(current_prediction)
                    else:
                        alphabet_history.append(None)

                    stable_prediction, _ = get_stable_prediction(
                        alphabet_history,
                        REQUIRED_VOTES
                    )

                except ValueError as error:
                    print("Alphabet error:", error)

            else:

                alphabet_history.clear()
                stable_prediction = "Waiting..."

        # ==========================================
        # DIGIT MODE
        # ==========================================

        elif mode == 2:

            current_prediction = "Show your hand"
            confidence = 0.0

            if hand_result.multi_hand_landmarks:

                hand = hand_result.multi_hand_landmarks[0]

                values = []

                for point in hand.landmark:

                    values.extend([
                        point.x,
                        point.y,
                        point.z
                    ])

                try:

                    normalized = normalize_landmarks(values)

                    features = pd.DataFrame(
                        [normalized],
                        columns=digit_features,
                        dtype=np.float32
                    )

                    probabilities = digit_model.predict_proba(
                        features
                    )[0]

                    best_index = int(
                        np.argmax(probabilities)
                    )

                    confidence = float(
                        probabilities[best_index]
                    )

                    raw_label = digit_model.classes_[best_index]

                    current_prediction = str(
                        raw_label
                    ).split("__")[-1]

                    if confidence >= DIGIT_THRESHOLD:
                        digit_history.append(current_prediction)
                    else:
                        digit_history.append(None)

                    stable_prediction, _ = get_stable_prediction(
                        digit_history,
                        REQUIRED_VOTES
                    )

                except ValueError as error:
                    print("Digit error:", error)

            else:

                digit_history.clear()
                stable_prediction = "Waiting..."

        # ==========================================
        # WORD MODE
        # FULL CAMERA IMAGE + TWO HAND LANDMARKS
        # ==========================================

        elif mode == 3:

            word_frame_count += 1

            if not hand_result.multi_hand_landmarks:

                last_word_prediction = "হাত দেখান"
                last_word_confidence = 0.0

                word_history.clear()
                word_stable_prediction = "অপেক্ষা করুন"
                word_top5 = []

            else:

                # Use the complete clean camera frame.
                # Do not crop hands for this model.
                # The model was trained on full images.

                if word_frame_count % WORD_INFERENCE_INTERVAL == 0:

                    results = word_model.predict(
                        source=clean_frame,
                        imgsz=160,
                        device="cpu",
                        verbose=False
                    )

                    probabilities = results[0].probs

                    if probabilities is not None:

                        best_id = int(
                            probabilities.top1
                        )

                        raw_word = word_model.names[best_id]

                        last_word_confidence = float(
                            probabilities.top1conf
                        )

                        bengali_word = bengali_mapping.get(
                            raw_word,
                            raw_word
                        )

                        if last_word_confidence >= WORD_THRESHOLD:

                            word_history.append(raw_word)

                            last_word_prediction = bengali_word

                        else:

                            word_history.append(None)

                            last_word_prediction = "নিশ্চিত নয়"

                        stable_raw_word, _ = get_stable_prediction(
                            word_history,
                            WORD_REQUIRED_VOTES
                        )

                        if stable_raw_word not in (
                            "Waiting...",
                            "Unstable"
                        ):

                            word_stable_prediction = bengali_mapping.get(
                                stable_raw_word,
                                stable_raw_word
                            )

                        else:

                            word_stable_prediction = "অপেক্ষা করুন"

                        # TOP 5
                        new_top5 = []

                        for class_id, score in zip(
                            probabilities.top5,
                            probabilities.top5conf
                        ):

                            raw_name = word_model.names[
                                int(class_id)
                            ]

                            bengali_name = bengali_mapping.get(
                                raw_name,
                                raw_name
                            )

                            new_top5.append(
                                (
                                    bengali_name,
                                    float(score)
                                )
                            )

                        word_top5 = new_top5

            current_prediction = last_word_prediction
            confidence = last_word_confidence
            stable_prediction = word_stable_prediction

        # ==========================================
        # DISPLAY PANEL
        # ==========================================

        panel_height = (
            WORD_PANEL_HEIGHT
            if mode == 3
            else NORMAL_PANEL_HEIGHT
        )

        cv2.rectangle(
            frame,
            (10, 10),
            (PANEL_WIDTH, panel_height),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "GLOBAL SIGN LANGUAGE TRANSLATOR",
            (28, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        mode_name = {
            1: "ALPHABET MODE",
            2: "DIGIT MODE",
            3: "WORD MODE"
        }

        cv2.putText(
            frame,
            "Mode: " + mode_name[mode],
            (28, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ==========================================
        # WORD DISPLAY
        # ==========================================

        if mode == 3:

            frame = draw_bengali(
                frame,
                "Prediction: " + str(current_prediction),
                (28, 115),
                small_font
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (28, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2
            )

            frame = draw_bengali(
                frame,
                "Stable: " + str(stable_prediction),
                (28, 195),
                small_font
            )

            cv2.putText(
                frame,
                "TOP 5:",
                (300, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            for i, (word, score) in enumerate(word_top5):

                frame = draw_bengali(
                    frame,
                    f"{i+1}. {word}: {score*100:.1f}%",
                    (300, 125 + i * 38),
                    small_font,
                    (0, 220, 255)
                )

        # ==========================================
        # ALPHABET / DIGIT DISPLAY
        # ==========================================

        else:

            cv2.putText(
                frame,
                "Prediction: " + str(current_prediction),
                (28, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (28, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            if mode == 2:

                bengali_digit = BANGLA_DIGITS.get(
                    stable_prediction
                )

                if bengali_digit:

                    frame = draw_bengali(
                        frame,
                        "Stable: " + bengali_digit,
                        (28, 205),
                        small_font
                    )

                    frame = draw_bengali(
                        frame,
                        bengali_digit,
                        (width - 150, 40),
                        bengali_font
                    )

                else:

                    cv2.putText(
                        frame,
                        "Stable: " + str(stable_prediction),
                        (28, 225),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

            else:

                cv2.putText(
                    frame,
                    "Stable: " + str(stable_prediction),
                    (28, 225),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

        # ==========================================
        # KEYBOARD INSTRUCTIONS
        # ==========================================

        cv2.putText(
            frame,
            "1: Alphabet   2: Digit   3: Word   R: Reset   Q: Quit",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # ==========================================
        # KEYBOARD CONTROLS
        # ==========================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("1"):

            mode = 1
            reset_predictions()
            print("Alphabet Mode activated")

        elif key == ord("2"):

            mode = 2
            reset_predictions()
            print("Digit Mode activated")

        elif key == ord("3"):

            mode = 3
            reset_predictions()
            print("Word Mode activated")

        elif key == ord("r"):

            reset_predictions()
            print("Prediction history reset!")


# ==================================================
# 18. CLEANUP
# ==================================================

finally:

    camera.release()
    hands.close()
    cv2.destroyAllWindows()

    print("Webcam closed.")
