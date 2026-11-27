
import cv2
import mediapipe as mp
import numpy as np
import joblib
import json
import os
from collections import Counter, deque
from PIL import Image, ImageDraw, ImageFont

# =====================================================
# PATHS
# =====================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    ROOT, "models", "bdsl126_landmark_model.joblib"
)

MAPPING_PATH = os.path.join(
    ROOT, "datasets", "processed",
    "bdsl126_bengali_mapping.json"
)

FONT_PATH = (
    r"C:\Users\KHAN GADGET\AppData\Local\Microsoft"
    r"\Windows\Fonts\NotoSansBengali-VariableFont_wdth,wght.ttf"
)

# =====================================================
# FAST SETTINGS
# =====================================================

CAMERA_INDEX = 0
MAX_HANDS = 2

HISTORY_SIZE = 3
STABLE_COUNT = 1

MIN_CONFIDENCE = 0.10

DEBUG_TOP5 = True
DEBUG_INTERVAL = 30

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading BdSL-126 model...")

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
feature_columns = model_data["feature_columns"]
class_names = model_data["classes"]

print("Model:", type(model).__name__)
print("Expected features:", model.n_features_in_)
print("Feature columns:", len(feature_columns))
print("Classes:", len(class_names))

if model.n_features_in_ != len(feature_columns):
    raise ValueError("Model feature count mismatch!")

# =====================================================
# LOAD BENGALI MAPPING
# =====================================================

mapping = {}

if os.path.exists(MAPPING_PATH):

    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    print("Bengali mapping loaded.")

def get_bengali_name(label):
    return str(mapping.get(str(label), str(label)))

# =====================================================
# FONT
# =====================================================

try:
    bengali_font = ImageFont.truetype(
        FONT_PATH, 30
    )
except Exception:
    bengali_font = ImageFont.load_default()
    print("Bengali font could not be loaded.")

# =====================================================
# MEDIAPIPE
# =====================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    model_complexity=0,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

# =====================================================
# FEATURE EXTRACTION
# 1 hand_count + 132 coordinates = 133
# =====================================================

def extract_features(result):

    detected = []

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            points = np.array(
                [
                    [lm.x, lm.y, lm.z]
                    for lm in hand_landmarks.landmark
                ],
                dtype=np.float32
            )

            if points.shape != (21, 3):
                continue

            wrist_position = points[0].copy()

            normalized_points = points - wrist_position

            distances = np.linalg.norm(
                normalized_points,
                axis=1
            )

            scale = np.max(distances)

            if scale < 1e-6:
                continue

            normalized_points = normalized_points / scale

            hand_features = np.concatenate([
                wrist_position,
                normalized_points.flatten()
            ])

            detected.append({
                "center_x": float(wrist_position[0]),
                "features": hand_features
            })

    if not detected:
        return None

    # Left-to-right ordering
    detected.sort(
        key=lambda item: item["center_x"]
    )

    hand_count = min(
        len(detected),
        MAX_HANDS
    )

    values = []

    for item in detected[:MAX_HANDS]:
        values.extend(
            item["features"].tolist()
        )

    # Pad second hand if only one hand detected
    while len(values) < 132:
        values.extend([0.0] * 66)

    final_features = [hand_count] + values

    if len(final_features) != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: {len(final_features)}"
        )

    return np.asarray(
        final_features,
        dtype=np.float32
    ).reshape(1, -1)

# =====================================================
# FAST PREDICTION
# =====================================================

def predict_sign(features):

    probabilities = model.predict_proba(features)[0]

    best_index = int(np.argmax(probabilities))

    label = str(model.classes_[best_index])

    confidence = float(probabilities[best_index])

    # Top 5 debug information
    if DEBUG_TOP5:

        top_indices = np.argsort(
            probabilities
        )[-5:][::-1]

        print("\n--- TOP 5 ---")

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            name = str(model.classes_[index])

            score = probabilities[index] * 100

            print(
                f"{rank}. {name} "
                f"({get_bengali_name(name)}) "
                f"{score:.1f}%"
            )

    return label, confidence

# =====================================================
# BENGALI TEXT
# =====================================================

def draw_bengali_text(
    frame,
    text,
    position,
    font
):

    try:

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        draw = ImageDraw.Draw(image)

        draw.text(
            position,
            str(text),
            font=font,
            fill=(255, 255, 255)
        )

        return cv2.cvtColor(
            np.array(image),
            cv2.COLOR_RGB2BGR
        )

    except Exception:

        return frame

# =====================================================
# STATE
# =====================================================

history = deque(
    maxlen=HISTORY_SIZE
)

last_prediction = ""
last_confidence = 0.0

error_message = ""

frame_counter = 0

# =====================================================
# CAMERA
# =====================================================

print("\nOpening camera...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    cap.release()

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

if not cap.isOpened():

    hands.close()

    raise RuntimeError(
        "Camera could not open."
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

# Reduce buffering where supported
cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

print("Camera started.")
print("Q = Quit | R = Reset")

# =====================================================
# MAIN LOOP
# =====================================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.flip(
            frame,
            1
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(
            rgb_frame
        )

        error_message = ""

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            try:

                features = extract_features(
                    result
                )

                if features is not None:

                    label, confidence = predict_sign(
                        features
                    )

                    frame_counter += 1

                    if confidence >= MIN_CONFIDENCE:

                        history.append(label)

                        # Fast majority voting
                        stable_label = Counter(
                            history
                        ).most_common(1)[0][0]

                        last_prediction = stable_label

                        last_confidence = confidence

            except Exception as e:

                error_message = str(e)

                print(
                    "Prediction error:",
                    error_message
                )

        else:

            history.clear()

        # =================================================
        # DISPLAY
        # =================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (640, 115),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "GLOBAL SIGN LANGUAGE TRANSLATOR",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        if last_prediction:

            bengali_word = get_bengali_name(
                last_prediction
            )

            frame = draw_bengali_text(
                frame,
                "Prediction: " + bengali_word,
                (12, 38),
                bengali_font
            )

            cv2.putText(
                frame,
                f"Class: {last_prediction}",
                (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 0),
                1
            )

            cv2.putText(
                frame,
                f"Score: {last_confidence * 100:.1f}%",
                (440, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 0),
                1
            )

        else:

            cv2.putText(
                frame,
                "Show a static sign with your hand",
                (12, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                1
            )

        if error_message:

            cv2.putText(
                frame,
                "Prediction error - check terminal",
                (12, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )

        cv2.putText(
            frame,
            "Q: Quit     R: Reset",
            (12, 455),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "BdSL-126 FAST Translator",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("r"):

            history.clear()

            last_prediction = ""

            last_confidence = 0.0

            print("Prediction reset.")

except KeyboardInterrupt:

    print("Stopped by user.")

finally:

    cap.release()

    hands.close()

    cv2.destroyAllWindows()

    print("Camera closed.")
