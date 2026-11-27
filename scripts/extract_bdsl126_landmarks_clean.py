
import csv
import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path
from landmark_utils import normalize_landmarks


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    BASE_DIR / "datasets" / "processed"
    / "bdsl126_classify_clean_v2"
)

OUTPUT_DIR = (
    BASE_DIR / "datasets" / "processed"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# SETTINGS
# ==========================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MAX_HANDS = 2
SAVE_EVERY = 100


# ==========================================
# OUTPUT FILES - NEW CSV
# ==========================================

OUTPUT_FILES = {
    "train": OUTPUT_DIR / "bdsl126_landmarks_clean_train.csv",
    "val": OUTPUT_DIR / "bdsl126_landmarks_clean_val.csv"
}


# ==========================================
# FEATURE COLUMNS
# ==========================================

FEATURE_COLUMNS = []

for hand_number in range(2):

    prefix = f"hand{hand_number + 1}"

    FEATURE_COLUMNS.extend([
        f"{prefix}_wrist_x",
        f"{prefix}_wrist_y",
        f"{prefix}_wrist_z"
    ])

    for point in range(21):

        FEATURE_COLUMNS.extend([
            f"{prefix}_p{point}_x",
            f"{prefix}_p{point}_y",
            f"{prefix}_p{point}_z"
        ])


CSV_COLUMNS = [
    "label",
    "split",
    "source",
    "hand_count"
] + FEATURE_COLUMNS


# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands


# ==========================================
# EXTRACT HAND FEATURES
# ==========================================

def extract_features(hand_result):

    if not hand_result.multi_hand_landmarks:
        return None

    detected_hands = []

    for hand in hand_result.multi_hand_landmarks:

        points = np.array(
            [[p.x, p.y, p.z] for p in hand.landmark],
            dtype=np.float32
        )

        if points.shape != (21, 3):
            continue

        wrist_position = points[0].copy()

        normalized = normalize_landmarks(
            points.flatten()
        )

        center_x = float(np.mean(points[:, 0]))

        hand_features = np.concatenate([
            wrist_position,
            normalized
        ])

        detected_hands.append(
            (center_x, hand_features)
        )

    if not detected_hands:
        return None

    detected_hands.sort(key=lambda item: item[0])

    features = []

    for _, hand_features in detected_hands[:MAX_HANDS]:
        features.extend(hand_features.tolist())

    hand_count = min(len(detected_hands), MAX_HANDS)

    while len(features) < 132:
        features.extend([0.0] * 66)

    return hand_count, features


# ==========================================
# PROCESS ONE SPLIT
# ==========================================

def process_split(split_name):

    image_root = DATASET_DIR / split_name
    output_csv = OUTPUT_FILES[split_name]

    if not image_root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {image_root}"
        )

    # Never append to an old CSV
    if output_csv.exists():
        raise FileExistsError(
            f"Output already exists: {output_csv}\n"
            "পুরোনো ফাইল নিরাপদ রাখতে নতুন নাম ব্যবহার করো।"
        )

    total_images = sum(
        1 for p in image_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print("\n================================")
    print("Processing:", split_name)
    print("Total images:", total_images)
    print("================================")

    processed = 0
    skipped = 0
    saved = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_HANDS,
        model_complexity=0,
        min_detection_confidence=0.5
    ) as hands:

        with open(
            output_csv,
            "w",
            encoding="utf-8",
            newline=""
        ) as file:

            writer = csv.writer(file)
            writer.writerow(CSV_COLUMNS)

            for image_path in image_root.rglob("*"):

                if (
                    not image_path.is_file()
                    or image_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                processed += 1

                image = cv2.imread(str(image_path))

                if image is None:
                    skipped += 1
                    continue

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                result = hands.process(rgb)

                extracted = extract_features(result)

                if extracted is None:
                    skipped += 1
                    continue

                hand_count, features = extracted

                class_name = image_path.parent.name

                source = str(
                    image_path.relative_to(DATASET_DIR)
                ).replace("\\", "/")

                row = [
                    class_name,
                    split_name,
                    source,
                    hand_count
                ] + features

                writer.writerow(row)
                saved += 1

                if processed % SAVE_EVERY == 0:
                    file.flush()

                    print(
                        f"{split_name}: "
                        f"{saved} saved | "
                        f"{processed}/{total_images} processed | "
                        f"{skipped} skipped"
                    )

            file.flush()

    print("\nFinished:", split_name)
    print("Saved:", saved)
    print("Skipped:", skipped)
    print("Output:", output_csv)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("BdSL-126 Clean Static Landmark Extraction")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {DATASET_DIR}"
        )

    process_split("train")
    process_split("val")

    print("\nAll clean extraction completed!")
