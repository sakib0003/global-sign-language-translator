
import csv
import re
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "bdsl126_classify_clean_v2"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed"

SPLITS = ["train", "val"]

# Frame images বাদ দেওয়ার নিয়ম
FRAME_PATTERN = re.compile(
    r"_frame\d+_jpg\.rf\.",
    re.IGNORECASE
)

# ==================================================
# 2. FEATURE CONFIGURATION
# ==================================================

# Hand: 2 x 21 landmarks x 3 coordinates = 126
HAND_FEATURES = 2 * 21 * 3

# Pose: 33 landmarks x 4 values = 132
POSE_FEATURES = 33 * 4

# hand_count + pose_detected = 2
EXTRA_FEATURES = 2

TOTAL_FEATURES = (
    HAND_FEATURES
    + POSE_FEATURES
    + EXTRA_FEATURES
)

# Total = 260 features? 
# 126 + 132 + 2 = 260


# ==================================================
# 3. MEDIAPIPE INITIALIZATION
# ==================================================

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


# ==================================================
# 4. FEATURE EXTRACTION FUNCTION
# ==================================================

def extract_features(image, hands_model, pose_model):

    # Two hands: Left and Right
    hand_features = {
        "Left": [0.0] * 63,
        "Right": [0.0] * 63
    }

    hand_count = 0

    # Pose default values
    pose_features = [0.0] * POSE_FEATURES

    pose_detected = 0

    # Convert BGR to RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ------------------------------------------------
    # HAND LANDMARK DETECTION
    # ------------------------------------------------

    hand_result = hands_model.process(rgb)

    if hand_result.multi_hand_landmarks:

        handedness_list = (
            hand_result.multi_handedness or []
        )

        for index, hand_landmarks in enumerate(
            hand_result.multi_hand_landmarks
        ):

            label = None

            if index < len(handedness_list):

                classification = (
                    handedness_list[index].classification
                )

                if classification:
                    label = classification[0].label

            # Fallback if handedness is unavailable
            if label not in ("Left", "Right"):

                if hand_features["Left"] == [0.0] * 63:
                    label = "Left"
                else:
                    label = "Right"

            # Do not overwrite an already detected hand
            if hand_features[label] != [0.0] * 63:
                continue

            values = []

            for landmark in hand_landmarks.landmark:

                values.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            hand_features[label] = values
            hand_count += 1

    # ------------------------------------------------
    # POSE LANDMARK DETECTION
    # ------------------------------------------------

    pose_result = pose_model.process(rgb)

    if pose_result.pose_landmarks:

        pose_detected = 1

        pose_features = []

        for landmark in pose_result.pose_landmarks.landmark:

            pose_features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])

    # ------------------------------------------------
    # FINAL FEATURE VECTOR
    # ------------------------------------------------

    features = (
        hand_features["Left"]
        + hand_features["Right"]
        + pose_features
        + [
            float(hand_count),
            float(pose_detected)
        ]
    )

    if len(features) != TOTAL_FEATURES:
        raise ValueError(
            f"Feature length mismatch: "
            f"{len(features)} != {TOTAL_FEATURES}"
        )

    return features, hand_count, pose_detected


# ==================================================
# 5. FIND INDIVIDUAL IMAGES
# ==================================================

def find_individual_images(split_dir):

    class_folders = sorted([
        folder
        for folder in split_dir.iterdir()
        if folder.is_dir()
    ])

    image_records = []

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    for class_folder in class_folders:

        label = class_folder.name

        for image_path in class_folder.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in valid_extensions:
                continue

            # Skip frame images
            if FRAME_PATTERN.search(image_path.name):
                continue

            image_records.append(
                (label, image_path)
            )

    return class_folders, image_records


# ==================================================
# 6. PROCESS TRAIN / VALIDATION
# ==================================================

def process_split(split):

    split_dir = DATASET_DIR / split

    if not split_dir.exists():

        raise FileNotFoundError(
            f"Dataset folder not found: {split_dir}"
        )

    class_folders, image_records = (
        find_individual_images(split_dir)
    )

    output_file = (
        OUTPUT_DIR
        / f"bdsl126_pose_individual_direct_{split}.csv"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_columns = [
        f"feature_{i}"
        for i in range(TOTAL_FEATURES)
    ]

    columns = (
        ["label", "image_path"]
        + feature_columns
        + [
            "image_read_ok",
            "hand_count",
            "pose_detected"
        ]
    )

    # Statistics
    saved = 0
    failed = 0
    hands_detected = 0
    pose_detected_count = 0

    print("\n" + "=" * 55)
    print(f"PROCESSING: {split.upper()}")
    print("=" * 55)

    print("Classes:", len(class_folders))
    print("Individual images:", len(image_records))
    print("Features per image:", TOTAL_FEATURES)
    print("Output:", output_file)

    print("=" * 55)

    # ------------------------------------------------
    # MEDIAPIPE MODELS
    # ------------------------------------------------

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5
    ) as hands_model:

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.5
        ) as pose_model:

            with open(
                output_file,
                "w",
                newline="",
                encoding="utf-8-sig"
            ) as csv_file:

                writer = csv.writer(csv_file)

                writer.writerow(columns)

                # ------------------------------------
                # PROCESS EACH IMAGE
                # ------------------------------------

                for label, image_path in tqdm(
                    image_records,
                    desc=f"Extracting {split}",
                    unit="image"
                ):

                    relative_path = str(
                        image_path.relative_to(DATASET_DIR)
                    )

                    image = cv2.imread(
                        str(image_path)
                    )

                    # --------------------------------
                    # IMAGE READ FAILURE
                    # --------------------------------

                    if image is None:

                        failed += 1

                        writer.writerow([
                            label,
                            relative_path,
                            *([0.0] * TOTAL_FEATURES),
                            0,
                            0,
                            0
                        ])

                        continue

                    # --------------------------------
                    # EXTRACT FEATURES
                    # --------------------------------

                    features, hand_count, pose_flag = (
                        extract_features(
                            image,
                            hands_model,
                            pose_model
                        )
                    )

                    # --------------------------------
                    # SAVE CSV ROW
                    # --------------------------------

                    writer.writerow([
                        label,
                        relative_path,
                        *features,
                        1,
                        hand_count,
                        pose_flag
                    ])

                    saved += 1

                    if hand_count > 0:
                        hands_detected += 1

                    if pose_flag == 1:
                        pose_detected_count += 1

    # ==================================================
    # 7. PRINT FINAL REPORT
    # ==================================================

    print("\n" + "=" * 55)
    print(f"{split.upper()} EXTRACTION COMPLETE")
    print("=" * 55)

    print("Classes:", len(class_folders))
    print("Images found:", len(image_records))
    print("Records saved:", saved)
    print("Failed image reads:", failed)
    print("Images with hands:", hands_detected)
    print("Images with pose:", pose_detected_count)
    print("Images without detected hands:", saved - failed - hands_detected)
    print("Images without detected pose:", saved - failed - pose_detected_count)

    print("\nCSV saved:")
    print(output_file)

    print("=" * 55)


# ==================================================
# 8. MAIN
# ==================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 55)
    print("BdSL126 INDIVIDUAL IMAGE LANDMARK EXTRACTION")
    print("=" * 55)

    print("Dataset:", DATASET_DIR)
    print("Feature count:", TOTAL_FEATURES)

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            "BdSL126 dataset directory does not exist."
        )

    # Process Train and Validation
    for split_name in SPLITS:

        process_split(split_name)

    print("\n")
    print("=" * 55)
    print("ALL EXTRACTION TASKS COMPLETED!")
    print("=" * 55)
