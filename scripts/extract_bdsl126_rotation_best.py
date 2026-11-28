
import csv
import re
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm


# ==================================================
# 1. PATHS
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

FRAME_PATTERN = re.compile(
    r"_frame\d+_jpg\.rf\.",
    re.IGNORECASE
)

ROTATIONS = [0, 90, 180, 270]

# ==================================================
# 2. FEATURES
# ==================================================

HAND_FEATURES = 2 * 21 * 3
POSE_FEATURES = 33 * 4
EXTRA_FEATURES = 2

TOTAL_FEATURES = (
    HAND_FEATURES + POSE_FEATURES + EXTRA_FEATURES
)

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


# ==================================================
# 3. ROTATE IMAGE
# ==================================================

def rotate_image(image, angle):

    if angle == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    if angle == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    if angle == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    return image


# ==================================================
# 4. EXTRACT FEATURES
# ==================================================

def extract_features(image, hands_model, pose_model):

    hand_features = {
        "Left": [0.0] * 63,
        "Right": [0.0] * 63
    }

    hand_count = 0

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

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

            if label not in ("Left", "Right"):

                if hand_features["Left"] == [0.0] * 63:
                    label = "Left"
                else:
                    label = "Right"

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

    pose_features = [0.0] * POSE_FEATURES
    pose_detected = 0
    pose_visibility = 0.0

    pose_result = pose_model.process(rgb)

    if pose_result.pose_landmarks:

        pose_detected = 1

        visibility_values = []
        pose_features = []

        for landmark in pose_result.pose_landmarks.landmark:

            visibility = float(landmark.visibility)

            visibility_values.append(visibility)

            pose_features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                visibility
            ])

        if visibility_values:
            pose_visibility = (
                sum(visibility_values) /
                len(visibility_values)
            )

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
            f"Feature mismatch: {len(features)}"
        )

    return (
        features,
        hand_count,
        pose_detected,
        pose_visibility
    )


# ==================================================
# 5. SELECT BEST ROTATION
# ==================================================

def select_best_rotation(image, hands_model, pose_model):

    results = []

    for angle in ROTATIONS:

        rotated = rotate_image(image, angle)

        features, hand_count, pose_flag, visibility = (
            extract_features(
                rotated,
                hands_model,
                pose_model
            )
        )

        results.append({
            "angle": angle,
            "features": features,
            "hand_count": hand_count,
            "pose_flag": pose_flag,
            "visibility": visibility
        })

    pose_results = [
        item for item in results
        if item["pose_flag"] == 1
    ]

    if not pose_results:

        # No pose in any rotation: fallback to 0°
        return results[0]

    # Highest pose visibility first.
    # If visibility ties, prefer more detected hands.
    # Remaining ties prefer 90°, then 270°, 180°, 0°.
    tie_priority = {
        90: 3,
        270: 2,
        180: 1,
        0: 0
    }

    best = max(
        pose_results,
        key=lambda item: (
            item["visibility"],
            item["hand_count"],
            tie_priority[item["angle"]]
        )
    )

    return best


# ==================================================
# 6. FIND INDIVIDUAL IMAGES
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

            if FRAME_PATTERN.search(image_path.name):
                continue

            image_records.append(
                (label, image_path)
            )

    return class_folders, image_records


# ==================================================
# 7. PROCESS SPLIT
# ==================================================

def process_split(split):

    split_dir = DATASET_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(split_dir)

    class_folders, image_records = (
        find_individual_images(split_dir)
    )

    output_file = (
        OUTPUT_DIR
        / f"bdsl126_pose_rotation_best_{split}.csv"
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
            "pose_detected",
            "selected_rotation",
            "pose_visibility"
        ]
    )

    saved = 0
    failed = 0
    hands_detected = 0
    pose_detected_count = 0

    rotation_counts = {
        angle: 0 for angle in ROTATIONS
    }

    print("\n" + "=" * 55)
    print(f"PROCESSING: {split.upper()}")
    print("=" * 55)

    print("Classes:", len(class_folders))
    print("Images:", len(image_records))
    print("Features:", TOTAL_FEATURES)
    print("Output:", output_file)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.1
    ) as hands_model:

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.3
        ) as pose_model:

            with open(
                output_file,
                "w",
                newline="",
                encoding="utf-8-sig"
            ) as csv_file:

                writer = csv.writer(csv_file)
                writer.writerow(columns)

                for label, image_path in tqdm(
                    image_records,
                    desc=f"Extracting {split}",
                    unit="image"
                ):

                    relative_path = str(
                        image_path.relative_to(DATASET_DIR)
                    )

                    image = cv2.imread(str(image_path))

                    if image is None:

                        failed += 1

                        writer.writerow([
                            label,
                            relative_path,
                            *([0.0] * TOTAL_FEATURES),
                            0,
                            0,
                            0,
                            0,
                            0.0
                        ])

                        continue

                    best = select_best_rotation(
                        image,
                        hands_model,
                        pose_model
                    )

                    writer.writerow([
                        label,
                        relative_path,
                        *best["features"],
                        1,
                        best["hand_count"],
                        best["pose_flag"],
                        best["angle"],
                        best["visibility"]
                    ])

                    saved += 1

                    rotation_counts[best["angle"]] += 1

                    if best["hand_count"] > 0:
                        hands_detected += 1

                    if best["pose_flag"] == 1:
                        pose_detected_count += 1

    print("\n" + "=" * 55)
    print(f"{split.upper()} EXTRACTION COMPLETE")
    print("=" * 55)

    print("Images found:", len(image_records))
    print("Records saved:", saved)
    print("Failed reads:", failed)
    print("Images with hands:", hands_detected)
    print("Images with pose:", pose_detected_count)

    print("\nSelected rotations:")

    for angle, count in rotation_counts.items():
        print(f"{angle} degrees: {count}")

    print("\nCSV saved:")
    print(output_file)


# ==================================================
# 8. MAIN
# ==================================================

if __name__ == "__main__":

    print("=" * 55)
    print("BdSL126 ROTATION-AWARE LANDMARK EXTRACTION")
    print("=" * 55)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    for split_name in SPLITS:
        process_split(split_name)

    print("\nALL EXTRACTION TASKS COMPLETED!")
