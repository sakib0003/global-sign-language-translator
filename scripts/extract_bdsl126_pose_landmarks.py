
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm

# ==========================================
# PATH SETTINGS
# ==========================================

BASE = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE / "datasets/processed/bdsl126_classify_clean_v2"
OUTPUT_DIR = BASE / "datasets/processed"

INPUTS = {
    "train": OUTPUT_DIR / "bdsl126_landmarks_clean_train.csv",
    "val": OUTPUT_DIR / "bdsl126_landmarks_clean_val.csv",
}

OUTPUTS = {
    "train": OUTPUT_DIR / "bdsl126_pose_landmarks_train.csv",
    "val": OUTPUT_DIR / "bdsl126_pose_landmarks_val.csv",
}

# ==========================================
# MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(image, hands_detector, pose_detector):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hand_result = hands_detector.process(rgb)
    pose_result = pose_detector.process(rgb)

    hand_features = []
    detected_hands = []

    # --------------------------------------
    # HAND LANDMARKS
    # --------------------------------------

    if hand_result.multi_hand_landmarks:

        detected_hands = list(hand_result.multi_hand_landmarks)

        # Sort hands from left to right
        detected_hands.sort(
            key=lambda hand: hand.landmark[0].x
        )

    hand_count = min(len(detected_hands), 2)

    for hand in detected_hands[:2]:

        points = np.array(
            [[p.x, p.y, p.z] for p in hand.landmark],
            dtype=np.float32
        )

        wrist = points[0].copy()

        normalized = points - wrist

        scale = np.max(
            np.linalg.norm(normalized, axis=1)
        )

        if scale > 1e-6:
            normalized = normalized / scale

        values = np.concatenate([
            wrist,
            normalized.flatten()
        ])

        hand_features.extend(values.tolist())

    # Pad missing hand
    while len(hand_features) < 132:
        hand_features.extend([0.0] * 66)

    # --------------------------------------
    # BODY POSE LANDMARKS
    # --------------------------------------

    pose_features = []
    pose_detected = 0

    if pose_result.pose_landmarks:

        pose_detected = 1

        for point in pose_result.pose_landmarks.landmark:

            pose_features.extend([
                point.x,
                point.y,
                point.z,
                point.visibility
            ])

    else:

        pose_features = [0.0] * (33 * 4)

    # --------------------------------------
    # FINAL FEATURE VECTOR
    # --------------------------------------

    features = (
        [hand_count, pose_detected]
        + hand_features
        + pose_features
    )

    return features


# ==========================================
# MAIN EXTRACTION
# ==========================================

for split in ["train", "val"]:

    print(f"\n===== {split.upper()} =====")

    input_csv = INPUTS[split]
    output_csv = OUTPUTS[split]

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}"
        )

    df = pd.read_csv(input_csv)

    rows = []

    missing_images = 0
    pose_detected_count = 0
    hand_detected_count = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5
    ) as hands_detector, mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose_detector:

        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing {split}"
        ):

            source = Path(str(row["source"]))

            # IMPORTANT:
            # source already contains train/ or val/
            # Do not add split a second time.

            if source.is_absolute():
                image_path = source
            else:
                image_path = DATASET_DIR / source

            if not image_path.exists():

                missing_images += 1

                if missing_images <= 5:
                    print(
                        "\nMissing image:",
                        image_path
                    )

                continue

            image = cv2.imread(str(image_path))

            if image is None:

                missing_images += 1
                continue

            features = extract_features(
                image,
                hands_detector,
                pose_detector
            )

            hand_count = features[0]
            pose_detected = features[1]

            if hand_count > 0:
                hand_detected_count += 1

            if pose_detected == 1:
                pose_detected_count += 1

            record = {
                "label": row["label"],
                "split": split,
                "source": str(row["source"]),
                "hand_count": hand_count,
                "pose_detected": pose_detected,
            }

            for i, value in enumerate(features[2:]):
                record[f"feature_{i:03d}"] = value

            rows.append(record)

    # --------------------------------------
    # SAVE CSV
    # --------------------------------------

    result = pd.DataFrame(rows)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        output_csv,
        index=False
    )

    # --------------------------------------
    # REPORT
    # --------------------------------------

    print("\nExtraction report")
    print("-------------------------")

    print("Input samples:", len(df))
    print("Saved samples:", len(result))
    print("Missing images:", missing_images)

    print("Hands detected:", hand_detected_count)

    print("Pose detected:", pose_detected_count)

    print(
        "Pose not detected:",
        len(result) - pose_detected_count
    )

    print("Feature count:", len(result.columns) - 3)

    print("Output:", output_csv)

    if len(result) == 0:
        raise RuntimeError(
            f"No samples extracted for {split}. "
            "Check dataset paths."
        )

print("\n================================")
print("POSE LANDMARK EXTRACTION COMPLETE")
print("================================")
