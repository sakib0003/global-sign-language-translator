
import cv2
import mediapipe as mp
import re
import random

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

VAL_DIR = (
    ROOT
    / "datasets"
    / "processed"
    / "bdsl126_classify_clean_v2"
    / "val"
)

SAMPLE_SIZE = 300
RANDOM_SEED = 42

CONFIDENCE_LEVELS = [0.5, 0.3, 0.1]

FRAME_PATTERN = re.compile(
    r"_frame\d+_jpg\.rf\.",
    re.IGNORECASE
)


# ==========================================
# FIND INDIVIDUAL IMAGES
# ==========================================

def find_images():

    records = []

    extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    for class_folder in sorted(VAL_DIR.iterdir()):

        if not class_folder.is_dir():
            continue

        for image_path in class_folder.iterdir():

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in extensions:
                continue

            if FRAME_PATTERN.search(image_path.name):
                continue

            records.append(image_path)

    return records


# ==========================================
# RUN CONFIDENCE TEST
# ==========================================

def main():

    if not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Validation folder not found: {VAL_DIR}"
        )

    images = find_images()

    random.seed(RANDOM_SEED)
    random.shuffle(images)

    selected = images[:min(SAMPLE_SIZE, len(images))]

    results = defaultdict(int)

    failed_images = 0
    valid_images = 0

    print("\n" + "=" * 55)
    print("BdSL126 HAND CONFIDENCE TEST")
    print("=" * 55)

    print("Available images:", len(images))
    print("Selected images:", len(selected))
    print("Confidence levels:", CONFIDENCE_LEVELS)

    mp_hands = mp.solutions.hands

    for confidence in CONFIDENCE_LEVELS:

        print(f"\nTesting confidence: {confidence}")

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=confidence
        ) as hands_model:

            for image_path in tqdm(
                selected,
                desc=f"Confidence {confidence}",
                unit="image"
            ):

                image = cv2.imread(str(image_path))

                if image is None:
                    continue

                # Test original and 90-degree rotation
                versions = {
                    "original": image,
                    "rotated_90": cv2.rotate(
                        image,
                        cv2.ROTATE_90_CLOCKWISE
                    )
                }

                for version_name, test_image in versions.items():

                    rgb = cv2.cvtColor(
                        test_image,
                        cv2.COLOR_BGR2RGB
                    )

                    result = hands_model.process(rgb)

                    key = (confidence, version_name)

                    if result.multi_hand_landmarks:
                        results[key] += 1

        print(f"Confidence {confidence} completed.")

    # ======================================
    # FINAL REPORT
    # ======================================

    print("\n")
    print("=" * 55)
    print("CONFIDENCE TEST RESULTS")
    print("=" * 55)

    print("Images tested:", len(selected))

    for confidence in CONFIDENCE_LEVELS:

        for version_name in ["original", "rotated_90"]:

            detected = results[(confidence, version_name)]

            percentage = (
                detected / len(selected) * 100
                if selected else 0
            )

            print(
                f"Confidence={confidence:.1f} | "
                f"{version_name:12} | "
                f"Detected={detected:3} | "
                f"{percentage:.2f}%"
            )

    print("=" * 55)
    print("CONFIDENCE TEST COMPLETED!")


if __name__ == "__main__":
    main()
