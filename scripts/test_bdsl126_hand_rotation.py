
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

FRAME_PATTERN = re.compile(
    r"_frame\d+_jpg\.rf\.",
    re.IGNORECASE
)

ROTATIONS = {
    "0_degree": None,
    "90_degree": cv2.ROTATE_90_CLOCKWISE,
    "180_degree": cv2.ROTATE_180,
    "270_degree": cv2.ROTATE_90_COUNTERCLOCKWISE
}


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
# MAIN TEST
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

    print("\n" + "=" * 55)
    print("BdSL126 HAND ROTATION TEST")
    print("=" * 55)

    print("Available images:", len(images))
    print("Images selected:", len(selected))
    print("Rotations:", len(ROTATIONS))

    results = defaultdict(int)
    valid_images = 0
    failed_images = 0

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5
    ) as hands_model:

        for image_path in tqdm(
            selected,
            desc="Testing rotations",
            unit="image"
        ):

            image = cv2.imread(str(image_path))

            if image is None:
                failed_images += 1
                continue

            valid_images += 1

            for rotation_name, rotation_code in ROTATIONS.items():

                if rotation_code is None:
                    rotated = image
                else:
                    rotated = cv2.rotate(
                        image,
                        rotation_code
                    )

                rgb = cv2.cvtColor(
                    rotated,
                    cv2.COLOR_BGR2RGB
                )

                result = hands_model.process(rgb)

                if result.multi_hand_landmarks:
                    results[rotation_name] += 1

    # ======================================
    # REPORT
    # ======================================

    print("\n")
    print("=" * 55)
    print("ROTATION TEST RESULTS")
    print("=" * 55)

    print("Valid images:", valid_images)
    print("Failed images:", failed_images)

    for rotation_name in ROTATIONS:

        detected = results[rotation_name]

        percentage = (
            detected / valid_images * 100
            if valid_images else 0
        )

        print(
            f"{rotation_name:15} "
            f"Hands detected: {detected:4} "
            f"({percentage:.2f}%)"
        )

    print("=" * 55)

    best_rotation = max(
        ROTATIONS,
        key=lambda name: results[name]
    )

    print("Best rotation:", best_rotation)
    print(
        "Best detections:",
        results[best_rotation]
    )

    print("\nROTATION TEST COMPLETED!")


if __name__ == "__main__":
    main()
