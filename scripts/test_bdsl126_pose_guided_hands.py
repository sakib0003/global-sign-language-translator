
import cv2
import mediapipe as mp
import random
import re

from pathlib import Path
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

HAND_CONFIDENCE = 0.1
POSE_CONFIDENCE = 0.5

# Crop size around each wrist
CROP_WIDTH_RATIO = 0.30
CROP_HEIGHT_RATIO = 0.40

FRAME_PATTERN = re.compile(
    r"_frame\d+_jpg\.rf\.",
    re.IGNORECASE
)


# ==========================================
# FIND INDIVIDUAL IMAGES
# ==========================================

def find_images():

    images = []

    extensions = (
        ".jpg", ".jpeg", ".png",
        ".bmp", ".webp"
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

            images.append(image_path)

    return images


# ==========================================
# CROP AROUND WRIST
# ==========================================

def crop_around_wrist(image, landmark):

    height, width = image.shape[:2]

    center_x = int(landmark.x * width)
    center_y = int(landmark.y * height)

    crop_width = int(width * CROP_WIDTH_RATIO)
    crop_height = int(height * CROP_HEIGHT_RATIO)

    x1 = max(0, center_x - crop_width // 2)
    y1 = max(0, center_y - crop_height // 2)

    x2 = min(width, center_x + crop_width // 2)
    y2 = min(height, center_y + crop_height // 2)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Enlarge crop for hand detection
    crop = cv2.resize(
        crop,
        (256, 256),
        interpolation=cv2.INTER_CUBIC
    )

    return crop


# ==========================================
# HAND DETECTION
# ==========================================

def detect_hand(image, hands_model):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    result = hands_model.process(rgb)

    return bool(result.multi_hand_landmarks)


# ==========================================
# MAIN TEST
# ==========================================

def main():

    if not VAL_DIR.exists():
        raise FileNotFoundError(VAL_DIR)

    images = find_images()

    random.seed(RANDOM_SEED)
    random.shuffle(images)

    selected = images[:min(SAMPLE_SIZE, len(images))]

    print("\n" + "=" * 55)
    print("POSE-GUIDED HAND DETECTION TEST")
    print("=" * 55)

    print("Available images:", len(images))
    print("Selected images:", len(selected))

    # Statistics
    full_image_detected = 0
    pose_detected_count = 0
    wrist_crops_tested = 0
    wrist_crop_detected = 0
    guided_image_detected = 0

    failed_images = 0

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        min_detection_confidence=POSE_CONFIDENCE
    ) as pose_model:

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=HAND_CONFIDENCE
        ) as hands_model:

            for image_path in tqdm(
                selected,
                desc="Pose-guided testing",
                unit="image"
            ):

                image = cv2.imread(str(image_path))

                if image is None:
                    failed_images += 1
                    continue

                # ----------------------------------
                # 1. FULL IMAGE HAND DETECTION
                # ----------------------------------

                full_detected = detect_hand(
                    image,
                    hands_model
                )

                if full_detected:
                    full_image_detected += 1

                # ----------------------------------
                # 2. POSE DETECTION
                # ----------------------------------

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                pose_result = pose_model.process(rgb)

                if not pose_result.pose_landmarks:
                    continue

                pose_detected_count += 1

                landmarks = pose_result.pose_landmarks.landmark

                # MediaPipe Pose:
                # 15 = Left wrist
                # 16 = Right wrist

                wrist_indices = [15, 16]

                image_guided_detected = False

                # ----------------------------------
                # 3. CROP WRIST REGIONS
                # ----------------------------------

                for wrist_index in wrist_indices:

                    wrist = landmarks[wrist_index]

                    # Skip wrist outside image
                    if not (
                        0 <= wrist.x <= 1
                        and 0 <= wrist.y <= 1
                    ):
                        continue

                    crop = crop_around_wrist(
                        image,
                        wrist
                    )

                    if crop is None:
                        continue

                    wrist_crops_tested += 1

                    # ----------------------------------
                    # 4. HAND DETECTION ON CROP
                    # ----------------------------------

                    crop_detected = detect_hand(
                        crop,
                        hands_model
                    )

                    if crop_detected:

                        wrist_crop_detected += 1
                        image_guided_detected = True

                if image_guided_detected:
                    guided_image_detected += 1

    # ==========================================
    # FINAL REPORT
    # ==========================================

    total = len(selected)

    print("\n")
    print("=" * 55)
    print("POSE-GUIDED TEST RESULTS")
    print("=" * 55)

    print("Images selected:", total)
    print("Failed images:", failed_images)

    print(
        "Full image hand detections:",
        full_image_detected
    )

    print(
        "Pose detected images:",
        pose_detected_count
    )

    print(
        "Wrist crops tested:",
        wrist_crops_tested
    )

    print(
        "Wrist crops with hands:",
        wrist_crop_detected
    )

    print(
        "Images with guided hand detection:",
        guided_image_detected
    )

    if total:

        print(
            "Full image detection rate:",
            f"{full_image_detected / total * 100:.2f}%"
        )

        print(
            "Guided image detection rate:",
            f"{guided_image_detected / total * 100:.2f}%"
        )

    print("=" * 55)
    print("POSE-GUIDED TEST COMPLETED!")


if __name__ == "__main__":
    main()
