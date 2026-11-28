
from pathlib import Path
import random
import cv2
import mediapipe as mp

ROOT = Path("datasets/processed/bdsl126_classify_clean_v2/val")
OUTPUT = Path("reports/bdsl126_landmark_visual_test")
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

files = [
    p for p in ROOT.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and "_frame" not in p.name
]

samples = random.sample(files, min(3, len(files)))

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.1
) as hands, mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    min_detection_confidence=0.5
) as pose:

    for index, path in enumerate(samples, 1):
        image = cv2.imread(str(path))

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hand_result = hands.process(rgb)
        pose_result = pose.process(rgb)

        hand_count = (
            len(hand_result.multi_hand_landmarks)
            if hand_result.multi_hand_landmarks else 0
        )

        pose_found = pose_result.pose_landmarks is not None

        if pose_found:
            mp_drawing.draw_landmarks(
                image,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        label = (
            f"{path.parent.name} | Hands: {hand_count} "
            f"| Pose: {pose_found}"
        )

        cv2.putText(
            image,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        output_path = OUTPUT / f"sample_{index}.jpg"
        cv2.imwrite(str(output_path), image)

        print(f"\nImage: {path}")
        print("Hands detected:", hand_count)
        print("Pose detected:", pose_found)
        print("Saved:", output_path)

print("\nDiagnostic images created successfully.")
