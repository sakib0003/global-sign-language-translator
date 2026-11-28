
from pathlib import Path
import cv2
import mediapipe as mp

ROOT = Path("datasets/processed/bdsl126_classify_clean_v2/val")
OUTPUT = Path("reports/bdsl126_rotation_test")
OUTPUT.mkdir(parents=True, exist_ok=True)

samples = [
    ROOT / "Canada/Canada__canada_0_9896_jpg.rf.ef04c511548509d68c000e89059ff5df.jpg",
    ROOT / "America/America__america_0_2088_jpg.rf.3776d4f7e116fc41d322d0328b8ef30f.jpg",
    ROOT / "Jailkhana/Jailkhana__jailkhana_0_4007_jpg.rf.8661cfdb0d7985b9f28e71554cdbaf41.jpg",
]

rotations = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

for path in samples:
    if not path.exists():
        print("Missing:", path)
        continue

    original = cv2.imread(str(path))
    print(f"\n========== {path.parent.name} ==========")

    best_score = -1
    best_image = None
    best_angle = None

    for angle, operation in rotations.items():
        image = original.copy()

        if operation is not None:
            image = cv2.rotate(image, operation)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.1
        ) as hands, mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.3
        ) as pose:

            hand_result = hands.process(rgb)
            pose_result = pose.process(rgb)

        hand_count = (
            len(hand_result.multi_hand_landmarks)
            if hand_result.multi_hand_landmarks else 0
        )
        pose_found = pose_result.pose_landmarks is not None

        print(
            f"{angle} degrees | Hands: {hand_count} "
            f"| Pose: {pose_found}"
        )

        if pose_found:
            mp_draw.draw_landmarks(
                image,
                pose_result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        if hand_result.multi_hand_landmarks:
            for landmarks in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image, landmarks, mp_hands.HAND_CONNECTIONS
                )

        score = hand_count * 2 + int(pose_found)

        if score > best_score:
            best_score = score
            best_image = image.copy()
            best_angle = angle

    if best_image is not None:
        output_path = OUTPUT / f"{path.parent.name}_best.jpg"
        cv2.imwrite(str(output_path), best_image)
        print("Best angle:", best_angle)
        print("Saved:", output_path)

print("\nRotation test completed.")
