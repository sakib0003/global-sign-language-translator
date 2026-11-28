
from pathlib import Path
import random
import cv2
import pandas as pd
import mediapipe as mp

ROOT = Path("datasets/processed/bdsl126_classify_clean_v2/val")
OUTPUT = Path("reports/bdsl126_rotation_300")
OUTPUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 300
random.seed(42)

files = [
    p for p in ROOT.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and "_frame" not in p.name
]

samples = random.sample(files, min(SAMPLE_SIZE, len(files)))

rotations = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE
}

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

results = []

print("Testing images:", len(samples))

for angle, operation in rotations.items():
    print(f"\nTesting rotation: {angle} degrees")

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.1
    ) as hands, mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.3
    ) as pose:

        for index, path in enumerate(samples, 1):
            image = cv2.imread(str(path))

            if image is None:
                continue

            if operation is not None:
                image = cv2.rotate(image, operation)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            hand_result = hands.process(rgb)
            pose_result = pose.process(rgb)

            hand_count = (
                len(hand_result.multi_hand_landmarks)
                if hand_result.multi_hand_landmarks else 0
            )

            pose_found = pose_result.pose_landmarks is not None

            results.append({
                "image": str(path),
                "class": path.parent.name,
                "rotation": angle,
                "hands": hand_count,
                "pose": int(pose_found)
            })

            if index % 50 == 0:
                print(f"Processed {index}/{len(samples)}")

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT / "rotation_details.csv",
    index=False
)

summary = df.groupby("rotation").agg(
    images_tested=("image", "count"),
    pose_detected=("pose", "sum"),
    images_with_hands=("hands", lambda x: (x > 0).sum()),
    total_hands=("hands", "sum")
)

summary["pose_rate"] = (
    summary["pose_detected"] / summary["images_tested"] * 100
)

summary["hand_rate"] = (
    summary["images_with_hands"] / summary["images_tested"] * 100
)

print("\n========== ROTATION SUMMARY ==========")
print(summary.round(2).to_string())

best_pose = (
    df[df["pose"] == 1]
    .groupby("image")["rotation"]
    .count()
)

print("\nImages with Pose in at least one rotation:",
      len(best_pose))

best_rotation = (
    df.groupby("rotation")["pose"].sum().sort_values(ascending=False)
)

print("\nBest rotation by total Pose detections:")
print(best_rotation.to_string())

print("\nSaved:", OUTPUT / "rotation_details.csv")
