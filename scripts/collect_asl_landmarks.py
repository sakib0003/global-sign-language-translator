import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp

DATA_FILE = "datasets/processed/asl_alphabet_landmarks.csv"
SAMPLES_PER_CLASS = 100

os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["label"] + [f"f{i}" for i in range(63)]
        )

label = input("Enter ASL letter (A-Z): ").strip().upper()

if label not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or len(label) != 1:
    raise ValueError("Enter one valid letter from A to Z")

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils

saved = 0
collecting = False
last_save = 0

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    print(f"Collecting sign: {label}")
    print("Press S to start collection, Q to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            points = np.array(
                [[p.x, p.y, p.z] for p in hand.landmark],
                dtype=np.float32
            )

            points -= points[0]
            scale = np.max(np.abs(points[:, :2]))

            if scale > 0:
                points /= scale

                if collecting and saved < SAMPLES_PER_CLASS:
                    if time.time() - last_save >= 0.15:
                        row = [label] + points.flatten().tolist()

                        with open(DATA_FILE, "a", newline="") as f:
                            csv.writer(f).writerow(row)

                        saved += 1
                        last_save = time.time()

        status = "COLLECTING" if collecting else "READY"
        cv2.putText(frame, f"Sign: {label} | {status}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {saved}/{SAMPLES_PER_CLASS}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow("ASL Landmark Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            collecting = True

        if key == ord("q") or saved >= SAMPLES_PER_CLASS:
            break

cap.release()
cv2.destroyAllWindows()

print(f"Saved {saved} samples for {label}")
print("Dataset:", DATA_FILE)
