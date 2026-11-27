"""Collect labeled landmark samples using a webcam.

Example:
  python collect_data.py --language ASL --label hello --samples 100
Press SPACE to save a sample; press Q to quit.
"""
import argparse
import csv
from pathlib import Path
import cv2
from src.features import extract_hand_features
from config_loader import LANGUAGES

parser = argparse.ArgumentParser()
parser.add_argument("--language", required=True, choices=["BdSL", "ASL", "BSL"])
parser.add_argument("--label", required=True, help="Exact sign class label, e.g. hello")
parser.add_argument("--samples", type=int, default=100)
args = parser.parse_args()

output = Path(LANGUAGES[args.language]["dataset_file"])
output.parent.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not found. Check camera permission/device.")

header = ["label"] + [f"f{i}" for i in range(63)]
new_file = not output.exists()
saved = 0
try:
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        print("Show the sign clearly. SPACE = save, Q = quit.")
        while saved < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{args.language} | {args.label} | {saved}/{args.samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Dataset collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                features = extract_hand_features(frame)
                if features is not None:
                    writer.writerow([args.label] + features.tolist())
                    f.flush()
                    saved += 1
                    print(f"Saved {saved}/{args.samples}")
                else:
                    print("No hand detected; try again.")
finally:
    cap.release()
    cv2.destroyAllWindows()
print(f"Finished. Saved {saved} samples to {output}")
