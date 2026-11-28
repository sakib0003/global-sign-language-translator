
from pathlib import Path
import random
import re
import cv2
import numpy as np

ROOT = Path("datasets/processed/bdsl126_classify_clean_v2/val")
CFG = "models/hand_detector/cross-hands-yolov4-tiny.cfg"
WEIGHTS = "models/hand_detector/cross-hands-yolov4-tiny.weights"

OUTPUT = Path("reports/bdsl126_yolo_hand_samples")
OUTPUT.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 300
INPUT_SIZE = 256
CONFIDENCE = 0.25
NMS_THRESHOLD = 0.30

random.seed(42)

# Load model
net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
output_layers = net.getUnconnectedOutLayersNames()

# Collect individual images, excluding video frames
extensions = {".jpg", ".jpeg", ".png"}
images = [
    p for p in ROOT.rglob("*")
    if p.suffix.lower() in extensions
    and not re.search(r"_frame\d+_jpg\.rf\.", p.name, re.I)
]

if not images:
    raise RuntimeError(f"No images found in {ROOT}")

samples = random.sample(images, min(SAMPLE_SIZE, len(images)))

detected_images = 0
total_hands = 0
failed_images = 0
saved = 0

for index, path in enumerate(samples, 1):
    image = cv2.imread(str(path))

    if image is None:
        failed_images += 1
        continue

    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        1 / 255.0,
        (INPUT_SIZE, INPUT_SIZE),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            confidence = float(detection[5])

            if confidence < CONFIDENCE:
                continue

            center_x, center_y, box_w, box_h = (
                detection[:4] * np.array([width, height, width, height])
            ).astype(int)

            x = int(center_x - box_w / 2)
            y = int(center_y - box_h / 2)

            boxes.append([x, y, int(box_w), int(box_h)])
            confidences.append(confidence)

    indexes = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE, NMS_THRESHOLD
    )

    count = len(indexes) if len(indexes) > 0 else 0

    if count:
        detected_images += 1
        total_hands += count

        for i in np.array(indexes).flatten():
            x, y, w, h = boxes[i]
            conf = confidences[i]

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Hand {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        if saved < 30:
            output_path = OUTPUT / f"detected_{index:03d}.jpg"
            cv2.imwrite(str(output_path), image)
            saved += 1

    if index % 50 == 0:
        print(f"Processed {index}/{len(samples)}")

print("\n===== YOLO HAND DETECTION RESULTS =====")
print(f"Images tested: {len(samples)}")
print(f"Images with hands detected: {detected_images}")
print(f"Detection rate: {detected_images / max(1, len(samples)) * 100:.2f}%")
print(f"Total hands detected: {total_hands}")
print(f"Failed images: {failed_images}")
print(f"Saved visual samples: {saved}")
print(f"Output folder: {OUTPUT}")
