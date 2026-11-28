
from pathlib import Path
import random
import cv2
import numpy as np

root = Path("datasets/processed/bdsl126_classify_clean_v2/val")
output = Path("reports/bdsl126_20_samples.jpg")

files = [
    p for p in root.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and "_frame" not in p.name
]

random.seed(42)
samples = random.sample(files, min(20, len(files)))

cell_w, cell_h = 240, 240
cols = 4
rows = (len(samples) + cols - 1) // cols

sheet = np.full(
    (rows * cell_h, cols * cell_w, 3),
    255,
    dtype=np.uint8
)

for i, path in enumerate(samples):
    image = cv2.imread(str(path))
    if image is None:
        continue

    image = cv2.resize(image, (cell_w, 205))
    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    sheet[y:y+205, x:x+cell_w] = image

    label = path.parent.name
    cv2.putText(
        sheet, label, (x + 5, y + 228),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 0, 0), 2
    )

output.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output), sheet)

print("Contact sheet saved:", output)
