
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "datasets" / "processed" / "bdsl126_classify" / "val"
OUTPUT = ROOT / "reports" / "bharot_beguni_comparison.jpg"

random.seed(42)

classes = ["Bharot", "Beguni"]
samples = []

for class_name in classes:
    folder = DATASET / class_name

    if not folder.exists():
        print("Class folder missing:", folder)
        continue

    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    for path in random.sample(images, min(10, len(images))):
        samples.append((class_name, path))

cell_w, cell_h = 320, 260
canvas = Image.new("RGB", (cell_w * 2, cell_h * 10), "white")
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font = ImageFont.load_default()

for i, (class_name, path) in enumerate(samples):
    image = Image.open(path).convert("RGB")
    image.thumbnail((cell_w - 10, cell_h - 45))

    x = 0 if class_name == "Bharot" else cell_w
    y = (i if class_name == "Bharot" else
         [name for name, _ in samples[:i + 1]].count("Beguni") - 1) * cell_h

    canvas.paste(image, (x, y))
    draw.text((x + 5, y + cell_h - 35), class_name, fill="black", font=font)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT)

print("Comparison saved:", OUTPUT)
