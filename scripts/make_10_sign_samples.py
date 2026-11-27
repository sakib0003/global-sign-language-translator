import os
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "datasets" / "processed" / "bdsl126_classify" / "val"
MAPPING = ROOT / "datasets" / "processed" / "bdsl126_bengali_mapping.json"
OUTPUT = ROOT / "reports" / "bdsl126_10_sign_samples.jpg"

IMAGE_SIZE = (240, 200)
COLS = 2
ROWS = 5

random.seed(42)

with open(MAPPING, "r", encoding="utf-8") as f:
    mapping = json.load(f)

classes = sorted([p for p in DATASET.iterdir() if p.is_dir()])

if len(classes) < 10:
    raise RuntimeError("Dataset-এ ১০টি class পাওয়া যায়নি।")

selected = classes[:10]

font_path = (
    r"C:\Users\KHAN GADGET\AppData\Local\Microsoft"
    r"\Windows\Fonts\NotoSansBengali-VariableFont_wdth,wght.ttf"
)

try:
    font = ImageFont.truetype(font_path, 22)
except Exception:
    font = ImageFont.load_default()

canvas = Image.new(
    "RGB",
    (COLS * IMAGE_SIZE[0], ROWS * (IMAGE_SIZE[1] + 55)),
    "white"
)

draw = ImageDraw.Draw(canvas)

for i, folder in enumerate(selected):
    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    if not images:
        print("No images:", folder.name)
        continue

    image_path = random.choice(images)

    image = Image.open(image_path).convert("RGB")
    image.thumbnail(IMAGE_SIZE)

    x = (i % COLS) * IMAGE_SIZE[0]
    y = (i // COLS) * (IMAGE_SIZE[1] + 55)

    canvas.paste(image, (x, y))

    bengali_name = mapping.get(folder.name, folder.name)

    draw.text(
        (x + 5, y + IMAGE_SIZE[1] + 5),
        f"{i + 1}. {bengali_name}",
        fill="black",
        font=font
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT, quality=95)

print("\n10 Sign samples created:")
print(OUTPUT)