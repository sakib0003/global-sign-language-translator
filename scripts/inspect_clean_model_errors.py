
import pandas as pd
import joblib
import cv2

from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent.parent

CSV_PATH = BASE / "datasets/processed/bdsl126_landmarks_clean_val.csv"
DATASET_DIR = BASE / "datasets/processed/bdsl126_classify_clean_v2"
MODEL_PATH = BASE / "models/bdsl126_landmark_model_clean.joblib"
OUTPUT_PATH = BASE / "reports/clean_model_error_samples.jpg"

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
feature_columns = model_data["feature_columns"]

df = pd.read_csv(CSV_PATH)

X = df[feature_columns].astype("float32")
predictions = model.predict(X)

df["predicted"] = predictions
wrong = df[df["label"] != df["predicted"]].copy()

wrong["pair"] = wrong["label"] + " -> " + wrong["predicted"]

top_pairs = wrong.groupby("pair").size().sort_values(ascending=False).head(6)

print("Top confusion pairs:")
print(top_pairs.to_string())

selected = []

for pair in top_pairs.index:
    actual, predicted = pair.split(" -> ", 1)

    samples = wrong[
        (wrong["label"] == actual) &
        (wrong["predicted"] == predicted)
    ].head(4)

    for _, row in samples.iterrows():
        selected.append(row)

if not selected:
    print("No errors found.")
    raise SystemExit

thumb_w = 240
thumb_h = 190
label_h = 48
columns = 4
rows = (len(selected) + columns - 1) // columns

canvas = Image.new(
    "RGB",
    (columns * thumb_w, rows * (thumb_h + label_h)),
    "white"
)

draw = ImageDraw.Draw(canvas)
shown = 0

for row in selected:
    source = Path(str(row["source"]))

    if source.is_absolute():
        image_path = source
    else:
        image_path = DATASET_DIR / source

    if not image_path.exists():
        print("Image not found:", image_path)
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print("Cannot read:", image_path)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    pil_image.thumbnail((thumb_w - 10, thumb_h - 10))

    x = (shown % columns) * thumb_w
    y = (shown // columns) * (thumb_h + label_h)

    canvas.paste(pil_image, (x + 5, y + 5))

    text = (
        f"Actual: {row['label']}\n"
        f"Predicted: {row['predicted']}"
    )

    draw.text((x + 5, y + thumb_h), text, fill="black")
    shown += 1

if shown == 0:
    print("No images could be loaded.")
    raise SystemExit

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
canvas = canvas.crop((0, 0, columns * thumb_w, ((shown + columns - 1) // columns) * (thumb_h + label_h)))
canvas.save(OUTPUT_PATH, quality=95)

print("\nImages displayed:", shown)
print("Saved:", OUTPUT_PATH)
