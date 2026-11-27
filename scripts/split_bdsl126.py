from pathlib import Path
import random
import shutil

random.seed(42)

root = Path("datasets/raw/bdsl126")
output = Path("datasets/processed/bdsl126")

train_dir = output / "images" / "train"
val_dir = output / "images" / "val"

for folder in [train_dir, val_dir]:
    folder.mkdir(parents=True, exist_ok=True)

total_train = 0
total_val = 0

for class_folder in sorted(root.iterdir()):
    if not class_folder.is_dir():
        continue

    image_dir = class_folder / "images"
    label_dir = class_folder / "labels"

    images = []

    for image in image_dir.glob("*.jpg"):
        label = label_dir / f"{image.stem}.txt"

        if label.exists():
            images.append(image)

    random.shuffle(images)

    split_index = int(len(images) * 0.8)

    train_images = images[:split_index]
    val_images = images[split_index:]

    for image in train_images:
        shutil.copy2(image, train_dir / image.name)
        shutil.copy2(
            label_dir / f"{image.stem}.txt",
            train_dir / f"{image.stem}.txt"
        )

    for image in val_images:
        shutil.copy2(image, val_dir / image.name)
        shutil.copy2(
            label_dir / f"{image.stem}.txt",
            val_dir / f"{image.stem}.txt"
        )

    total_train += len(train_images)
    total_val += len(val_images)

print("Training images:", total_train)
print("Validation images:", total_val)
print("Total:", total_train + total_val)