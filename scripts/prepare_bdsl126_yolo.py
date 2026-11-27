from pathlib import Path
import random
import os

random.seed(42)

source = Path("datasets/raw/bdsl126")
output = Path("datasets/processed/bdsl126")

folders = {
    "train_images": output / "images/train",
    "val_images": output / "images/val",
    "train_labels": output / "labels/train",
    "val_labels": output / "labels/val",
}

for folder in folders.values():
    folder.mkdir(parents=True, exist_ok=True)

classes = sorted(
    folder.name for folder in source.iterdir()
    if folder.is_dir()
)

train_count = 0
val_count = 0
missing = 0

for class_id, class_name in enumerate(classes):
    image_dir = source / class_name / "images"
    label_dir = source / class_name / "labels"

    valid = []

    for img in image_dir.glob("*.jpg"):
        label = label_dir / (img.stem + ".txt")

        if label.exists():
            valid.append((img, label))
        else:
            missing += 1

    random.shuffle(valid)
    split = int(len(valid) * 0.8)

    for index, (img, label) in enumerate(valid):
        subset = "train" if index < split else "val"

        # Unique filenames
        name = class_name + "__" + img.name

        image_target = folders[f"{subset}_images"] / name
        label_target = folders[f"{subset}_labels"] / (Path(name).stem + ".txt")

        for line in label.read_text().splitlines():
            if line.strip() and int(line.split()[0]) != class_id:
                raise ValueError(
                    f"Wrong class ID: {class_name} - {img.name}"
                )

        if not image_target.exists():
            os.link(img, image_target)

        if not label_target.exists():
            os.link(label, label_target)

        if subset == "train":
            train_count += 1
        else:
            val_count += 1

print("Classes:", len(classes))
print("Training images:", train_count)
print("Validation images:", val_count)
print("Missing labels:", missing)
print("Dataset preparation complete!")