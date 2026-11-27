from pathlib import Path
import os

source = Path("datasets/processed/bdsl126/images")
output = Path("datasets/processed/bdsl126_classify")

for split in ["train", "val"]:
    image_dir = source / split
    output_dir = output / split
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for img in image_dir.iterdir():
        if not img.is_file() or img.suffix.lower() != ".jpg":
            continue

        class_name = img.name.split("__")[0]
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        target = class_dir / img.name

        if not target.exists():
            os.link(img, target)

        count += 1

    print(f"{split}: {count} images processed")

print("Dataset preparation complete!")