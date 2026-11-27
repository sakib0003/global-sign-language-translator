from pathlib import Path

root = Path("datasets/processed/bdsl126")

total_removed = 0

for split in ["train", "val"]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    removed = 0

    for image in image_dir.glob("*.jpg"):
        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            image.unlink()
            removed += 1

    total_removed += removed
    print(f"{split}: Removed {removed} unmatched images")

print("Total removed:", total_removed)
print("Cleanup complete!")