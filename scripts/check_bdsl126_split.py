from pathlib import Path

root = Path("datasets/processed/bdsl126")

for split in ["train", "val"]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    images = {p.stem for p in image_dir.glob("*.jpg")}
    labels = {p.stem for p in label_dir.glob("*.txt")}

    missing_labels = sorted(images - labels)
    missing_images = sorted(labels - images)

    print(f"\n--- {split.upper()} ---")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(missing_images))

    print("Examples:", missing_labels[:5])