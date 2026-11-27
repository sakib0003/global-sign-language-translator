from pathlib import Path
import yaml

yaml_path = Path("datasets/processed/bdsl126/data.yaml")

with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

root = Path(data["path"])

print("Classes:", len(data["names"]))
print("First class:", data["names"][0])
print("Last class:", data["names"][125])

for split in ["train", "val"]:
    image_dir = root / data[split]
    label_dir = root / "labels" / split

    images = list(image_dir.glob("*.jpg"))
    labels = list(label_dir.glob("*.txt"))

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Match:", len(images) == len(labels))