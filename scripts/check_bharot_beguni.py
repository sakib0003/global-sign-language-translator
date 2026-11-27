
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets" / "processed" / "bdsl126_classify"

for split in ["train", "val"]:
    print(f"\n========== {split.upper()} ==========")

    for class_name in ["Bharot", "Beguni"]:
        folder = DATASET / split / class_name

        if not folder.exists():
            print(f"\n{class_name}: Folder not found!")
            continue

        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        )

        print(f"\n{class_name} | Total images: {len(files)}")
        print("First 20 filenames:")

        for file in files[:20]:
            print(" ", file.name)

