
from pathlib import Path

ROOT = Path("datasets/processed/bdsl126_classify_clean_v2")

for split in ["train", "val"]:
    split_dir = ROOT / split

    classes = [
        p for p in split_dir.iterdir()
        if p.is_dir()
    ]

    total = 0
    beguni_bharot = 0
    beguni_own = 0

    for class_dir in classes:
        images = [
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]

        total += len(images)

        if class_dir.name.lower() == "beguni":
            beguni_bharot += sum(
                "bharot_" in p.name.lower() for p in images
            )
            beguni_own += sum(
                "beguni_" in p.name.lower() for p in images
            )

    print(f"\n===== {split.upper()} =====")
    print("Classes:", len(classes))
    print("Total images:", total)
    print("Beguni-এর Bharot ছবি:", beguni_bharot)
    print("Beguni-এর নিজস্ব ছবি:", beguni_own)

print("\nVerification complete.")
