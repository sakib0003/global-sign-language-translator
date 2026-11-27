
from pathlib import Path
from collections import defaultdict
import hashlib

ROOT = Path("datasets/raw/bdsl126")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

hash_groups = defaultdict(list)
total_images = 0

class_dirs = sorted(
    p for p in ROOT.iterdir()
    if p.is_dir() and (p / "images").is_dir()
)

print("মোট ক্লাস:", len(class_dirs))
print("ছবি পরীক্ষা শুরু হচ্ছে...\n")

for class_dir in class_dirs:
    image_dir = class_dir / "images"

    for path in image_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

        hash_groups[h.hexdigest()].append(
            (class_dir.name, path.name)
        )
        total_images += 1

pair_counts = defaultdict(int)
duplicate_groups = []

for image_hash, files in hash_groups.items():
    classes = sorted(set(item[0] for item in files))

    if len(classes) > 1:
        duplicate_groups.append(files)

        for i, class_a in enumerate(classes):
            for class_b in classes[i + 1:]:
                pair_counts[(class_a, class_b)] += 1

print("\n========== AUDIT REPORT ==========")
print("মোট ছবি পরীক্ষা:", total_images)
print("একাধিক ক্লাসে পাওয়া Duplicate group:", len(duplicate_groups))

print("\n========== CLASS PAIRS ==========")

for pair, count in sorted(
    pair_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{pair[0]} <-> {pair[1]} : {count} টি")

print("\n========== SAMPLE DUPLICATES ==========")

for index, files in enumerate(duplicate_groups[:20], 1):
    print(f"\nDuplicate {index}:")
    for class_name, filename in files:
        print(f"  {class_name}: {filename}")

print("\nAudit শেষ। কোনো ফাইল পরিবর্তন বা ডিলিট করা হয়নি।")
