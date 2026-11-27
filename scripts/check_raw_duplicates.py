
from pathlib import Path
import hashlib
from collections import defaultdict

ROOT = Path("datasets/raw/bdsl126")

CLASSES = ["Bharot", "Beguni"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

def get_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

hash_groups = defaultdict(list)

for class_name in CLASSES:
    image_dir = ROOT / class_name / "images"

    print(f"\n{class_name}:")

    if not image_dir.exists():
        print("Images folder পাওয়া যায়নি")
        continue

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    print("মোট ছবি:", len(images))

    for image_path in images:
        image_hash = get_hash(image_path)
        hash_groups[image_hash].append(
            (class_name, image_path.name)
        )

print("\n========== DUPLICATE REPORT ==========")

duplicate_count = 0

for image_hash, files in hash_groups.items():
    class_names = {item[0] for item in files}

    if len(files) > 1 and len(class_names) > 1:
        duplicate_count += 1
        print(f"\nDuplicate #{duplicate_count}")
        for class_name, filename in files:
            print(f"{class_name}: {filename}")

print("\nদুই ক্লাসের মধ্যে একই ছবির সংখ্যা:", duplicate_count)
print("পরীক্ষা শেষ। কোনো ফাইল পরিবর্তন করা হয়নি।")
