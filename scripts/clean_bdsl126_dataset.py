
from pathlib import Path
import shutil

SOURCE = Path("datasets/processed/bdsl126_classify")
DEST = Path("datasets/processed/bdsl126_classify_clean_v2")

SPLITS = ["train", "val"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

if DEST.exists():
    print("ERROR: Clean v2 folder already exists!")
    print("নতুন folder name ব্যবহার করো।")
    raise SystemExit

total_copied = 0
total_skipped = 0
skipped_train = 0
skipped_val = 0

for split in SPLITS:
    source_split = SOURCE / split
    dest_split = DEST / split

    if not source_split.exists():
        print("Missing folder:", source_split)
        raise SystemExit

    for class_dir in sorted(source_split.iterdir()):
        if not class_dir.is_dir():
            continue

        output_class = dest_split / class_dir.name
        output_class.mkdir(parents=True, exist_ok=True)

        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            filename = image_path.name.lower()

            # শুধু Beguni ক্লাসের ভিতরে Bharot-এর ছবি বাদ
            is_wrong_duplicate = (
                class_dir.name.lower() == "beguni"
                and "bharot_" in filename
            )

            if is_wrong_duplicate:
                total_skipped += 1

                if split == "train":
                    skipped_train += 1
                else:
                    skipped_val += 1

                continue

            destination = output_class / image_path.name

            try:
                destination.hardlink_to(image_path)
            except OSError:
                shutil.copy2(image_path, destination)

            total_copied += 1

    print(f"{split} সম্পন্ন")

print("\n========== CLEAN DATASET REPORT ==========")
print("মোট রাখা হয়েছে:", total_copied)
print("মোট বাদ:", total_skipped)
print("Train থেকে বাদ:", skipped_train)
print("Validation থেকে বাদ:", skipped_val)
print("Clean dataset:", DEST)
print("\nমূল Dataset পরিবর্তন করা হয়নি।")
