
import pandas as pd
import re
from pathlib import Path

TRAIN_CSV = Path("datasets/processed/bdsl126_pose_landmarks_train.csv")
VAL_CSV = Path("datasets/processed/bdsl126_pose_landmarks_val.csv")


def get_group(source):
    filename = Path(source).name

    # Frame number এবং Roboflow hash বাদ দিয়ে
    # একই sequence-এর সব Frame-কে একটি Group করা
    filename = re.sub(
        r"_frame\d+_jpg\.rf\.[a-f0-9]+\.jpg$",
        "_frameGROUP",
        filename,
        flags=re.IGNORECASE
    )

    return filename.lower()


train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)

train_groups = train["source"].map(get_group)
val_groups = val["source"].map(get_group)

train_set = set(train_groups)
val_set = set(val_groups)

overlap = train_set & val_set

print("\n===== CORRECTED FRAME GROUP AUDIT =====")
print("Train samples:", len(train))
print("Validation samples:", len(val))
print("Train unique groups:", len(train_set))
print("Validation unique groups:", len(val_set))
print("Groups present in BOTH:", len(overlap))

print("\n===== OVERLAPPING GROUP DETAILS =====")

for group in sorted(overlap)[:30]:
    train_count = (train_groups == group).sum()
    val_count = (val_groups == group).sum()

    print(
        f"{group} | "
        f"Train: {train_count} | "
        f"Validation: {val_count}"
    )

print("\n===== FRAME GROUP COUNTS =====")
print("Train frame samples:", train["source"].str.contains(
    r"frame\d+", case=False, regex=True
).sum())

print("Validation frame samples:", val["source"].str.contains(
    r"frame\d+", case=False, regex=True
).sum())
