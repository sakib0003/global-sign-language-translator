
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent

files = [
    ROOT / "datasets/processed/bdsl126_landmarks_train.csv",
    ROOT / "datasets/processed/bdsl126_landmarks_val.csv",
]

rows = []

for file in files:
    df = pd.read_csv(file)
    split = "TRAIN" if "train" in file.name else "VAL"

    for _, row in df.iterrows():
        filename = Path(str(row["source"])).stem

        # Roboflow hash বের করা
        match = re.search(r"\.rf\.([a-f0-9]+)", filename, re.I)
        image_hash = match.group(1).lower() if match else None

        rows.append({
            "split": split,
            "label": row["label"],
            "hash": image_hash,
            "filename": filename
        })

data = pd.DataFrame(rows)
data = data.dropna(subset=["hash"])

print("মোট hash-যুক্ত ছবি:", len(data))

# একই hash একাধিক ক্লাসে আছে কি না
print("\n===== DIFFERENT CLASS DUPLICATES =====")

for image_hash, group in data.groupby("hash"):
    if group["label"].nunique() > 1:
        print("\nHash:", image_hash)
        print(group[["split", "label", "filename"]].to_string(index=False))

# একই hash Train ও Val-এ আছে কি না
print("\n===== TRAIN / VAL DUPLICATES =====")

for image_hash, group in data.groupby("hash"):
    if group["split"].nunique() > 1:
        print("\nHash:", image_hash)
        print(group[["split", "label", "filename"]].to_string(index=False))

print("\nপরীক্ষা শেষ।")
