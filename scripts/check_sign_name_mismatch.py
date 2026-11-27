
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent

files = [
    ROOT / "datasets/processed/bdsl126_landmarks_train.csv",
    ROOT / "datasets/processed/bdsl126_landmarks_val.csv",
]

data = []

for file in files:
    df = pd.read_csv(file)
    split = "TRAIN" if "train" in file.name else "VAL"

    for _, row in df.iterrows():
        filename = Path(str(row["source"])).stem

        match = re.search(r"__(.*?)_frame\d+", filename, re.I)
        source_name = match.group(1).lower() if match else "unknown"

        data.append({
            "split": split,
            "label": row["label"],
            "source_name": source_name,
            "filename": filename
        })

result = pd.DataFrame(data)

for cls in ["Bharot", "Beguni"]:
    print(f"\n========== {cls} ==========")

    subset = result[result["label"] == cls]

    for split in ["TRAIN", "VAL"]:
        part = subset[subset["split"] == split]

        print(f"\n{split}:")
        print(part["source_name"].value_counts().to_string())

    print("\nফাইলের নাম ও ক্লাসের অমিল:")
    mismatch = subset[
        subset["source_name"].str.lower() != cls.lower()
    ]

    print("মোট সন্দেহজনক ফাইল:", len(mismatch))

    for name in mismatch["filename"].head(10):
        print(" ", name)

print("\nপরীক্ষা শেষ।")
