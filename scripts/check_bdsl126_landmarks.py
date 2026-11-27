
import pandas as pd
from pathlib import Path

base = Path("datasets/processed")

for split in ["train", "val"]:

    path = base / f"bdsl126_landmarks_{split}.csv"

    if not path.exists():
        print(f"\n{split}: CSV not found")
        continue

    df = pd.read_csv(path)

    print(f"\n========== {split.upper()} ==========")
    print("Total saved:", len(df))
    print("Classes:", df["label"].nunique())
    print("One hand:", (df["hand_count"] == 1).sum())
    print("Two hands:", (df["hand_count"] == 2).sum())

    counts = df["label"].value_counts()

    print("\nLowest 15 classes:")
    print(counts.sort_values().head(15).to_string())

    print("\nClasses with zero samples:")

    all_classes = {
        p.name for p in (base / "bdsl126_classify" / split).iterdir()
        if p.is_dir()
    }

    missing = sorted(all_classes - set(counts.index))

    print(missing if missing else "None")
