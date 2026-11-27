from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent

train = pd.read_csv(ROOT / "datasets/processed/bdsl126_landmarks_train.csv")
val = pd.read_csv(ROOT / "datasets/processed/bdsl126_landmarks_val.csv")

for cls in ["Bharot", "Beguni"]:
    print(f"\n========== {cls} ==========")

    tr = train[train["label"] == cls]["source"].astype(str)
    va = val[val["label"] == cls]["source"].astype(str)

    def get_frame(path):
        match = re.search(r"frame(\d+)", Path(path).stem, re.I)
        return match.group(1) if match else None

    train_frames = {}
    for path in tr:
        frame = get_frame(path)
        if frame:
            train_frames.setdefault(frame, []).append(Path(path).name)

    val_frames = {}
    for path in va:
        frame = get_frame(path)
        if frame:
            val_frames.setdefault(frame, []).append(Path(path).name)

    overlap = sorted(set(train_frames) & set(val_frames))

    print("মিল পাওয়া frame:", len(overlap))

    for frame in overlap[:10]:
        print(f"\nFrame {frame}")
        print("TRAIN:")
        for name in train_frames[frame]:
            print(" ", name)

        print("VAL:")
        for name in val_frames[frame]:
            print(" ", name)