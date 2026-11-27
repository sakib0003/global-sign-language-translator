
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

for split in ["train", "val"]:
    csv_path = ROOT / "datasets" / "processed" / f"bdsl126_landmarks_{split}.csv"

    print(f"\n========== {split.upper()} LANDMARK DATA ==========")

    df = pd.read_csv(csv_path)

    for name in ["Bharot", "Beguni"]:
        rows = df[df["label"] == name]

        print(f"\n{name}: {len(rows)} extracted samples")

        for source in rows["source"].head(10):
            print(Path(str(source)).name)
