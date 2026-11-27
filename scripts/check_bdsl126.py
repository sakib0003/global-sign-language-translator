from pathlib import Path

root = Path("datasets/raw/bdsl126")

names = sorted(
    folder.name
    for folder in root.iterdir()
    if folder.is_dir()
)

for name in ["Aage", "Aat", "Elaka", "Uttar"]:
    label_files = list((root / name / "labels").glob("*.txt"))

    if label_files:
        actual_id = label_files[0].read_text().split()[0]
        expected_id = names.index(name)

        print(
            name,
            "| Expected:", expected_id,
            "| Actual:", actual_id
        )