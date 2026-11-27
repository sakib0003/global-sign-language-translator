from pathlib import Path
import json

root = Path("datasets/raw/bdsl126")
output = Path("datasets/processed")

output.mkdir(parents=True, exist_ok=True)

classes = sorted(
    folder.name
    for folder in root.iterdir()
    if folder.is_dir()
)

mapping = {
    str(index): name
    for index, name in enumerate(classes)
}

file_path = output / "bdsl126_classes.json"

with open(file_path, "w", encoding="utf-8") as file:
    json.dump(mapping, file, ensure_ascii=False, indent=4)

print("Total classes:", len(classes))
print("First class:", mapping["0"])
print("Last class:", mapping[str(len(classes) - 1)])
print("Mapping saved:", file_path)