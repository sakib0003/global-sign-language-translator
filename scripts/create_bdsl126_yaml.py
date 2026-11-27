import json
from pathlib import Path

mapping_path = Path("datasets/processed/bdsl126_classes.json")
output_path = Path("datasets/processed/bdsl126/data.yaml")

with open(mapping_path, "r", encoding="utf-8") as f:
    classes = json.load(f)

dataset_path = output_path.parent.resolve().as_posix()

yaml_lines = [
    f"path: {dataset_path}",
    "train: images/train",
    "val: images/val",
    "names:"
]

for class_id, class_name in classes.items():
    yaml_lines.append(f"  {class_id}: {class_name}")

output_path.write_text(
    "\n".join(yaml_lines) + "\n",
    encoding="utf-8"
)

print("YAML created:", output_path)
print("Total classes:", len(classes))
print("First class:", list(classes.values())[0])
print("Last class:", list(classes.values())[-1])