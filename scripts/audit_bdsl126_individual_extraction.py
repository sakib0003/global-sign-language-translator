
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "processed"

SPLITS = ["train", "val"]

for split in SPLITS:

    input_file = (
        DATA_DIR / f"bdsl126_pose_individual_direct_{split}.csv"
    )

    report_file = (
        DATA_DIR / f"bdsl126_individual_audit_{split}.csv"
    )

    stats = defaultdict(lambda: {
        "total": 0,
        "hands": 0,
        "pose": 0,
        "both": 0,
        "neither": 0,
        "failed": 0
    })

    with open(input_file, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            label = row["label"]
            hand_count = int(row["hand_count"])
            pose_detected = int(row["pose_detected"])
            image_ok = int(row["image_read_ok"])

            item = stats[label]

            item["total"] += 1

            if image_ok == 0:
                item["failed"] += 1

            if hand_count > 0:
                item["hands"] += 1

            if pose_detected == 1:
                item["pose"] += 1

            if hand_count > 0 and pose_detected == 1:
                item["both"] += 1

            if hand_count == 0 and pose_detected == 0:
                item["neither"] += 1

    total_images = sum(x["total"] for x in stats.values())
    total_hands = sum(x["hands"] for x in stats.values())
    total_pose = sum(x["pose"] for x in stats.values())
    total_both = sum(x["both"] for x in stats.values())
    total_failed = sum(x["failed"] for x in stats.values())

    with open(report_file, "w", encoding="utf-8-sig", newline="") as f:

        columns = [
            "label",
            "total",
            "hands_detected",
            "pose_detected",
            "both_detected",
            "neither_detected",
            "failed"
        ]

        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for label in sorted(stats):

            item = stats[label]

            writer.writerow({
                "label": label,
                "total": item["total"],
                "hands_detected": item["hands"],
                "pose_detected": item["pose"],
                "both_detected": item["both"],
                "neither_detected": item["neither"],
                "failed": item["failed"]
            })

    zero_hand = sum(
        1 for x in stats.values() if x["hands"] == 0
    )

    zero_pose = sum(
        1 for x in stats.values() if x["pose"] == 0
    )

    print("\n" + "=" * 55)
    print(f"{split.upper()} DATASET AUDIT")
    print("=" * 55)

    print("Total classes:", len(stats))
    print("Total images:", total_images)
    print("Images with hands:", total_hands)
    print("Images with pose:", total_pose)
    print("Images with both:", total_both)
    print("Images with neither:", sum(
        x["neither"] for x in stats.values()
    ))
    print("Failed images:", total_failed)
    print("Classes with zero hand detection:", zero_hand)
    print("Classes with zero pose detection:", zero_pose)

    print("\nTop 10 classes with most hand detections:")

    ranked = sorted(
        stats.items(),
        key=lambda x: x[1]["hands"],
        reverse=True
    )

    for label, item in ranked[:10]:
        print(
            f"{label:20} "
            f"Total={item['total']:4} "
            f"Hands={item['hands']:4} "
            f"Pose={item['pose']:4}"
        )

    print("\nAudit CSV saved:", report_file)

print("\nALL AUDITS COMPLETED!")
