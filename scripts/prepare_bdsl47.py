
"""
Prepare complete BdSL47 landmark dataset.

Supports:
- Alphabet signs: sign00, sign 00, Sign 00
- Digit signs: Sign 0, sign0
- Multiple signer folders
- 21 hand landmarks (x, y, z) = 63 features
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# 21 landmarks x 3 coordinates = 63 features
FEATURES = [
    f"{axis}{i:02d}"
    for i in range(21)
    for axis in ("x", "y", "z")
]


def main():

    parser = argparse.ArgumentParser(
        description="Prepare BdSL47 landmark dataset"
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Path to raw BdSL47 dataset"
    )

    parser.add_argument(
        "--out",
        default="datasets/processed/bdsl47_landmarks.csv",
        help="Output CSV path"
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    frames = []
    skipped = []

    all_csv = sorted(root.rglob("*.csv"))

    print(f"Total CSV files found: {len(all_csv)}")
    print("\nProcessing dataset...\n")

    processed_files = 0

    for path in all_csv:

        # Find sign folder:
        # sign00, sign 00, Sign 0, etc.
        sign_dir = next(
            (
                parent
                for parent in path.parents
                if re.fullmatch(
                    r"sign\s*\d+",
                    parent.name,
                    re.IGNORECASE
                )
            ),
            None
        )

        if sign_dir is None:
            skipped.append(
                (path, "Sign folder not identified")
            )
            continue

        try:

            relative_path = path.relative_to(root)

            # First folder is dataset section
            section = relative_path.parts[0]

            # Parent of sign folder is signer folder
            signer_folder = sign_dir.parent.name

        except (ValueError, IndexError):

            skipped.append(
                (path, "Could not identify section or signer")
            )
            continue

        # Extract sign number
        match = re.search(r"\d+", sign_dir.name)

        if match is None:
            skipped.append(
                (path, "Sign number not found")
            )
            continue

        sign_number = int(match.group())

        try:

            df = pd.read_csv(path)

        except Exception as error:

            skipped.append(
                (path, f"CSV read error: {error}")
            )
            continue

        # Clean column names
        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        # Check required landmark columns
        missing = [
            column
            for column in FEATURES
            if column not in df.columns
        ]

        if missing:

            skipped.append(
                (
                    path,
                    f"Missing {len(missing)} landmark columns"
                )
            )
            continue

        # Extract landmark features
        part = df[FEATURES].copy()

        # Convert features to numeric
        for column in FEATURES:

            part[column] = pd.to_numeric(
                part[column],
                errors="coerce"
            )

        # Remove invalid rows
        part = part.dropna(
            subset=FEATURES
        )

        if part.empty:

            skipped.append(
                (path, "No valid landmark rows")
            )
            continue

        # Normalize class IDs
        # sign00, sign 00 and Sign 0 become sign00
        label = (
            f"{section}__sign{sign_number:02d}"
        )

        # Keep signer identity separate per section
        signer = (
            f"{section}__{signer_folder}"
        )

        part["label"] = label
        part["signer"] = signer
        part["source_csv"] = str(relative_path)

        # Preserve original source label if available
        if "Label" in df.columns:

            part["source_label"] = (
                df.loc[part.index, "Label"].astype(str)
            )

        frames.append(part)

        processed_files += 1

        print(
            f"{relative_path}: "
            f"{len(part)} rows -> {label}"
        )

    # Stop if nothing was processed
    if not frames:

        raise SystemExit(
            "No usable CSV files found."
        )

    # Combine all processed CSV files
    output = pd.concat(
        frames,
        ignore_index=True
    )

    # Save output
    output_path = Path(args.out)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        output_path,
        index=False
    )

    # Dataset summary
    print("\n")
    print("=" * 45)
    print("       BdSL47 DATASET SUMMARY")
    print("=" * 45)

    print(
        "CSV files found:",
        len(all_csv)
    )

    print(
        "Processed CSV files:",
        processed_files
    )

    print(
        "Skipped files:",
        len(skipped)
    )

    print(
        "Usable rows:",
        len(output)
    )

    print(
        "Classes:",
        output["label"].nunique()
    )

    print(
        "Signer groups:",
        output["signer"].nunique()
    )

    print(
        "Saved:",
        output_path
    )

    print("\nRows per section:")

    print(
        output.groupby("label").size()
        .groupby(level=0).sum()
    )

    print("\nSigner groups:")

    for signer in sorted(output["signer"].unique()):
        print(signer)

    if skipped:

        print("\nSkipped files (first 20):")

        for file_path, reason in skipped[:20]:

            print(
                file_path,
                "-",
                reason
            )

    print("\n")
    print("=" * 45)
    print("Dataset preparation completed.")
    print("=" * 45)

    print(
        "\nNote: Class IDs are not mapped "
        "to Bangla letters yet."
    )


if __name__ == "__main__":
    main()
