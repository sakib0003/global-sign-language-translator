
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

# Import from the same folder
from landmark_utils import normalize_landmarks


FEATURES = [
    f"{axis}{i:02d}"
    for i in range(21)
    for axis in ("x", "y", "z")
]


def train_section(df, section, output_path, max_per_class):

    print("\n" + "=" * 55)
    print("Training:", section)
    print("=" * 55)

    data = df[
        df["label"].str.startswith(section + "__")
    ].copy()

    if data.empty:
        raise ValueError(f"No data found for {section}")

    # Limit samples per class
    if max_per_class > 0:
        data = pd.concat(
            [
                group.sample(
                    n=min(len(group), max_per_class),
                    random_state=42
                )
                for _, group in data.groupby("label")
            ],
            ignore_index=True
        )

    train_parts = []
    test_parts = []

    # Split by signer within each class
    for label, class_df in data.groupby("label"):

        signers = sorted(
            class_df["signer"].unique()
        )

        if len(signers) < 2:
            raise ValueError(
                f"Not enough signers for {label}"
            )

        test_signers = set(signers[::5])
        train_signers = set(signers) - test_signers

        train_parts.append(
            class_df[
                class_df["signer"].isin(train_signers)
            ]
        )

        test_parts.append(
            class_df[
                class_df["signer"].isin(test_signers)
            ]
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True
    )

    # Verify no signer overlap
    overlap = (
        set(train_df["signer"])
        & set(test_df["signer"])
    )

    if overlap:
        raise ValueError(
            f"Signer leakage detected: {overlap}"
        )

    print("\nNormalizing training landmarks...")

    X_train_values = np.vstack([
        normalize_landmarks(row)
        for row in train_df[FEATURES].to_numpy()
    ])

    print("Normalizing testing landmarks...")

    X_test_values = np.vstack([
        normalize_landmarks(row)
        for row in test_df[FEATURES].to_numpy()
    ])

    X_train = pd.DataFrame(
        X_train_values,
        columns=FEATURES
    )

    X_test = pd.DataFrame(
        X_test_values,
        columns=FEATURES
    )

    y_train = train_df["label"].astype(str)
    y_test = test_df["label"].astype(str)

    print("\nClasses:", data["label"].nunique())
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training signers:", train_df["signer"].nunique())
    print("Testing signers:", test_df["signer"].nunique())

    # Create classifier
    model = ExtraTreesClassifier(
        n_estimators=200,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    print("\nModel training started...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save model
    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURES,
            "section": section,
            "normalized": True
        },
        output
    )

    print("\nModel saved:", output)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="datasets/processed/bdsl47_landmarks.csv"
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    print("Loading dataset...")

    df = pd.read_csv(csv_path)

    missing_columns = [
        col for col in FEATURES + ["label", "signer"]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    df = df.dropna(
        subset=FEATURES + ["label", "signer"]
    ).copy()

    df["label"] = df["label"].astype(str)
    df["signer"] = df["signer"].astype(str)

    print("Total usable rows:", len(df))

    # Train alphabet model
    train_section(
        df,
        "Bangla Sign Language Dataset - Sign Alphabets",
        "models/bdsl47_alphabet_normalized.joblib",
        args.max_per_class
    )

    # Train digit model
    train_section(
        df,
        "Bangla Sign Language Dataset - Sign Digits",
        "models/bdsl47_digits_normalized.joblib",
        args.max_per_class
    )

    print("\n" + "=" * 55)
    print("BOTH NORMALIZED MODELS TRAINED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    main()
