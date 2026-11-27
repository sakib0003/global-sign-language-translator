
import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report


FEATURES = [
    f"{axis}{i:02d}"
    for i in range(21)
    for axis in ("x", "y", "z")
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="datasets/processed/bdsl47_landmarks.csv"
    )

    parser.add_argument(
        "--out",
        default="models/bdsl47.joblib"
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=FEATURES + ["label", "signer"])

    if args.max_per_class > 0:
        df = pd.concat(
            [
                group.sample(
                    min(len(group), args.max_per_class),
                    random_state=42
                )
                for _, group in df.groupby("label")
            ],
            ignore_index=True
        )

    # Split by signer within each class
    train_parts = []
    test_parts = []

    for label, class_df in df.groupby("label"):
        signers = sorted(class_df["signer"].unique())

        if len(signers) < 2:
            raise SystemExit(
                f"Not enough signers for class: {label}"
            )

        test_signers = set(signers[::5])
        train_signers = set(signers) - test_signers

        train_parts.append(
            class_df[class_df["signer"].isin(train_signers)]
        )

        test_parts.append(
            class_df[class_df["signer"].isin(test_signers)]
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # Check for signer leakage
    overlap = (
        set(train_df["signer"])
        & set(test_df["signer"])
    )

    if overlap:
        raise SystemExit(
            f"Signer leakage detected: {overlap}"
        )

    X_train = train_df[FEATURES].astype("float32")
    y_train = train_df["label"].astype(str)

    X_test = test_df[FEATURES].astype("float32")
    y_test = test_df["label"].astype(str)

    model = ExtraTreesClassifier(
        n_estimators=160,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    print("Training started...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTrain rows:", len(train_df))
    print("Test rows:", len(test_df))
    print("Train signers:", train_df["signer"].nunique())
    print("Test signers:", test_df["signer"].nunique())
    print("Signer overlap:", len(overlap))

    print(
        "\nAccuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print(
        "\nClassification Report:\n",
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURES
        },
        output
    )

    print("\nSaved model:", output)
    print("Training completed!")


if __name__ == "__main__":
    main()
