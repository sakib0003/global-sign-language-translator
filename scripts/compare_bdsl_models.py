
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(
    0,
    os.path.join(ROOT, "scripts")
)

from landmark_utils import normalize_landmarks

CSV_PATH = os.path.join(
    ROOT,
    "datasets",
    "processed",
    "bdsl47_landmarks.csv"
)

OUTPUT_DIR = os.path.join(ROOT, "reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(CSV_PATH)

print("Dataset rows:", len(df))
print("Dataset columns:", len(df.columns))

# Alphabet dataset only
alphabet = df[
    df["label"].astype(str).str.contains(
        "Sign Alphabets",
        case=False,
        na=False
    )
].copy()

if alphabet.empty:
    raise ValueError("Alphabet dataset not found.")

# ==========================================
# 3. SELECT ONLY 63 LANDMARK FEATURES
# ==========================================

feature_columns = [
    col for col in alphabet.columns
    if col.startswith(("x", "y", "z"))
]

if len(feature_columns) != 63:
    raise ValueError(
        f"Expected 63 landmark features, "
        f"but found {len(feature_columns)}.\n"
        f"Detected columns: {feature_columns}"
    )

print("\nLandmark features:", len(feature_columns))

# Verify numeric data
for col in feature_columns:
    alphabet[col] = pd.to_numeric(
        alphabet[col],
        errors="coerce"
    )

if alphabet[feature_columns].isna().any().any():
    raise ValueError(
        "Missing or non-numeric landmark values found."
    )

# ==========================================
# 4. SIGNER-INDEPENDENT SPLIT
# ==========================================

if "signer" not in alphabet.columns:
    raise ValueError("Signer column not found.")

signers = sorted(
    alphabet["signer"].unique()
)

test_signers = set(signers[::5])

train_df = alphabet[
    ~alphabet["signer"].isin(test_signers)
].copy()

test_df = alphabet[
    alphabet["signer"].isin(test_signers)
].copy()

print("\nTraining signers:", len(
    train_df["signer"].unique()
))

print("Testing signers:", len(
    test_df["signer"].unique()
))

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

# ==========================================
# 5. NORMALIZE LANDMARKS
# ==========================================

print("\nNormalizing training landmarks...")

X_train = np.array(
    [
        normalize_landmarks(row)
        for row in train_df[feature_columns].values
    ],
    dtype=np.float32
)

print("Normalizing testing landmarks...")

X_test = np.array(
    [
        normalize_landmarks(row)
        for row in test_df[feature_columns].values
    ],
    dtype=np.float32
)

y_train = train_df["label"].astype(str).values
y_test = test_df["label"].astype(str).values

# ==========================================
# 6. DEFINE MODELS
# ==========================================

models = {

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "SVM_RBF": SVC(
        C=10,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42
    )

}

# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

results = []

for name, model in models.items():

    print("\n" + "=" * 55)
    print("TRAINING:", name)
    print("=" * 55)

    start_time = time.time()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    duration = time.time() - start_time

    results.append({
        "Model": name,
        "Accuracy (%)": round(
            accuracy * 100, 2
        ),
        "Training Time (seconds)": round(
            duration, 2
        )
    })

    print(
        "\nAccuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    print(
        "Training time:",
        round(duration, 2),
        "seconds"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save model
    model_path = os.path.join(
        OUTPUT_DIR,
        f"comparison_{name}.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "section": "alphabet",
            "normalized": True
        },
        model_path
    )

    print("Model saved:", model_path)

# ==========================================
# 8. FINAL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

print(
    results_df.to_string(index=False)
)

# Save CSV report
report_path = os.path.join(
    OUTPUT_DIR,
    "bdsl_model_comparison.csv"
)

results_df.to_csv(
    report_path,
    index=False
)

print("\nComparison report saved:", report_path)

print("\nALL MODEL TESTS COMPLETED SUCCESSFULLY!")
