
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ==========================================
# 1. PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TRAIN_FILE = DATA_DIR / "bdsl126_pose_rotation_best_train.csv"
VAL_FILE = DATA_DIR / "bdsl126_pose_rotation_best_val.csv"

MODEL_FILE = MODEL_DIR / "bdsl126_rotation_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

feature_columns = sorted(
    [c for c in train_df.columns if c.startswith("feature_")],
    key=lambda x: int(x.split("_")[1])
)

if len(feature_columns) != 260:
    raise ValueError(
        f"Expected 260 features, got {len(feature_columns)}"
    )

if feature_columns != [
    c for c in val_df.columns if c.startswith("feature_")
]:
    raise ValueError("Train and validation feature columns differ.")

train_df = train_df[train_df["image_read_ok"] == 1].copy()
val_df = val_df[val_df["image_read_ok"] == 1].copy()

X_train = train_df[feature_columns].fillna(0).to_numpy(
    dtype=np.float32
)
X_val = val_df[feature_columns].fillna(0).to_numpy(
    dtype=np.float32
)

y_train = train_df["label"].astype(str).to_numpy()
y_val = val_df["label"].astype(str).to_numpy()

train_classes = set(y_train)
val_classes = set(y_val)

if train_classes != val_classes:
    raise ValueError(
        "Train and validation classes do not match."
    )

if set(train_df["image_path"]) & set(val_df["image_path"]):
    raise ValueError(
        "Same image paths found in train and validation."
    )

print("Train images:", len(X_train))
print("Validation images:", len(X_val))
print("Classes:", len(train_classes))
print("Features:", len(feature_columns))


# ==========================================
# 3. TRAIN MODEL
# ==========================================

print("\nTraining ExtraTrees model...")

model = ExtraTreesClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 4. PREDICTIONS
# ==========================================

print("\nEvaluating model...")

predictions = model.predict(X_val)
probabilities = model.predict_proba(X_val)

accuracy = accuracy_score(y_val, predictions)

macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_val,
    predictions,
    average="weighted",
    zero_division=0
)

top5_indices = np.argsort(
    probabilities,
    axis=1
)[:, -5:]

classes = model.classes_

top5_correct = sum(
    y_val[i] in classes[top5_indices[i]]
    for i in range(len(y_val))
)

top5_accuracy = top5_correct / len(y_val)


# ==========================================
# 5. CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_val,
    predictions,
    zero_division=0
)

class_report = classification_report(
    y_val,
    predictions,
    output_dict=True,
    zero_division=0
)

confusion = confusion_matrix(
    y_val,
    predictions,
    labels=classes
)

confusion_df = pd.DataFrame(
    confusion,
    index=classes,
    columns=classes
)

confusion_file = REPORT_DIR / "bdsl126_rotation_confusion.csv"
confusion_df.to_csv(confusion_file, encoding="utf-8-sig")


# ==========================================
# 6. SAVE MODEL
# ==========================================

joblib.dump(model, MODEL_FILE)

print("\nModel saved:", MODEL_FILE)


# ==========================================
# 7. SAVE REPORT
# ==========================================

report_file = REPORT_DIR / "bdsl126_rotation_report.txt"

with open(report_file, "w", encoding="utf-8") as f:

    f.write("BdSL126 ROTATION-AWARE MODEL REPORT\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Train images: {len(X_train)}\n")
    f.write(f"Validation images: {len(X_val)}\n")
    f.write(f"Classes: {len(train_classes)}\n")
    f.write(f"Features: {len(feature_columns)}\n\n")

    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Top-5 Accuracy: {top5_accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 50 + "\n")
    f.write(report)

    f.write("\n\nWEAKEST CLASSES\n")
    f.write("-" * 50 + "\n")

    class_scores = []

    for label, values in class_report.items():
        if label in train_classes:
            class_scores.append(
                (label, values["f1-score"], values["support"])
            )

    class_scores.sort(key=lambda x: x[1])

    for label, score, support in class_scores[:20]:
        f.write(
            f"{label}: F1={score:.3f}, "
            f"Support={support}\n"
        )


# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("BdSL126 ROTATION MODEL RESULTS")
print("=" * 55)

print(f"Accuracy:       {accuracy * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
print(f"Macro F1:       {macro_f1:.4f}")
print(f"Weighted F1:    {weighted_f1:.4f}")

print("\nWeakest 10 classes:")

for label, score, support in class_scores[:10]:
    print(f"{label}: F1={score:.3f} | Images={support}")

print("\nSaved files:")
print(MODEL_FILE)
print(report_file)
print(confusion_file)

print("\nTRAINING COMPLETED!")
