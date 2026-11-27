
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = PROCESSED_DIR / "bdsl126_landmarks_clean_train.csv"
VAL_CSV = PROCESSED_DIR / "bdsl126_landmarks_clean_val.csv"

MODEL_PATH = MODEL_DIR / "bdsl126_landmark_model_clean.joblib"


print("Loading clean landmark datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Train classes:", train_df["label"].nunique())
print("Validation classes:", val_df["label"].nunique())


# ==========================================
# FEATURES AND LABELS
# ==========================================

excluded_columns = ["label", "split", "source"]

feature_columns = [
    col for col in train_df.columns
    if col not in excluded_columns
]

X_train = train_df[feature_columns].astype(np.float32)
y_train = train_df["label"]

X_val = val_df[feature_columns].astype(np.float32)
y_val = val_df["label"]

if feature_columns != [
    col for col in val_df.columns
    if col not in excluded_columns
]:
    raise ValueError("Train/Validation feature columns do not match!")

if set(y_train.unique()) != set(y_val.unique()):
    raise ValueError("Train/Validation class lists do not match!")


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining ExtraTrees model...")

model = ExtraTreesClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ==========================================
# VALIDATION
# ==========================================

print("\nEvaluating model...")

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

report = classification_report(
    y_val,
    predictions,
    zero_division=0
)

matrix = confusion_matrix(
    y_val,
    predictions,
    labels=model.classes_
)

print("\n========== CLEAN MODEL RESULTS ==========")
print("Validation Accuracy:", round(accuracy * 100, 2), "%")
print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Classes:", len(model.classes_))

print("\nClassification Report:")
print(report)


# ==========================================
# SAVE MODEL
# ==========================================

model_data = {
    "model": model,
    "feature_columns": feature_columns,
    "classes": list(model.classes_),
    "normalized": True
}

joblib.dump(model_data, MODEL_PATH)

print("\nModel saved:", MODEL_PATH)


# ==========================================
# SAVE REPORTS
# ==========================================

report_path = REPORT_DIR / "bdsl126_landmark_clean_report.txt"
matrix_path = REPORT_DIR / "bdsl126_landmark_clean_confusion.csv"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("BdSL-126 Clean Landmark Model Report\n\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Train samples: {len(X_train)}\n")
    file.write(f"Validation samples: {len(X_val)}\n")
    file.write(f"Classes: {len(model.classes_)}\n\n")
    file.write(report)

confusion_df = pd.DataFrame(
    matrix,
    index=model.classes_,
    columns=model.classes_
)

confusion_df.to_csv(
    matrix_path,
    encoding="utf-8-sig"
)

print("Report saved:", report_path)
print("Confusion matrix saved:", matrix_path)

print("\nClean model training completed!")
