
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

BASE = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE / "datasets/processed/bdsl126_pose_landmarks_train.csv"
VAL_PATH = BASE / "datasets/processed/bdsl126_pose_landmarks_val.csv"

MODEL_PATH = BASE / "models/bdsl126_pose_model.joblib"
REPORT_PATH = BASE / "reports/bdsl126_pose_report.txt"
CONFUSION_PATH = BASE / "reports/bdsl126_pose_confusion.csv"

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

# Same feature columns for both datasets
feature_columns = [
    col for col in train_df.columns
    if col not in ["label", "split", "source"]
]

X_train = train_df[feature_columns].to_numpy(dtype=np.float32)
y_train = train_df["label"].to_numpy()

X_val = val_df[feature_columns].to_numpy(dtype=np.float32)
y_val = val_df["label"].to_numpy()

print("\nDataset information")
print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Features:", len(feature_columns))
print("Train classes:", len(np.unique(y_train)))
print("Validation classes:", len(np.unique(y_val)))

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

print("\nEvaluating model...")

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

report = classification_report(
    y_val,
    predictions,
    zero_division=0
)

print("\n================================")
print("POSE MODEL RESULTS")
print("================================")

print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(report)

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "classes": model.classes_.tolist(),
        "normalized": True
    },
    MODEL_PATH
)

# Save report
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Validation Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(report)

# Save confusion matrix
cm = confusion_matrix(
    y_val,
    predictions,
    labels=model.classes_
)

cm_df = pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
)

cm_df.to_csv(CONFUSION_PATH)

print("\nSaved files:")
print("Model:", MODEL_PATH)
print("Report:", REPORT_PATH)
print("Confusion Matrix:", CONFUSION_PATH)

print("\nTraining completed.")
