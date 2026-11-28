
from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

TRAIN_CSV = Path(
    "datasets/processed/bdsl126_pose_individual_direct_train.csv"
)
VAL_CSV = Path(
    "datasets/processed/bdsl126_pose_individual_direct_val.csv"
)

MODEL_PATH = Path("models/bdsl126_individual_pose_model.joblib")
REPORT_PATH = Path("reports/bdsl126_individual_pose_report.txt")
CONFUSION_PATH = Path("reports/bdsl126_individual_pose_confusion.csv")

print("Loading datasets...")
train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)

feature_columns = [f"feature_{i}" for i in range(260)]

for column in feature_columns:
    if column not in train.columns or column not in val.columns:
        raise ValueError(f"Missing feature column: {column}")

X_train = train[feature_columns].fillna(0)
y_train = train["label"].astype(str)

X_val = val[feature_columns].fillna(0)
y_val = val["label"].astype(str)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Training classes:", y_train.nunique())
print("Validation classes:", y_val.nunique())

missing_classes = set(y_val.unique()) - set(y_train.unique())
if missing_classes:
    raise ValueError(f"Validation classes missing in training: {missing_classes}")

print("\nTraining ExtraTrees model...")

model = ExtraTreesClassifier(
    n_estimators=400,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nEvaluating model...")
predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
macro_f1 = f1_score(y_val, predictions, average="macro")
weighted_f1 = f1_score(y_val, predictions, average="weighted")

report = classification_report(
    y_val,
    predictions,
    zero_division=0
)

top5_correct = 0
probabilities = model.predict_proba(X_val)
classes = model.classes_

for index, actual in enumerate(y_val):
    top_indices = probabilities[index].argsort()[-5:]
    top_labels = classes[top_indices]

    if actual in top_labels:
        top5_correct += 1

top5_accuracy = top5_correct / len(y_val)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "classes": list(model.classes_)
    },
    MODEL_PATH
)

cm = confusion_matrix(y_val, predictions, labels=classes)
pd.DataFrame(
    cm,
    index=classes,
    columns=classes
).to_csv(CONFUSION_PATH)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("BdSL126 Individual Image Pose + Hand Model\n")
    file.write("=" * 55 + "\n")
    file.write(f"Training images: {len(X_train)}\n")
    file.write(f"Validation images: {len(X_val)}\n")
    file.write(f"Classes: {len(classes)}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Top-5 Accuracy: {top5_accuracy:.4f}\n")
    file.write(f"Macro F1: {macro_f1:.4f}\n")
    file.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    file.write(report)

print("\n========== RESULTS ==========")
print(f"Accuracy:       {accuracy * 100:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
print(f"Macro F1:       {macro_f1:.4f}")
print(f"Weighted F1:   {weighted_f1:.4f}")

print("\nSaved model:", MODEL_PATH)
print("Saved report:", REPORT_PATH)
print("Saved confusion matrix:", CONFUSION_PATH)
