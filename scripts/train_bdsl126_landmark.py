
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "datasets" / "processed"
MODELS = BASE / "models"
REPORTS = BASE / "reports"

MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

train_path = DATA / "bdsl126_landmarks_train.csv"
val_path = DATA / "bdsl126_landmarks_val.csv"

print("Loading datasets...")

train = pd.read_csv(train_path)
val = pd.read_csv(val_path)

metadata = ["label", "split", "source"]

feature_columns = [
    col for col in train.columns
    if col not in metadata
]

X_train = train[feature_columns].fillna(0)
y_train = train["label"]

X_val = val[feature_columns].fillna(0)
y_val = val["label"]

print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Features:", len(feature_columns))
print("Classes:", y_train.nunique())

print("\nTraining ExtraTrees Model...")

model = ExtraTreesClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nEvaluating Model...")

predictions = model.predict(X_val)

accuracy = accuracy_score(
    y_val,
    predictions
)

print("\nValidation Accuracy:", round(accuracy * 100, 2), "%")

report = classification_report(
    y_val,
    predictions,
    zero_division=0
)

print("\nClassification Report:")
print(report)

report_path = REPORTS / "bdsl126_landmark_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

confusion = confusion_matrix(
    y_val,
    predictions,
    labels=model.classes_
)

pd.DataFrame(
    confusion,
    index=model.classes_,
    columns=model.classes_
).to_csv(
    REPORTS / "bdsl126_landmark_confusion.csv"
)

model_path = MODELS / "bdsl126_landmark_model.joblib"

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "classes": list(model.classes_),
        "normalized": True
    },
    model_path
)

print("\nModel saved:", model_path)
print("Report saved:", report_path)
print("\nTraining completed!")
