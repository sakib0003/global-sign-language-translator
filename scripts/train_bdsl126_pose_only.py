
from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

train = pd.read_csv(
    "datasets/processed/bdsl126_pose_individual_direct_train.csv"
)
val = pd.read_csv(
    "datasets/processed/bdsl126_pose_individual_direct_val.csv"
)

# Pose landmarks: feature_126 to feature_257
# Plus pose_detected flag: feature_259
pose_columns = (
    [f"feature_{i}" for i in range(126, 258)]
    + ["feature_259"]
)

X_train = train[pose_columns].fillna(0)
y_train = train["label"].astype(str)

X_val = val[pose_columns].fillna(0)
y_val = val["label"].astype(str)

print("Training Pose-Only Model...")
print("Features:", len(pose_columns))

model = ExtraTreesClassifier(
    n_estimators=400,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
macro_f1 = f1_score(y_val, predictions, average="macro")
weighted_f1 = f1_score(y_val, predictions, average="weighted")

print("\n======= POSE ONLY RESULTS =======")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

Path("reports").mkdir(exist_ok=True)

with open(
    "reports/bdsl126_pose_only_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    f.write(classification_report(y_val, predictions, zero_division=0))

joblib.dump(
    {"model": model, "feature_columns": pose_columns},
    "models/bdsl126_pose_only_model.joblib"
)

print("\nModel and report saved.")
