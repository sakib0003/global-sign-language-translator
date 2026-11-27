
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    ROOT, "models", "bdsl126_landmark_model.joblib"
)

CSV_PATH = os.path.join(
    ROOT, "datasets", "processed",
    "bdsl126_landmarks_val.csv"
)

MAPPING_PATH = os.path.join(
    ROOT, "datasets", "processed",
    "bdsl126_bengali_mapping.json"
)

REPORT_PATH = os.path.join(
    ROOT, "reports", "bdsl126_test_predictions.csv"
)

print("\nLoading model and validation data...")

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
feature_columns = model_data["feature_columns"]

df = pd.read_csv(CSV_PATH)

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

X = df[feature_columns].values.astype(np.float32)
y = df["label"].astype(str).values

print("Validation samples:", len(df))
print("Classes:", len(np.unique(y)))

print("\nRunning predictions...")

probabilities = model.predict_proba(X)

predicted_indices = np.argmax(probabilities, axis=1)
predictions = model.classes_[predicted_indices].astype(str)

confidence = np.max(probabilities, axis=1) * 100

accuracy = accuracy_score(y, predictions) * 100

top5_indices = np.argsort(
    probabilities, axis=1
)[:, -5:][:, ::-1]

top5_correct = []

for i, true_label in enumerate(y):
    top5_labels = model.classes_[top5_indices[i]]
    top5_correct.append(true_label in top5_labels)

top5_accuracy = np.mean(top5_correct) * 100

df["actual"] = y
df["predicted"] = predictions
df["actual_bengali"] = [
    mapping.get(label, label) for label in y
]
df["predicted_bengali"] = [
    mapping.get(label, label) for label in predictions
]
df["confidence_percent"] = np.round(confidence, 2)
df["correct"] = df["actual"] == df["predicted"]

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

print("\n========== RESULTS ==========")
print(f"Top-1 Accuracy: {accuracy:.2f}%")
print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
print("Correct:", int(df["correct"].sum()))
print("Wrong:", int((~df["correct"]).sum()))

print("\n========== 20 SAMPLE PREDICTIONS ==========")

sample = df.sample(
    n=min(20, len(df)),
    random_state=42
)

for _, row in sample.iterrows():
    status = "CORRECT" if row["correct"] else "WRONG"

    print("\n----------------------------")
    print("Actual:", row["actual_bengali"])
    print("Predicted:", row["predicted_bengali"])
    print("Confidence:", row["confidence_percent"], "%")
    print("Result:", status)

print("\nFull report saved:")
print(REPORT_PATH)
