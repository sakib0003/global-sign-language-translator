
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from sklearn.model_selection import train_test_split

# Project root path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from landmark_utils import normalize_landmarks

CSV_PATH = os.path.join(ROOT, "datasets", "processed", "bdsl47_landmarks.csv")
MODEL_PATH = os.path.join(ROOT, "models", "bdsl47_alphabet_normalized.joblib")
OUTPUT_DIR = os.path.join(ROOT, "reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset and model...")

df = pd.read_csv(CSV_PATH)
data = joblib.load(MODEL_PATH)

model = data["model"]
features = data["feature_columns"]

alphabet = df[
    df["label"].astype(str).str.contains("Sign Alphabets", case=False)
].copy()

if alphabet.empty:
    raise ValueError("Alphabet dataset not found. Check CSV label column.")

# Same signer-independent split as training
signer_column = "signer"

if signer_column not in alphabet.columns:
    raise ValueError("Signer column not found in CSV.")

signers = sorted(alphabet[signer_column].unique())
test_signers = set(signers[::5])

test_df = alphabet[
    alphabet[signer_column].isin(test_signers)
].copy()

X_test = test_df[features].values
y_test = test_df["label"].astype(str).values

print("Normalizing test landmarks...")

X_test_normalized = np.array(
    [normalize_landmarks(row) for row in X_test],
    dtype=np.float32
)

print("Generating predictions...")

y_pred = model.predict(X_test_normalized)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

labels = sorted(model.classes_)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

csv_output = os.path.join(
    OUTPUT_DIR,
    "bdsl_alphabet_confusion_matrix.csv"
)

cm_df.to_csv(csv_output)

# Find most confused classes
errors = []

for i, true_label in enumerate(labels):
    for j, predicted_label in enumerate(labels):
        if i != j and cm[i, j] > 0:
            errors.append({
                "True Class": true_label.split("__")[-1],
                "Predicted As": predicted_label.split("__")[-1],
                "Wrong Predictions": int(cm[i, j])
            })

errors_df = pd.DataFrame(errors)

if not errors_df.empty:
    errors_df = errors_df.sort_values(
        "Wrong Predictions",
        ascending=False
    )

    print("\nTop 20 Confusion Pairs:")
    print(errors_df.head(20).to_string(index=False))

    errors_df.to_csv(
        os.path.join(OUTPUT_DIR, "bdsl_alphabet_top_errors.csv"),
        index=False
    )

# Plot confusion matrix
plt.figure(figsize=(18, 15))

plt.imshow(cm, interpolation="nearest", aspect="auto")
plt.title("BdSL Alphabet Confusion Matrix")
plt.colorbar()

short_labels = [label.split("__")[-1] for label in labels]

plt.xticks(
    np.arange(len(labels)),
    short_labels,
    rotation=90
)

plt.yticks(
    np.arange(len(labels)),
    short_labels
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()

image_output = os.path.join(
    OUTPUT_DIR,
    "bdsl_alphabet_confusion_matrix.png"
)

plt.savefig(image_output, dpi=180)
plt.show()

print("\nAnalysis completed!")
print("Saved:", csv_output)
print("Saved:", image_output)
print("Reports folder:", OUTPUT_DIR)
