"""Train one static-sign classifier for a language."""
import argparse
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from config_loader import LANGUAGES

parser = argparse.ArgumentParser()
parser.add_argument("--language", required=True, choices=["BdSL", "ASL", "BSL"])
args = parser.parse_args()

cfg = LANGUAGES[args.language]
csv_path = Path(cfg["dataset_file"])
if not csv_path.exists():
    raise SystemExit(f"Dataset missing: {csv_path}. Collect samples first.")

df = pd.read_csv(csv_path).dropna()
if df["label"].nunique() < 2:
    raise SystemExit("At least TWO different sign classes are required.")
X = df.drop(columns=["label"])
y = df["label"].astype(str)
if y.value_counts().min() < 2:
    raise SystemExit("Each class needs at least 2 samples; aim for 100-300+ per class.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, zero_division=0))

model_path = Path(cfg["model_file"])
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)
print("Saved model:", model_path)
