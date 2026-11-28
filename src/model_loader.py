import json
from pathlib import Path
import joblib
import numpy as np

from src.features import normalize_bdsl, normalize_asl


ROOT = Path(__file__).resolve().parent.parent


class SignModelLoader:
    def __init__(self):
        self.bdsl_alphabet = joblib.load(
            ROOT / "models/bdsl47_alphabet_normalized.joblib"
        )
        self.bdsl_digits = joblib.load(
            ROOT / "models/bdsl47_digits_normalized.joblib"
        )

        asl_dir = ROOT / "external/asl_landmark_project/models"

        self.asl_model = joblib.load(
            asl_dir / "sign_language_model.pkl"
        )
        self.asl_scaler = joblib.load(
            asl_dir / "feature_scaler.pkl"
        )

        with open(asl_dir / "metadata.json", encoding="utf-8") as f:
            self.asl_metadata = json.load(f)

    def predict_bdsl(self, raw_landmarks, mode="alphabet"):
        artifact = (
            self.bdsl_digits if mode == "digits"
            else self.bdsl_alphabet
        )

        features = normalize_bdsl(raw_landmarks).reshape(1, -1)
        model = artifact["model"]

        probabilities = model.predict_proba(features)[0]
        index = int(np.argmax(probabilities))
        label = str(model.classes_[index])

        return {
            "label": label.split("__")[-1],
            "confidence": float(probabilities[index]),
            "language": "BdSL"
        }

    def predict_asl(self, raw_landmarks):
        features = normalize_asl(raw_landmarks).reshape(1, -1)
        scaled = self.asl_scaler.transform(features)

        probabilities = self.asl_model.predict_proba(scaled)[0]
        index = int(np.argmax(probabilities))
        class_id = str(self.asl_model.classes_[index])

        label = self.asl_metadata["label_decoder"].get(
            class_id, class_id
        )

        return {
            "label": label,
            "confidence": float(probabilities[index]),
            "language": "ASL"
        }
