"""Streamlit demo: upload/capture a hand-sign image and classify it."""
from pathlib import Path
import io
import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from gtts import gTTS
from src.features import extract_hand_features
from config_loader import LANGUAGES

st.set_page_config(page_title="Global Sign Language Translator", page_icon="🤟")
st.title("🤟 Global AI Sign Language Translator")
st.caption("Research prototype — static single-hand signs; language-specific models.")

language = st.selectbox(
    "Choose sign language",
    list(LANGUAGES.keys()),
    format_func=lambda key: LANGUAGES[key]["display_name"]
)
model_path = Path(LANGUAGES[language]["model_file"])
if not model_path.exists():
    st.warning(f"No trained model for {language}. Collect data and run train.py first.")
    st.stop()

model = joblib.load(model_path)
picture = st.camera_input("Show one sign to the camera")
if picture:
    image = Image.open(picture).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    features = extract_hand_features(frame)
    if features is None:
        st.error("No hand detected. Improve lighting and keep the hand inside the frame.")
    else:
        probabilities = model.predict_proba([features])[0]
        best_index = int(np.argmax(probabilities))
        label = str(model.classes_[best_index])
        confidence = float(probabilities[best_index])
        st.subheader(f"Prediction: {label}")
        st.metric("Model confidence", f"{confidence:.1%}")
        st.caption("Confidence is not a guarantee of correctness. Test with unseen signers.")
        if st.button("🔊 Speak prediction"):
            try:
                audio = gTTS(text=label, lang=LANGUAGES[language]["tts_language"])
                buffer = io.BytesIO()
                audio.write_to_fp(buffer)
                st.audio(buffer.getvalue(), format="audio/mp3")
            except Exception as exc:
                st.error(f"Speech generation failed: {exc}")
