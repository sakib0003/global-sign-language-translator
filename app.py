import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from gtts import gTTS

from src.features import extract_raw_hand_landmarks
from src.model_loader import SignModelLoader


st.set_page_config(
    page_title="Global AI Sign Language Translator",
    page_icon="🤟",
    layout="wide"
)

@st.cache_resource
def load_models():
    return SignModelLoader()


models = load_models()

st.title("🤟 Global AI Sign Language Translator")
st.caption("BdSL + ASL | Static Sign Recognition MVP")

language = st.selectbox(
    "Choose Sign Language",
    ["BdSL", "ASL"]
)

mode = "alphabet"

if language == "BdSL":
    mode = st.radio(
        "Recognition Mode",
        ["alphabet", "digits"],
        format_func=lambda x: "Alphabet" if x == "alphabet" else "Digits",
        horizontal=True
    )

st.divider()

picture = st.camera_input("Show your hand sign and capture")

if "sentence" not in st.session_state:
    st.session_state.sentence = ""

if picture:
    image = Image.open(picture).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    raw_landmarks = extract_raw_hand_landmarks(frame)

    if raw_landmarks is None:
        st.error("No hand detected. Improve lighting and try again.")

    else:
        if language == "BdSL":
            result = models.predict_bdsl(raw_landmarks, mode)
        else:
            result = models.predict_asl(raw_landmarks)

        label = result["label"]
        confidence = result["confidence"]

        digit_map = {
            "sign00": "০", "sign01": "১",
            "sign02": "২", "sign03": "৩",
            "sign04": "৪", "sign05": "৫",
            "sign06": "৬", "sign07": "৭",
            "sign08": "৮", "sign09": "৯"
        }

        display_label = (
            digit_map.get(label, label)
            if language == "BdSL" and mode == "digits"
            else label
        )

        st.subheader("Prediction")
        st.title(display_label)
        st.metric("Model Confidence", f"{confidence:.1%}")

        st.caption(
            "Confidence is not a guarantee of correctness. "
            "Real-world performance may differ from validation results."
        )

        if st.button("Add to Sentence"):
            st.session_state.sentence += display_label
            st.rerun()

st.divider()

st.subheader("Sentence Builder")

st.text_area(
    "Your translated sentence",
    key="sentence",
    height=120
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Speak Sentence"):
        if st.session_state.sentence.strip():
            try:
                speech_language = "bn" if language == "BdSL" else "en"
                audio = gTTS(
                    text=st.session_state.sentence,
                    lang=speech_language
                )
                buffer = io.BytesIO()
                audio.write_to_fp(buffer)
                st.audio(buffer.getvalue(), format="audio/mp3")
            except Exception as exc:
                st.error(f"Speech error: {exc}")

with col2:
    if st.button("Clear Sentence"):
        st.session_state.sentence = ""
        st.rerun()

st.caption(
    "Current version: single-sign camera capture. "
    "Continuous live translation and word recognition are future integration steps."
)
