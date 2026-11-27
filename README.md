# Global AI Sign Language Translator (starter project)

A modular Python research prototype based on the uploaded proposal: MediaPipe hand landmarks, OpenCV, a static-sign classifier, Streamlit UI, and optional text-to-speech.

## Important scope note
Sign languages are distinct natural languages. BdSL, ASL, and BSL require separately collected and validated datasets and separately trained models. This starter supports separate language folders/configuration, but it does **not** ship with a sign-language dataset or a pretrained model. The first milestone recognizes static, single-hand classes that you record yourself. It is not yet continuous sentence translation or a production interpreter.

## 1. Requirements
- Python 3.10 or 3.11 recommended
- Webcam
- VS Code (or another editor)
- Internet for installing packages and gTTS speech generation

## 2. Setup (Windows PowerShell)
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
macOS/Linux:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If MediaPipe installation fails, use Python 3.10/3.11 in a fresh virtual environment and reinstall the pinned requirements.

## 3. Collect your first dataset
Start with **one language only**, e.g. ASL, and two clearly distinguishable static signs. Collect at least 100 samples per class, varying distance, angle, lighting, and signer. Do not mix different languages under one model.
```bash
python collect_data.py --language ASL --label hello --samples 120
python collect_data.py --language ASL --label thanks --samples 120
```
A webcam window opens. Press SPACE for each sample; Q quits. Keep labels consistent. The CSV is saved under `data/`.

For BdSL, use signs verified by fluent BdSL users and repeat with `--language BdSL`. Do not assume an ASL sign has the same meaning or form in BdSL.

## 4. Train
```bash
python train.py --language ASL
```
The script prints accuracy, precision, recall, and F1-score, then saves `models/asl_model.joblib`. Collect at least two classes before training.

## 5. Run the web app
```bash
streamlit run app.py
```
Select the matching language, use the camera widget, and inspect the predicted class. The browser may ask for camera permission.

## 6. Expand toward a global system
1. Build separate verified datasets for BdSL, ASL, BSL, etc.
2. Add signer-independent train/test splits (do not let the same person's samples leak into both sets).
3. Add two-hand support and facial/body features where the language requires them.
4. For dynamic signs, record landmark sequences (e.g. 30-60 frames per example) and train an LSTM/GRU or temporal CNN.
5. Add a language-specific vocabulary/grammar layer. A sign classifier alone is not a sentence translator.
6. Evaluate with Deaf/native signers; report per-class metrics, confusion matrix, latency, and failure cases.

## Data ethics
Obtain informed consent before recording people. Avoid collecting names or identifying information unless essential and consented. Store videos securely, allow deletion, and document dataset license and provenance. Do not present this prototype as an authoritative interpreter for medical, legal, or emergency settings.
