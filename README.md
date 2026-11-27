# Global AI Sign Language Translator

An AI-powered sign language recognition project, starting with **Bangladesh Sign Language (BdSL)**, with a long-term vision of supporting multiple sign languages worldwide.

## Project Overview

Global AI Sign Language Translator aims to recognize sign language gestures using computer vision and machine learning.

The project currently focuses on three recognition tasks:

- **Alphabet Recognition:** 37 BdSL alphabet classes.
- **Digit Recognition:** 10 BdSL digit classes.
- **Word Recognition:** 126 BdSL word classes.

The long-term goal is to develop a system that can help bridge communication gaps between Deaf and hearing communities.

## Current Development Status

| Feature | Status |
|---|---|
| BdSL Alphabet Recognition | Implemented |
| BdSL Digit Recognition | Implemented |
| BdSL Word Recognition | Implemented |
| Hand Landmark Extraction | Implemented |
| Pose Landmark Extraction | Implemented |
| Webcam Recognition | In development |
| Sentence Builder | Planned |
| Bengali Text-to-Speech | Planned |
| English Translation | Planned |
| ASL Support | Future Development |

## Technology Stack

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- YOLO
- NumPy
- Pandas
- Joblib

## Machine Learning Models

### 1. BdSL Alphabet Recognition

- Dataset: BdSL47
- Classes: 37
- Features: 63 hand landmark coordinates
- Model: Support Vector Machine (RBF)
- Signer-independent test accuracy: 80.69%

### 2. BdSL Digit Recognition

- Dataset: BdSL47
- Classes: 10
- Features: Normalized hand landmarks
- Model: Machine Learning classifier
- Signer-independent test accuracy: 96%

### 3. BdSL Word Recognition

- Dataset: BdSL126
- Classes: 126
- Features: Hand and body pose landmarks
- Model: ExtraTrees Classifier
- Reported validation accuracy: 95.82%

**Evaluation Note:** The word recognition result comes from the dataset's existing train/validation split. A frame-group overlap audit found related frame sequences in both splits. Therefore, this score should not be interpreted as independent signer-level performance or real-world webcam accuracy. Further evaluation is required.

## Datasets

The project uses the following research datasets:

- **BdSL47:** [Mendeley Data – BdSL47](https://data.mendeley.com/datasets/pbb3w3f92y/3)
- **BdSL126:** [Mendeley Data – BdSL126](https://data.mendeley.com/datasets/39j93bttsr/1)

Please follow the original dataset licenses and usage conditions.

Datasets and trained model files are not included in this GitHub repository.

## Project Structure

```text
global_sign_language_translator/
│
├── app.py
├── collect_data.py
├── train.py
├── config_loader.py
├── requirements.txt
├── README.md
│
├── config/
│   └── languages.json
│
├── scripts/
│   ├── landmark_utils.py
│   ├── prepare_bdsl47.py
│   ├── train_bdsl47.py
│   ├── train_bdsl126_pose_model.py
│   ├── unified_bdsl_webcam.py
│   └── other training and evaluation scripts
│
├── src/
│   └── features.py
│
└── reports/
    └── model evaluation reports
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sakib0003/global-sign-language-translator.git
cd global-sign-language-translator
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download datasets

Download the BdSL47 and BdSL126 datasets from their original sources and place them in the appropriate dataset folders.

### 5. Train models

Use the relevant scripts in the `scripts/` directory to prepare datasets, extract landmarks, and train models.

### 6. Run webcam recognition

```powershell
python scripts/unified_bdsl_webcam.py
```

Select a recognition mode from the application.

Note: Required datasets and trained model files must be prepared locally before running the corresponding recognition features.

## Important Limitations

- Current development focuses on static sign recognition.
- Recognition accuracy can vary with lighting, camera position, background, and signer.
- Dataset validation accuracy does not guarantee real-world performance.
- Continuous sign language and full sentence translation are not yet complete.
- BdSL, ASL, and other sign languages require separate datasets and language-specific models.

## Future Roadmap

1. Improve BdSL alphabet and digit recognition.
2. Improve word recognition using independent signer and recording splits.
3. Develop a Bengali sentence builder.
4. Add Bengali text-to-speech.
5. Explore English translation.
6. Expand support to ASL and other sign languages.
7. Evaluate the system with Deaf and fluent sign language users.

## Ethical Considerations

Sign languages are complete natural languages, not simply collections of hand gestures.

The project aims to support accessibility and communication. Dataset collection should respect consent, privacy, licensing, and the linguistic knowledge of Deaf communities.

This project is a research and development prototype, not an authoritative interpreter for medical, legal, or emergency communication.

## Contributing

Contributions, suggestions, research collaboration, and feedback are welcome.

## License

A license has not yet been selected. Please contact the repository owner before redistributing or reusing project materials.

---

**Project:** Global AI Sign Language Translator  
**Initial Language:** Bangladesh Sign Language (BdSL)  
**GitHub:** https://github.com/sakib0003/global-sign-language-translator