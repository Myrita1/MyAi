# 🌍 ILR Multilingual Language Assessment App

This AI-powered app evaluates speaking or writing samples using ILR (Interagency Language Roundtable) standards across **30+ languages**. It automatically detects the language, translates it into English for analysis, and provides detailed feedback mapped to ILR levels. It even translates the feedback back into the user's language — with **optional audio playback**.

---

## 🔍 ILR Abilities Assessed
- **Functionality** – Communicative purpose and range
- **Content** – Richness, elaboration, and topical breadth
- **Accuracy** – Grammar, precision, and structural correctness
- **Context Appropriateness** – Style, register, tone, and sentiment

---

## ⚙️ Tech Stack
- `streamlit` – Web interface
- `transformers` – Language translation & sentiment analysis (MarianMT, Multilingual BERT)
- `torch` – Model inference
- `pydub` – Audio processing
- `textblob` – Sentiment and grammatical insights
- `nltk` – Sentence tokenization
- `gTTS` – Text-to-speech audio feedback
- `langdetect` – Language identification

---

## 📁 Files in This Folder
- `app.py` – Main application logic
- `requirements.txt` – All necessary Python dependencies
- `README.md` – You're reading it!
- `.gitignore` – Files to ignore when uploading to version control

---

## ✅ How to Use
Clone the repository and install the requirements:

```bash
pip install -r requirements.txt
