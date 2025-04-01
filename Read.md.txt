# 🌍 ILR Multilingual Language Assessment App

This AI-powered app evaluates speaking or writing samples using ILR standards across 30+ languages. It auto-detects language, translates to English for analysis, then provides ILR-level feedback translated back to the user's native language — with optional audio playback.

---

## 🔍 ILR Abilities Assessed
- **Functionality** – Communicative purpose
- **Content** – Richness and range
- **Accuracy** – Grammar and structure
- **Context Appropriateness** – Tone and sentiment

---

## ⚙️ Tech Stack
- `streamlit`, `transformers`, `nltk`, `textblob`
- `speechrecognition`, `gTTS`, `langdetect`
- Translation: Helsinki-NLP (MarianMT)
- Feedback: Multilingual BERT for sentiment

---

## 📁 Files in This Folder
- `app.py` – Main application
- `requirements.txt` – Required packages
- `README.md` – This file

---

## ✅ How to Use (When Ready)
```bash
pip install -r requirements.txt
streamlit run app.py
