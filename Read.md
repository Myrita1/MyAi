# Step 4: Write the updated README.md file
readme_md = """
# Multilingual ILR Language Assessment Tool

This app evaluates spoken or written language samples using ILR proficiency levels (1–5), with automatic language detection, translation into English, summarization of main ideas, and rationale behind ILR level assignment.

---

## Features

- Language detection
- English translation (MarianMT + fallback to TextBlob)
- Summarization using `facebook/bart-large-cnn`
- ILR-level scoring logic with rationale
- Audio input via `wav` and transcription using Wav2Vec2
- Text-to-speech playback of feedback

---

## Tech Stack

- Streamlit for UI
- Transformers (`facebook/bart-large-cnn`, `nlptown/bert-base-multilingual-uncased-sentiment`)
- Wav2Vec2 for audio transcription
- TextBlob + NLTK for text analysis
- MarianMT + fallback translation

---

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
