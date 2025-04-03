import streamlit as st
import numpy as np
import os
from textblob import TextBlob, download_corpora
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from langdetect import detect
from gtts import gTTS
from transformers import MarianMTModel, MarianTokenizer

# Setup
nltk_data_dir = os.path.expanduser(os.path.join("~", "nltk_data"))
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
download_corpora.download_all()

LANG_CODE_MAP = {
    "en": "en", "fr": "fr", "es": "es", "ar": "ar", "zh-cn": "zh", "ru": "ru",
    "pt": "pt", "de": "de", "ja": "ja", "ko": "ko", "it": "it"
}

def get_sentiment_label(blob):
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

def translate(text, src_lang, tgt_lang="en"):
    try:
        # Some environments need fixed model names
        if src_lang == "ar" and tgt_lang == "en":
            model_name = "Helsinki-NLP/opus-mt-ar-en"
        else:
            model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        tokens = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception:
        try:
            return str(TextBlob(text).translate(from_lang=src_lang, to=tgt_lang))
        except Exception:
            return text

def summarize_text(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    if not sentences:
        return "No summary available."
    return " ".join(str(s) for s in sentences[:2])

def generate_ilr_level(text_blob, sentences, sentiment_label):
    wc = len(text_blob.words)
    sentence_count = len(sentences)
    subjectivity = text_blob.sentiment.subjectivity
    polarity = text_blob.sentiment.polarity

    level = 1
    if wc > 150 and sentence_count > 5:
        if subjectivity < 0.5 and abs(polarity) < 0.4:
            level = 3
        if wc > 250:
            level = 4
        if wc > 300 and polarity > 0.2 and subjectivity < 0.3:
            level = 5
    elif wc > 80:
        level = 2
    return min(level, 5)

def speak_text(text, lang_code):
    lang = LANG_CODE_MAP.get(lang_code, "en")
    tts = gTTS(text=text, lang=lang)
    tts.save("feedback.mp3")
    os.system("start feedback.mp3" if os.name == "nt" else "afplay feedback.mp3")

# Streamlit Interface
st.title("Multilingual ILR Language Assessment Tool")
st.markdown("Detect language, translate, summarize key ideas, and assign an ILR level (1–5).")

user_input = st.text_area("Enter your text (any language):")
detected_lang = detect(user_input) if user_input.strip() else "en"

if st.button("Analyze"):
    if user_input.strip():
        st.markdown(f"**Detected Language:** `{detected_lang}`")

        with st.spinner("Translating and analyzing..."):
            if detected_lang != "en":
                translated_text = translate(user_input, src_lang=detected_lang, tgt_lang="en")
                st.markdown("**Translated to English:**")
                st.write(translated_text)
                if translated_text.strip() == user_input.strip():
                    st.warning("Translation failed. Using original input.")
            else:
                translated_text = user_input
                st.info("Input is in English — skipping translation.")

            summary = summarize_text(translated_text)
            st.markdown("**Summary of Key Ideas:**")
            st.write(summary)

            blob = TextBlob(translated_text)
            punkt_param = PunktParameters()
            tokenizer = PunktSentenceTokenizer(punkt_param)
            sentences = tokenizer.tokenize(translated_text)
            sentiment_label = get_sentiment_label(blob)
            ilr_level = generate_ilr_level(blob, sentences, sentiment_label)

        st.subheader("ILR Assessment Result (Overall Level):")
        st.markdown(f"- **Estimated ILR Level:** {ilr_level}")
        rationale = f"The content exhibits characteristics of ILR Level {ilr_level} based on word count, structural range, and coherence."
        st.markdown("**Rationale:** " + rationale)

        translated_summary = translate(
            f"Estimated ILR Level is {ilr_level}. Summary: {summary}. Rationale: {rationale}",
            src_lang="en", tgt_lang=detected_lang
        )
        st.markdown("**Feedback (translated):**")
        st.write(translated_summary)

        speak_text(translated_summary, lang_code=detected_lang.split("-")[0])
    else:
        st.warning("Please enter some text to analyze.")
