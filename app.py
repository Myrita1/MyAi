import streamlit as st
import numpy as np
import os
from textblob import TextBlob, download_corpora
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from langdetect import detect
from deep_translator import GoogleTranslator

# Setup
nltk_data_dir = os.path.expanduser(os.path.join("~", "nltk_data"))
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
download_corpora.download_all()

def fix_lang_code(code):
    substitutions = {
        "zh-cn": "chinese (simplified)",
        "zh": "chinese (simplified)",
        "zh-tw": "chinese (traditional)",
        "he": "iw",  # Hebrew
        "jw": "jv",  # Javanese
        "fil": "tl", # Filipino
        "nb": "no",  # Norwegian Bokmål
        "ar": "arabic",
        "en": "english"
    }
    return substitutions.get(code, code)

def translate(text, src_lang, tgt_lang="en"):
    try:
        return GoogleTranslator(source=fix_lang_code(src_lang), target=fix_lang_code(tgt_lang)).translate(text)
    except Exception as e:
        st.warning(f"Translation failed. Using original input. ({e})")
        return text

def summarize_text(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    return " ".join(str(s) for s in sentences[:2]) if sentences else "No summary available."

def generate_ilr_level(blob, sentences):
    wc = len(blob.words)
    sc = len(sentences)
    subj = blob.sentiment.subjectivity
    polar = blob.sentiment.polarity

    level = 1
    if wc > 150 and sc > 5:
        level = 3
        if wc > 250:
            level = 4
        if wc > 300 and polar > 0.2 and subj < 0.3:
            level = 5
    elif wc > 80:
        level = 2
    return level

def get_rationale(level, blob, sentences):
    wc = len(blob.words)
    sc = len(sentences)
    vocab = len(set(blob.words))
    subj = blob.sentiment.subjectivity
    polar = blob.sentiment.polarity

    return (
        f"The ILR Level {level} was assigned based on the following observed features:\n\n"
        f"- **Word Count**: {wc} words indicate a {'short' if wc < 80 else 'moderate' if wc < 150 else 'extensive'} text.\n"
        f"- **Sentence Count**: {sc} sentence(s) suggesting {'limited' if sc <= 2 else 'developed'} structural complexity.\n"
        f"- **Vocabulary Variety**: {vocab} unique words reflect a {'basic' if vocab < 50 else 'moderate' if vocab < 120 else 'rich'} lexicon.\n"
        f"- **Subjectivity/Polarity**: ({subj:.2f}/{polar:.2f}) suggesting the tone is {'neutral' if -0.2 <= polar <= 0.2 else 'opinionated'} "
        f"and the focus is {'informational' if subj < 0.4 else 'personal'}.\n"
        f"- **Overall**: The text demonstrates features aligning with ILR Level {level} expectations for reading proficiency."
    )

# --- Streamlit Interface ---
st.title("Multilingual ILR Language Assessment Tool (Reading Only)")
st.markdown("Detect language, translate, summarize key ideas, and assign an ILR reading level (1–5).")

user_input = st.text_area("Enter your text (any language):")
detected_lang = detect(user_input) if user_input.strip() else "en"

if st.button("Analyze"):
    if user_input.strip():
        st.markdown(f"**Detected Language:** `{detected_lang}`")

        with st.spinner("Processing..."):
            translated_text = translate(user_input, src_lang=detected_lang, tgt_lang="en")
            st.markdown("**Translated to English:**")
            st.write(translated_text)

            summary = summarize_text(translated_text)
            st.markdown("**Summary of Key Ideas:**")
            st.write(summary)

            blob = TextBlob(translated_text)
            punkt_param = PunktParameters()
            tokenizer = PunktSentenceTokenizer(punkt_param)
            sentences = tokenizer.tokenize(translated_text)
            ilr_level = generate_ilr_level(blob, sentences)

        st.subheader("ILR Assessment Result (Reading Proficiency):")
        st.markdown(f"- **Estimated ILR Level:** {ilr_level}")
        st.markdown("**Detailed Justification:**")
        st.markdown(get_rationale(ilr_level, blob, sentences))
    else:
        st.warning("Please input text to analyze.")
