import streamlit as st
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

# Fix misdetected or unsupported language codes
def fix_lang_code(code):
    substitutions = {
        "zh-cn": "chinese (simplified)",
        "zh": "chinese (simplified)",
        "zh-tw": "chinese (traditional)",
        "he": "iw",  # Hebrew
        "jw": "jv",
        "fil": "tl",
        "nb": "no",
        "ar": "arabic",
        "en": "english"
    }
    return substitutions.get(code, code)

def translate(text, src_lang, tgt_lang="en"):
    try:
        fixed_src = fix_lang_code(src_lang)
        fixed_tgt = fix_lang_code(tgt_lang)
        return GoogleTranslator(source=fixed_src, target=fixed_tgt).translate(text)
    except Exception as e:
        st.warning(f"Translation failed. Using original input. ({e})")
        return text

def summarize_text(text):
    blob = TextBlob(text)
    sentences = blob.sentences
    return " ".join(str(s) for s in sentences[:2]) if sentences else "No summary available."

def get_ilr_level(wc, sc, vocab):
    if wc < 50 or sc < 2:
        return 1
    elif wc < 150 or vocab < 100:
        return 2
    elif wc < 300 or vocab < 150:
        return 3
    elif wc < 500 or vocab < 250:
        return 4
    else:
        return 5

def get_ilr_rationale(level):
    rationales = {
        1: "Text displays simple structure, short sentences, and limited vocabulary. Likely understood by someone who can read signs, forms, and common labels.",
        2: "Text includes connected ideas with moderate vocabulary. Suitable for readers who can understand routine news, descriptions, and messages.",
        3: "Text demonstrates good organization, paragraph development, and range of vocabulary. Suitable for readers who handle general professional reading.",
        4: "Text shows advanced structure, with abstract or technical language. Suitable for readers in academic or formal workplace contexts.",
        5: "Text reflects near-native reading ability — stylistic nuance, idiomatic precision, and full cultural fluency across genres."
    }
    return rationales.get(level, "No rationale available.")

# --- Streamlit UI ---
st.title("Multilingual ILR Language Assessment Tool (Reading Only)")
st.markdown("Detect language, translate to English, summarize key ideas, and assign an ILR Reading Level (1–5).")

user_input = st.text_area("Enter your text (any language):")

if st.button("Analyze"):
    if user_input.strip():
        detected_lang = detect(user_input)

        # Fix Arabic falsely detected as Hebrew
        if detected_lang == "he" and "ال" in user_input:
            detected_lang = "ar"

        st.markdown(f"**Detected Language:** `{detected_lang}`")

        with st.spinner("Translating and analyzing..."):
            translated_text = translate(user_input, src_lang=detected_lang, tgt_lang="en")
            st.markdown("**Translated to English:**")
            st.write(translated_text)

            blob = TextBlob(translated_text)
            punkt_param = PunktParameters()
            tokenizer = PunktSentenceTokenizer(punkt_param)
            sentences = tokenizer.tokenize(translated_text)

            summary = summarize_text(translated_text)
            wc = len(blob.words)
            vocab = len(set(blob.words))
            sc = len(sentences)
            ilr_level = get_ilr_level(wc, sc, vocab)

            st.markdown("**Summary of Key Ideas:**")
            st.write(summary)

        st.subheader("ILR Assessment Result (Reading Proficiency):")
        st.markdown(f"- **Estimated ILR Level:** {ilr_level}")
        st.markdown("**Detailed Justification (based on ILR Reading Guidelines):**")
        st.markdown(get_ilr_rationale(ilr_level))
    else:
        st.warning("Please input text to analyze.")
