import streamlit as st
st.set_page_config(page_title="ILR Multilingual Language Assessment", layout="centered")

import torch
import numpy as np
from pydub import AudioSegment
from transformers import (
    pipeline,
    MarianMTModel,
    MarianTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Tokenizer
)
from textblob import TextBlob, download_corpora
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from langdetect import detect
from gtts import gTTS
import os

# Setup NLTK + TextBlob
nltk_data_dir = os.path.expanduser(os.path.join("~", "nltk_data"))
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
download_corpora.download_all()

# Load sentiment classifier
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_resource
def load_wav2vec_model():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer, model

tokenizer_wav2vec, model_wav2vec = load_wav2vec_model()

# Language code mapping for TTS compatibility
LANG_CODE_MAP = {
    "en": "en", "fr": "fr", "es": "es", "ar": "ar", "zh-cn": "zh", "ru": "ru",
    "pt": "pt", "de": "de", "ja": "ja", "ko": "ko", "it": "it"
}

# ILR Descriptors
ILR_LEVELS = {
    "Functionality": {
        "Emerging": "Performs basic language tasks like greetings and introducing oneself. Can understand simple questions and statements.",
        "Functional": "Handles routine tasks requiring a simple and direct exchange of information on familiar topics. Can describe in simple terms aspects of their background, immediate environment, and basic needs."
    },
    "Content": {
        "Basic": "Uses limited vocabulary and common expressions. Stays within concrete topics.",
        "Moderate": "Discusses topics beyond immediate needs. Shows some ability to connect ideas and express opinions.",
        "Advanced": "Communicates with a wide range of vocabulary. Handles unfamiliar topics with some fluency."
    },
    "Accuracy": {
        "Developing": "Frequent grammatical errors. Meaning is often unclear.",
        "Neutral Accuracy": "Occasional errors. Meaning is generally clear.",
        "Polished": "Grammar and usage are mostly accurate with minor, non-impeding errors.",
        "Imprecise": "Errors sometimes interfere with communication."
    },
    "Context Appropriateness": {
        "Appropriate": "Language is well-suited to the context and audience. Registers and styles are used correctly.",
        "Inappropriate": "Language may be too formal, too informal, or contextually inaccurate. Misuse of expressions common."
    }
}

def generate_detailed_feedback(results):
    return "\n".join([
        f"- **{cat} ({lvl}):** {ILR_LEVELS.get(cat, {}).get(lvl, 'No descriptor available.')}"
        for cat, lvl in results.items()
    ])

def translate(text, src_lang, tgt_lang="en"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def back_translate(text, tgt_lang, src_lang="en"):
    return translate(text, src_lang=src_lang, tgt_lang=tgt_lang)

def assess_ilr_abilities(text):
    punkt_param = PunktParameters()
    tokenizer = PunktSentenceTokenizer(punkt_param)
    sentences = tokenizer.tokenize(text)
    blob = TextBlob(text)

    function_score = sum(1 for s in sentences if len(s.split()) > 10)
    function_level = "Emerging" if function_score < 2 else "Functional"

    word_count = len(blob.words)
    content_level = "Basic"
    if word_count > 60:
        content_level = "Moderate" if blob.sentiment.subjectivity > 0.4 else "Advanced"

    polarity = blob.sentiment.polarity
    accuracy_level = "Developing"
    if -0.2 < polarity < 0.2:
        accuracy_level = "Neutral Accuracy"
    elif polarity >= 0.2:
        accuracy_level = "Polished"
    else:
        accuracy_level = "Imprecise"

    sentiment = classifier(text)[0]
    context_level = "Appropriate" if sentiment['label'].lower().startswith("positive") else "Inappropriate"

    return {
        "Functionality": function_level,
        "Content": content_level,
        "Accuracy": accuracy_level,
        "Context Appropriateness": context_level
    }

def transcribe_audio_file(uploaded_file):
    audio = AudioSegment.from_file(uploaded_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    input_values = tokenizer_wav2vec(samples, return_tensors='pt', padding='longest').input_values
    with torch.no_grad():
        logits = model_wav2vec(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return tokenizer_wav2vec.batch_decode(predicted_ids)[0]

def speak_text(text, lang_code):
    lang = LANG_CODE_MAP.get(lang_code, "en")
    tts = gTTS(text=text, lang=lang)
    tts.save("feedback.mp3")
    os.system("start feedback.mp3" if os.name == "nt" else "afplay feedback.mp3")

# --- Streamlit Interface ---
st.title("🌍 Multilingual ILR Language Assessment Tool")
st.markdown("Evaluate your text or speech based on ILR levels across 30+ languages.")

input_method = st.radio("Choose Input Type", ["Type Text", "Upload Audio File"])
user_input = ""
detected_lang = "en"

if input_method == "Type Text":
    user_input = st.text_area("Enter text:")
    if user_input.strip():
        detected_lang = detect(user_input)
    else:
        st.warning("No input detected. Defaulting to English.")

elif input_method == "Upload Audio File":
    uploaded_audio = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/wav")
        with st.spinner("Transcribing audio..."):
            user_input = transcribe_audio_file(uploaded_audio)
        st.success("Transcription:")
        st.write(user_input)
        detected_lang = detect(user_input) if user_input.strip() else "en"

if st.button("🚀 Analyze"):
    if user_input.strip():
        st.markdown(f"**Detected Language:** `{detected_lang}`")

        with st.spinner("Translating and analyzing..."):
            try:
                translated_text = translate(user_input, src_lang=detected_lang, tgt_lang="en")
                st.markdown("**Translated to English:**")
                st.write(translated_text)
            except Exception as e:
                translated_text = user_input
                st.warning(f"Translation failed: {e}")

            results = assess_ilr_abilities(translated_text)
            st.subheader("ILR Assessment Results:")
            st.markdown("### Detailed ILR Feedback:")
            st.markdown(generate_detailed_feedback(results))

            summary = ", ".join([f"{k} is {v}" for k, v in results.items()])
            try:
                translated_summary = back_translate(summary, tgt_lang=detected_lang)
            except:
                translated_summary = summary

            st.markdown("**Feedback (translated):**")
            st.write(translated_summary)

            speak_text(translated_summary, lang_code=detected_lang.split("-")[0])
    else:
        st.warning("Please input or upload something first.")
