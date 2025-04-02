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

# Setup for NLTK + TextBlob corpora
nltk_data_dir = os.path.expanduser(os.path.join("~", "nltk_data"))
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
download_corpora.download_all()

# Load multilingual sentiment model
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Load Wav2Vec2 model once
@st.cache_resource
def load_wav2vec_model():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer, model

tokenizer_wav2vec, model_wav2vec = load_wav2vec_model()

# Translate
def translate(text, src_lang, tgt_lang="en"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def back_translate(text, tgt_lang, src_lang="en"):
    return translate(text, src_lang=src_lang, tgt_lang=tgt_lang)

# ILR scoring logic
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

# Transcription using pydub instead of torchaudio
def transcribe_audio_file(uploaded_file):
    audio = AudioSegment.from_file(uploaded_file, format="wav")
    audio = audio.set_channels(1).set_frame_rate(16000)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    input_values = tokenizer_wav2vec(samples, return_tensors='pt', padding='longest').input_values

    with torch.no_grad():
        logits = model_wav2vec(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer_wav2vec.batch_decode(predicted_ids)[0]

    return transcription

# Text-to-speech
def speak_text(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    tts.save("feedback.mp3")
    os.system("start feedback.mp3" if os.name == "nt" else "afplay feedback.mp3")

# Streamlit app interface
st.title("🌍 Multilingual ILR Language Assessment Tool")
st.markdown("Evaluate your text or speech based on ILR levels across 30+ languages.")

input_method = st.radio("Choose Input Type", ["Type Text", "Upload WAV File"])
user_input = ""
detected_lang = "en"

if input_method == "Type Text":
    user_input = st.text_area("Enter text:")
    if user_input.strip():
        detected_lang = detect(user_input)
    else:
        detected_lang = "en"
        st.warning("No valid input to detect language. Defaulting to English.")

elif input_method == "Upload WAV File":
    uploaded_audio = st.file_uploader("Upload a WAV file only", type=["wav"])
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        user_input = transcribe_audio_file(uploaded_audio)
        st.success("Transcription:")
        st.write(user_input)
        if user_input.strip():
            detected_lang = detect(user_input)
        else:
            detected_lang = "en"
            st.warning("No valid input to detect language. Defaulting to English.")

if st.button("🚀 Analyze"):
    if user_input.strip():
        st.markdown(f"**Detected Language:** `{detected_lang}`")

        if detected_lang == "en":
            translated_text = user_input
            st.info("Input is in English — no translation needed.")
        else:
            try:
                translated_text = translate(user_input, src_lang=detected_lang, tgt_lang="en")
            except:
                st.warning("Translation model not available — using original input.")
                translated_text = user_input

        st.markdown("**Translated to English:**")
        st.write(translated_text)

        results = assess_ilr_abilities(translated_text)
        st.subheader("ILR Assessment Results:")
        for k, v in results.items():
            st.markdown(f"- **{k}:** {v}")

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
