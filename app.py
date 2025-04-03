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
import tempfile

nltk_data_dir = os.path.expanduser(os.path.join("~", "nltk_data"))
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
download_corpora.download_all()

classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_resource
def load_wav2vec_model():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer, model

tokenizer_wav2vec, model_wav2vec = load_wav2vec_model()

LANG_CODE_MAP = {
    "en": "en", "fr": "fr", "es": "es", "ar": "ar", "zh-cn": "zh", "ru": "ru",
    "pt": "pt", "de": "de", "ja": "ja", "ko": "ko", "it": "it"
}

def translate(text, src_lang, tgt_lang="en"):
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        tokens = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except:
        try:
            return str(TextBlob(text).translate(from_lang=src_lang, to=tgt_lang))
        except:
            return text

def back_translate(text, tgt_lang, src_lang="en"):
    return translate(text, src_lang=src_lang, tgt_lang=tgt_lang)

def generate_ilr_numeric_levels(text_blob, sentences, sentiment_label):
    if len(sentences) < 2:
        functionality_score = 1
    elif len(sentences) < 5:
        functionality_score = 2
    elif any(len(s.split()) > 12 for s in sentences):
        functionality_score = 3
    else:
        functionality_score = 4 if len(sentences) >= 5 else 2

    word_count = len(text_blob.words)
    subjectivity = text_blob.sentiment.subjectivity
    if word_count < 30:
        content_score = 1
    elif word_count < 60:
        content_score = 2
    elif subjectivity > 0.3:
        content_score = 4
    else:
        content_score = 3

    polarity = text_blob.sentiment.polarity
    if polarity < -0.3:
        accuracy_score = 1
    elif -0.3 <= polarity < -0.1:
        accuracy_score = 2
    elif -0.1 <= polarity <= 0.1:
        accuracy_score = 3
    elif 0.1 < polarity <= 0.3:
        accuracy_score = 4
    else:
        accuracy_score = 5

    if sentiment_label.lower().startswith("positive"):
        context_score = 4
    elif "neutral" in sentiment_label.lower():
        context_score = 3
    elif "negative" in sentiment_label.lower():
        context_score = 2
    else:
        context_score = 1

    return {
        "Functionality": functionality_score,
        "Content": content_score,
        "Accuracy": accuracy_score,
        "Context Appropriateness": context_score
    }

def transcribe_audio_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    audio = AudioSegment.from_file(tmp_path, format="wav")
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
st.title("Multilingual ILR Language Assessment Tool")
st.markdown("Get an ILR proficiency level (1–5) and a clear explanation of why.")

input_method = st.radio("Choose Input Type", ["Type Text", "Upload Audio File"])
user_input = ""
detected_lang = "en"

if input_method == "Type Text":
    user_input = st.text_area("Enter text:")
    if user_input.strip():
        detected_lang = detect(user_input)
elif input_method == "Upload Audio File":
    uploaded_audio = st.file_uploader("Upload a WAV file", type=["wav"])
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/wav")
        with st.spinner("Transcribing audio..."):
            user_input = transcribe_audio_file(uploaded_audio)
        st.success("Transcription:")
        st.write(user_input)
        detected_lang = detect(user_input) if user_input.strip() else "en"

if st.button("Analyze"):
    if user_input.strip():
        st.markdown(f"**Detected Language:** `{detected_lang}`")

        with st.spinner("Translating and analyzing..."):
            translated_text = translate(user_input, src_lang=detected_lang, tgt_lang="en")
            st.markdown("**Translated to English:**")
            st.write(translated_text)

            blob = TextBlob(translated_text)
            punkt_param = PunktParameters()
            sentence_tokenizer = PunktSentenceTokenizer(punkt_param)
            sentences = sentence_tokenizer.tokenize(translated_text)
            trimmed_text = " ".join(translated_text.split()[:500])
            sentiment = classifier(trimmed_text)[0]
            ilr_scores = generate_ilr_numeric_levels(blob, sentences, sentiment["label"])

        st.subheader("ILR Assessment Result (Overall Level):")
        overall_score = round(sum(ilr_scores.values()) / len(ilr_scores))

        if overall_score == 1:
            rationale = "Very basic sentence structure and vocabulary. Likely limited to survival phrases."
        elif overall_score == 2:
            rationale = "Simple language and limited elaboration. Suitable for basic social or transactional exchanges."
        elif overall_score == 3:
            rationale = "Routine communication skills with moderate vocabulary and basic coherence."
        elif overall_score == 4:
            rationale = "Extended discourse evident. Ability to narrate and describe with fair grammatical control."
        else:
            rationale = "Advanced fluency, control, and appropriateness. Rich vocabulary and abstract expression possible."

        st.markdown(f"- **Estimated ILR Level:** {overall_score}")
        st.markdown("**Rationale:** " + rationale)

        translated_summary = back_translate(f"Estimated ILR Level is {overall_score}. " + rationale, tgt_lang=detected_lang)
        st.markdown("**Feedback (translated):**")
        st.write(translated_summary)

        speak_text(translated_summary, lang_code=detected_lang.split("-")[0])
    else:
        st.warning("Please input or upload something first.")
