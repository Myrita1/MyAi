import streamlit as st
import numpy as np
import os
import tempfile

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

# Setup
nltk_data_dir = os.path.expanduser(os.path.join("~", "nltk_data"))
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
download_corpora.download_all()

classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

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

def summarize_text(text):
    trimmed = " ".join(text.split()[:800])
    result = summarizer(trimmed, max_length=130, min_length=30, do_sample=False)
    return result[0]['summary_text']

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

def transcribe_audio_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    audio = AudioSegment.from_file(tmp_path, format="wav")
    audio = audio.set_channels(1).set_frame_rate(16000)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    input_values = tokenizer_wav2vec(samples, return_tensors='pt', padding='longest').input_values
    logits = model_wav2vec(input_values).logits
    predicted_ids = np.argmax(logits, axis=-1)
    return tokenizer_wav2vec.batch_decode(predicted_ids)[0]

def speak_text(text, lang_code):
    lang = LANG_CODE_MAP.get(lang_code, "en")
    tts = gTTS(text=text, lang=lang)
    tts.save("feedback.mp3")
    os.system("start feedback.mp3" if os.name == "nt" else "afplay feedback.mp3")

# Streamlit Interface
st.title("Multilingual ILR Language Assessment Tool")
st.markdown("Detect language, translate, summarize key ideas, and assign an ILR level (1–5).")

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
            if detected_lang != "en":
                translated_text = translate(user_input, src_lang=detected_lang, tgt_lang="en")
                st.markdown("**Translated to English:**")
                st.write(translated_text)
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
            trimmed_text = " ".join(translated_text.split()[:500])
            sentiment = classifier(trimmed_text)[0]
            ilr_level = generate_ilr_level(blob, sentences, sentiment["label"])

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
        st.warning("Please input or upload something first.")
