import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Inisialisasi NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Fungsi preprocessing teks (sama dengan kode Anda)
def clean(text):
    text = text.lower()
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    punct = set(string.punctuation)
    text = "".join([ch for ch in text if ch not in punct])
    return text

def tokenize(text):
    return word_tokenize(text)

def remove_stop_words(text):
    word_tokens_no_stopwords = [w for w in text if not w in stop_words]
    return word_tokens_no_stopwords

def preprocess(text):
    text = clean(text)
    text = tokenize(text)
    text = remove_stop_words(text)
    return text

# Cache model dan tokenizer untuk efisiensi
@st.cache_resource
def load_lstm_model():
    return load_model('hoax_lstm_model.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

# Muat model dan tokenizer
model = load_lstm_model()
tokenizer = load_tokenizer()

# Parameter tokenisasi (sesuaikan dengan kode pelatihan)
max_features = 1000

# UI Streamlit
st.set_page_config(page_title="HoaxBuster", page_icon="📰", layout="wide")
st.title("📰 HoaxBuster: Deteksi Berita Hoax")
st.markdown("""
    Masukkan teks berita di bawah ini untuk memeriksa apakah berita tersebut **hoax** atau **valid**.
    Aplikasi ini menggunakan model LSTM untuk analisis teks berbahasa Indonesia.
""")

# Input teks
news_text = st.text_area("Teks Berita", placeholder="Tempel teks berita di sini...", height=200)

# Tombol prediksi
if st.button("🔍 Periksa Berita", type="primary"):
    if news_text.strip() == "":
        st.warning("Mohon masukkan teks berita!", icon="⚠️")
    else:
        with st.spinner("Menganalisis teks..."):
            # Preprocessing teks
            processed_text = preprocess(news_text)
            text_seq = tokenizer.texts_to_sequences([" ".join(processed_text)])
            text_padded = pad_sequences(sequences=text_seq, maxlen=max_features, padding='pre')

            # Prediksi
            prediction = model.predict(text_padded)
            pred_class = np.argmax(prediction, axis=1)[0]
            pred_prob = prediction[0][pred_class] * 100

            # Tampilkan hasil
            if pred_class == 1:
                st.error(f"**Peringatan**: Berita ini kemungkinan **HOAX** (Kepercayaan: {pred_prob:.2f}%)", icon="🚨")
            else:
                st.success(f"**Hasil**: Berita ini kemungkinan **VALID** (Kepercayaan: {pred_prob:.2f}%)", icon="✅")

# Footer
st.markdown("---")
st.markdown("© 2025 HoaxBuster. Dibuat untuk mendeteksi berita hoax dengan AI.")