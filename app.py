import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import requests
import json

# Set page config
st.set_page_config(page_title="Validin", page_icon="📰", layout="wide")

# Konfigurasi Google Gemma API
GEMMA_API_KEY = "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"
GEMMA_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Set custom NLTK data directory to a writable location
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download NLTK data with error handling
try:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
except Exception as e:
    st.error(f"Failed to download NLTK data: {str(e)}. Please ensure you have internet access and write permissions.")
    st.stop()

# Inisialisasi NLTK
stop_words = set(stopwords.words('indonesian'))

# Fungsi preprocessing teks
def clean(text):
    text = str(text).lower()
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    punct = set(string.punctuation)
    text = "".join([ch for ch in text if ch not in punct])
    return text

def tokenize(text):
    return word_tokenize(text)

def remove_stop_words(text):
    word_tokens_no_stopwords = [w for w in text if w not in stop_words]
    return word_tokens_no_stopwords

def preprocess(text):
    text = clean(text)
    text = tokenize(text)
    text = remove_stop_words(text)
    return text

# Fungsi untuk mendapatkan rekomendasi dari Gemma
def get_gemma_recommendation(news_text, prediction_result, confidence):
    try:
        if prediction_result == "HOAX":
            prompt = f"""Berdasarkan analisis AI, berita berikut terdeteksi sebagai HOAX dengan tingkat kepercayaan {confidence:.2f}%.

Berita: "{news_text[:500]}..."

Sebagai asisten AI yang membantu literasi digital, berikan:
1. Penjelasan singkat mengapa berita ini berpotensi hoax
2. 3 langkah verifikasi yang dapat dilakukan pembaca
3. Saran untuk tidak menyebarkan informasi yang belum terverifikasi
4. Rekomendasi sumber berita terpercaya di Indonesia

Berikan dalam format yang mudah dibaca dan informatif dalam bahasa Indonesia."""
        else:
            prompt = f"""Berdasarkan analisis AI, berita berikut terdeteksi sebagai VALID dengan tingkat kepercayaan {confidence:.2f}%.

Berita: "{news_text[:500]}..."

Sebagai asisten AI yang membantu literasi digital, berikan:
1. Penjelasan singkat mengapa berita ini tampak valid
2. Tetap berikan 3 tips untuk selalu memverifikasi berita, meskipun sudah terdeteksi valid
3. Pentingnya tetap berpikir kritis dalam mengonsumsi berita
4. Saran untuk berbagi informasi secara bertanggung jawab

Berikan dalam format yang mudah dibaca dan informatif dalam bahasa Indonesia."""

        headers = {
            "Authorization": f"Bearer {GEMMA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "google/gemma-2-9b-it:free",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(GEMMA_BASE_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Maaf, permintaan timeout. Silakan coba lagi."
    except requests.exceptions.RequestException as e:
        return f"Maaf, terjadi kesalahan koneksi: {str(e)}"
    except Exception as e:
        return f"Maaf, terjadi kesalahan dalam mendapatkan rekomendasi: {str(e)}"

# Cache model dan tokenizer
@st.cache_resource
def load_lstm_model():
    return load_model('hoax_lstm_model.h5')

@st.cache_resource
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as handle:
        return pickle.load(handle)

# Muat model dan tokenizer
try:
    model = load_lstm_model()
    tokenizer = load_tokenizer()
except Exception as e:
    st.error(f"Error loading model or tokenizer: {str(e)}")
    st.stop()

# Parameter tokenisasi
max_features = 5000
max_len = 300

# Custom CSS untuk tampilan modern
st.markdown("""
    <style>
    /* Reset default Streamlit styles */
    .stApp {
        background-color: #F3F4F6;
        font-family: 'Inter', sans-serif;
    }
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    /* Main container */
    .main {
        background-color: #F3F4F6;
        min-height: 100vh;
        padding: 2rem;
    }
    /* Header */
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Card untuk input */
    .card {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0 auto;
        max-width: 800px;
    }
    /* Text Area */
    div[data-testid="stTextArea"] textarea {
        border: 1px solid #D1D5DB !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        background-color: #F9FAFB !important;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #1E3A8A !important;
        box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1) !important;
    }
    /* Button */
    div[data-testid="stButton"] button {
        background-color: #1E3A8A !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        transition: background-color 0.3s ease !important;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #1C2F6B !important;
    }
    /* Result Box */
    .result-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        font-size: 1.1rem;
        font-weight: 500;
    }
    .success {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #10B981;
    }
    .error {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #EF4444;
    }
    /* Recommendation Box */
    .recommendation-box {
        background-color: #F0F9FF;
        border: 1px solid #0284C7;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    .recommendation-title {
        color: #0284C7;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .recommendation-content {
        color: #1E40AF;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #6B7280;
        margin-top: 3rem;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# UI Streamlit
st.markdown('<div class="main">', unsafe_allow_html=True)

# Header
st.markdown('<h1 class="title">VALIDIN</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deteksi berita hoax dengan cepat dan akurat menggunakan AI canggih, dilengkapi rekomendasi dari Google Gemma.</p>', unsafe_allow_html=True)

# Konten utama dalam card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Input teks
    news_text = st.text_area("Masukkan Teks Berita", placeholder="Tempel teks berita di sini...", height=200)

    # Tombol prediksi
    if st.button("🔍 Periksa Sekarang", type="primary"):
        if news_text.strip() == "":
            st.markdown('<div class="result-box error">⚠️ Mohon masukkan teks berita!</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Menganalisis berita..."):
                # Preprocessing teks
                processed_text = preprocess(news_text)
                text_seq = tokenizer.texts_to_sequences([" ".join(processed_text)])
                if not text_seq[0]:
                    st.markdown('<div class="result-box error">⚠️ Teks tidak dapat diproses. Pastikan teks relevan.</div>', unsafe_allow_html=True)
                    st.stop()
                text_padded = pad_sequences(sequences=text_seq, maxlen=max_len, padding='pre')
                
                # Prediksi
                prediction = model.predict(text_padded)
                threshold = 0.6
                hoax_prob = prediction[0][1]
                pred_class = 1 if hoax_prob > threshold else 0
                pred_prob = hoax_prob * 100 if pred_class == 1 else (1 - hoax_prob) * 100

                # Tampilkan hasil prediksi
                if pred_class == 1:
                    st.markdown(f'<div class="result-box error">🚨 <b>Peringatan</b>: Berita ini kemungkinan <b>HOAX</b> (Kepercayaan: {pred_prob:.2f}%)</div>', unsafe_allow_html=True)
                    result_type = "HOAX"
                else:
                    st.markdown(f'<div class="result-box success">✅ <b>Hasil</b>: Berita ini kemungkinan <b>VALID</b> (Kepercayaan: {pred_prob:.2f}%)</div>', unsafe_allow_html=True)
                    result_type = "VALID"
                
                # Mendapatkan rekomendasi dari Gemma
                st.markdown("---")
                with st.spinner("Mendapatkan rekomendasi dari Google Gemma..."):
                    recommendation = get_gemma_recommendation(news_text, result_type, pred_prob)
                    
                    # Tampilkan rekomendasi
                    st.markdown(f'''
                    <div class="recommendation-box">
                        <div class="recommendation-title">
                            🤖 Rekomendasi & Panduan dari Google Gemma
                        </div>
                        <div class="recommendation-content">{recommendation}</div>
                    </div>
                    ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<p class="footer">© 2025 Validin - Powered by AI & Google Gemma.</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
