# 🐼 PersonaTalk

**Teman curhat digital dengan deteksi emosi & analisis kepribadian MBTI**, dibangun pakai Streamlit.

PersonaTalk punya dua mode interaksi:
- **💬 Curhat** — ngobrol santai, bot mendeteksi emosi dari tiap pesan dan merespons dengan gaya empatik ala teman dekat.
- **🧬 Analisis MBTI** — kuis 10 pertanyaan (A/B) untuk menentukan tipe kepribadian MBTI-mu, lengkap dengan radar chart dan penjelasan tiap dimensi.

Selain itu, tipe MBTI juga bisa terdeteksi otomatis secara *pasif* dari histori chat kamu selama mode Curhat.

---

## ✨ Fitur

- **Deteksi emosi** dari teks (6 kategori: Sedih, Bahagia, Cinta, Marah, Takut, Terkejut) — kombinasi rule-based, lexicon, dan model machine learning (fallback berlapis).
- **Chatbot empatik** berbahasa Indonesia casual, didukung Groq API (Llama 3.3 70B) dengan fallback ke respons template kalau API tidak tersedia.
- **Analisis MBTI**:
  - Kuis interaktif 10 pertanyaan.
  - Deteksi pasif dari histori percakapan (butuh minimal 2 pesan).
  - Radar chart visualisasi 4 dimensi (E/I, S/N, T/F, J/P) pakai Plotly.
- **UI custom** bertema gelap/neon dengan avatar karakter yang berubah sesuai mood.
- Model emosi & MBTI di-load otomatis dari HuggingFace Hub (`Jooou139/personatalk`).

---

## 🗂️ Struktur Project

```
personatalk-main/
├── app.py                  # Seluruh logic aplikasi (UI, model, chatbot, MBTI)
├── requirements.txt        # Dependencies Python
└── .streamlit/
    └── config.toml         # Konfigurasi tema & server Streamlit
```

---

## 🚀 Cara Menjalankan

### 1. Clone / masuk ke folder project
```bash
cd personatalk-main
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Siapkan secrets
Buat file `.streamlit/secrets.toml` di root project:
```toml
GROQ_API_KEY = "isi_dengan_api_key_groq_kamu"
HF_TOKEN = "isi_dengan_token_huggingface_kamu"   # opsional, tergantung akses repo model
```

> `GROQ_API_KEY` dipakai untuk respons chatbot berbasis LLM (Llama 3.3 70B via Groq). Kalau tidak diisi, aplikasi tetap jalan pakai respons fallback berbasis template.
> `HF_TOKEN` dipakai untuk mengunduh model `.pkl` dari HuggingFace Hub repo `Jooou139/personatalk`.

### 4. Jalankan aplikasi
```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di `http://localhost:8501`.

---

## 🧠 Model & Data

Aplikasi memuat 4 file model dari HuggingFace Hub (`Jooou139/personatalk`) saat pertama kali dijalankan (dan di-cache oleh Streamlit):

| File | Fungsi |
|---|---|
| `emo_model.pkl` | Model klasifikasi emosi |
| `emo_vectorizer.pkl` | Vectorizer teks untuk model emosi |
| `mbti_model.pkl` | Model klasifikasi MBTI |
| `mbti_vectorizer.pkl` | Vectorizer teks untuk model MBTI |

Teks pengguna melalui tahap **preprocessing** (lowercase, hapus URL/karakter non-huruf, hapus stopwords Inggris via NLTK, lemmatization) sebelum masuk ke model.

Untuk emosi, sistem mengecek 3 lapis secara berurutan sebelum fallback ke model ML:
1. **Rule-based** — deteksi frasa spesifik bahasa Indonesia (mis. "patah hati", "overthinking").
2. **Lexicon-based** — pencocokan kata kunci individual.
3. **Model ML** — dipakai kalau dua lapis di atas tidak menemukan kecocokan.

---

## ⚙️ Konfigurasi Tema

Tema aplikasi (dark mode, warna aksen neon `#00ffc8`) sudah diatur di `.streamlit/config.toml` dan bisa disesuaikan sesuai kebutuhan.

---

## 📦 Dependencies Utama

- `streamlit` — framework UI
- `groq` — client API untuk respons chatbot LLM
- `huggingface_hub` — download model dari HF Hub
- `scikit-learn`, `joblib` — load & jalankan model ML
- `nltk` — preprocessing teks (stopwords, lemmatization)
- `plotly` — radar chart MBTI
- `pandas`, `numpy` — pengolahan data pendukung

---

## ⚠️ Catatan

- PersonaTalk **bukan pengganti konseling profesional**. Chatbot tidak melakukan diagnosis medis/psikologis dan didesain hanya sebagai teman ngobrol santai.
- Jika `GROQ_API_KEY` tidak tersedia atau terjadi error saat memanggil API, aplikasi otomatis pakai respons fallback berbasis template agar tetap bisa dipakai.
