"""
🐼 PersonaTalk — Streamlit App
Deploy ke Streamlit Cloud via GitHub
Otak: model diload dari Hugging Face Hub, respons dari Gemini AI
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import random
import joblib

# NLTK
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',  quiet=True)

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Gemini
import google.generativeai as genai

# Hugging Face Hub
from huggingface_hub import hf_hub_download

# ============================================================================
# KONFIGURASI
# ============================================================================

HF_REPO_ID  = "GANTI_USERNAME_HF/personatalk-models"   # ← ganti setelah buat repo HF
GEMINI_KEY  = os.environ.get("GEMINI_API_KEY", "")      # dari Streamlit Secrets

# ============================================================================
# PREPROCESSING
# ============================================================================

STOPWORDS_EN = set(stopwords.words('english'))
lemmatizer   = WordNetLemmatizer()

def text_preprocessing(text: str) -> str:
    if not text or not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOPWORDS_EN and len(w) > 2
    ]
    return ' '.join(words)

# ============================================================================
# LOAD MODEL DARI HUGGING FACE
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        emo_model_path  = hf_hub_download(repo_id=HF_REPO_ID, filename="emo_model.pkl")
        emo_vec_path    = hf_hub_download(repo_id=HF_REPO_ID, filename="emo_vectorizer.pkl")
        mbti_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="mbti_model.pkl")
        mbti_vec_path   = hf_hub_download(repo_id=HF_REPO_ID, filename="mbti_vectorizer.pkl")

        emo_model      = joblib.load(emo_model_path)
        emo_vectorizer = joblib.load(emo_vec_path)
        mbti_model     = joblib.load(mbti_model_path)
        mbti_vectorizer= joblib.load(mbti_vec_path)

        return emo_model, emo_vectorizer, mbti_model, mbti_vectorizer

    except Exception as e:
        st.error(f"❌ Gagal load model dari Hugging Face: {e}")
        st.stop()

# ============================================================================
# EMOTION DETECTION
# ============================================================================

EMOTION_NAMES_ID = {
    0: 'Sedih', 1: 'Bahagia', 2: 'Cinta',
    3: 'Marah', 4: 'Takut',  5: 'Terkejut'
}
EMOTION_ICONS = {
    0: '🦊', 1: '🐱', 2: '🐰',
    3: '🐯', 4: '🐭', 5: '🐨'
}
EMOTION_EMOJI = {
    0: '😔', 1: '😊', 2: '❤️',
    3: '😠', 4: '😨', 5: '😲'
}

EMOTION_LEXICON = {
    'sedih':0,'kecewa':0,'galau':0,'nangis':0,'menangis':0,'sepi':0,
    'kesepian':0,'ditinggal':0,'diputus':0,'putus':0,'kehilangan':0,
    'patah hati':0,'down':0,'murung':0,'hopeless':0,'nyesel':0,
    'hampa':0,'kosong':0,'frustasi':0,'bingung':0,'capek banget':0,
    'marah':3,'kesal':3,'benci':3,'jengkel':3,'muak':3,'emosi':3,
    'kesel':3,'sebel':3,'dongkol':3,'geram':3,'ngeselin':3,
    'senang':1,'bahagia':1,'happy':1,'gembira':1,'lega':1,'bangga':1,
    'excited':1,'ceria':1,'bersyukur':1,'alhamdulillah':1,'semangat':1,
    'takut':4,'cemas':4,'khawatir':4,'panik':4,'nervous':4,'gelisah':4,
    'anxious':4,'anxiety':4,'stress':4,'overthinking':4,'insecure':4,
    'cinta':2,'sayang':2,'rindu':2,'kangen':2,'suka':2,'naksir':2,
    'jatuh cinta':2,'pdkt':2,'gebetan':2,
    'kaget':5,'terkejut':5,'shock':5,'astaga':5,'ga nyangka':5,
    'nggak percaya':5,'serius':5,
}

def rule_based_emotion(text: str):
    t = text.lower()
    rules = {
        0: ['sedih','nangis','down','putus','ditinggal','patah hati',
            'kehilangan','sepi','galau','bingung','hampa','nyesel','hopeless'],
        1: ['bahagia','senang','happy','gembira','excited','lega','bangga'],
        2: ['cinta','sayang','rindu','kangen','jatuh cinta','naksir'],
        3: ['marah','kesal','benci','emosi','jengkel','muak'],
        4: ['takut','cemas','khawatir','panik','nervous','gelisah','stress','overthinking'],
        5: ['kaget','shock','terkejut','astaga']
    }
    for emo, kws in rules.items():
        if any(kw in t for kw in kws):
            return emo
    return None

def lexicon_emotion(text: str):
    t = text.lower()
    scores = {}
    for word, emo in EMOTION_LEXICON.items():
        if word in t:
            scores[emo] = scores.get(emo, 0) + 1
    return max(scores, key=scores.get) if scores else None

def predict_emotion(text: str, emo_model, emo_vectorizer):
    rb = rule_based_emotion(text)
    if rb is not None:
        return rb, 0.95
    lx = lexicon_emotion(text)
    if lx is not None:
        return lx, 0.85
    cleaned = text_preprocessing(text)
    if not cleaned:
        return 0, 0.5
    X     = emo_vectorizer.transform([cleaned])
    probs = emo_model.predict_proba(X)[0]
    return int(np.argmax(probs)), float(probs.max())

def predict_mbti(text: str, mbti_model, mbti_vectorizer, history: list):
    history.append(text)
    if len(history) < 2:
        return None, 0.0
    combined = ' '.join(history[-10:])
    cleaned  = text_preprocessing(combined)
    if not cleaned:
        return None, 0.0
    X    = mbti_vectorizer.transform([cleaned])
    pred = mbti_model.predict(X)[0]
    conf = float(mbti_model.predict_proba(X)[0].max())
    return pred, conf

# ============================================================================
# GEMINI
# ============================================================================

GEMINI_SYSTEM_PROMPT = """Kamu adalah PersonaTalk — sahabat curhat digital yang beneran peduli, bukan robot atau konselor formal.

KARAKTER KAMU:
Kamu seperti teman yang udah kenal lama, nyambung, hangat, dan nggak pernah ngehakimin. Kamu dengerin beneran, dan selalu bikin orang ngerasa dimengerti.

CARA NGOBROL:
- Bahasa Indonesia sehari-hari, santai, campuran indo-english boleh
- Mulai SELALU dengan validasi perasaan dulu — jangan langsung kasih saran
- Empati harus berasa nyata, bukan template
- Gunakan ekspresi natural: "duh", "hmm", "ya Allah", "aduh", "serius?"
- Panjang respons: 2-4 kalimat aja
- Di akhir selalu tanya SATU pertanyaan terbuka yang relevan
- Jangan pakai bullet point atau format kaku
- Jangan bilang "sebagai AI"

YANG NGGAK BOLEH:
- Diagnosis medis atau psikologis
- Saran berbahaya
- Ngehakimin atau menyalahkan
- Kalimat pembuka yang sama terus-menerus"""

def generate_gemini_response(user_text: str, emotion_label: str, history: list) -> str:
    try:
        ctx = ''
        if len(history) > 1:
            recent = history[-6:-1]
            lines  = [f"{m['role'].capitalize()}: {m['content']}" for m in recent]
            ctx    = 'Percakapan sebelumnya:\n' + '\n'.join(lines) + '\n\n'
        prompt = (
            f"{GEMINI_SYSTEM_PROMPT}\n\n"
            f"{ctx}"
            f"Pesan user: \"{user_text}\"\n"
            f"Emosi terdeteksi: {emotion_label}\n\n"
            f"Balas sebagai PersonaTalk — natural, hangat, spesifik. Max 3-4 kalimat."
        )
        model  = genai.GenerativeModel('gemini-2.0-flash')
        resp   = model.generate_content(prompt)
        result = resp.text.strip().replace('**', '').replace('##', '')
        return result if result else None
    except Exception:
        return None

def fallback_response(text: str, emotion: int) -> str:
    t = text.lower()
    if any(w in t for w in ['putus','selingkuh','ditinggal','diputus']):
        opts = [
            'Aduh, itu pasti nyakitin banget... putus ditambah selingkuh tuh double sakit ya. Gimana kondisi kamu sekarang?',
            'Ya Allah, itu berat banget. Dikhianatin sama orang yang kamu percaya... wajar banget kalau sekarang rasanya hancur. Udah cerita ke siapa belum?'
        ]
    elif any(w in t for w in ['capek','lelah','exhausted']):
        opts = [
            'Hmm, capek yang kayak gini tuh beda ya sama capek biasa. Ini capek dari mana — fisik, pikiran, atau keduanya?',
            'Ngerasa kelelahan kayak gini tuh tanda kamu udah ngasih banyak banget. Udah berapa lama ngerasa kayak gini?'
        ]
    elif emotion == 3:
        opts = [
            'Iya aku ngerti, situasi kayak gitu emang bikin darah naik. Boleh cerita lebih — ini karena apa atau siapa?',
            'Wajar banget kesel. Kadang kemarahan itu sinyal kalau ada sesuatu yang beneran nggak beres. Udah lama nahan ini?'
        ]
    elif emotion == 4:
        opts = [
            'Hmm, ngerasa cemas itu nggak enak banget ya. Ini soal apa yang bikin kamu khawatir?',
            'Overthinking paling susah dimatiin emang. Kamu cemas soal hal yang udah terjadi atau yang belum?'
        ]
    else:
        openers = {
            0: ['Duh, kedengarannya berat banget... cerita lebih dong, aku dengerin.'],
            1: ['Wah, ada kabar baik nih kayaknya! Cerita dong!'],
            2: ['Ada yang spesial nih kayaknya 😊 Cerita yuk!'],
            5: ['Wah, ada kejutan nih! Cerita dong lebih lengkapnya!']
        }
        opts = openers.get(emotion, ['Hmm, cerita lebih yuk. Aku dengerin kok.'])
    return random.choice(opts)

# ============================================================================
# CSS
# ============================================================================

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
        color: #ffffff !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c1f 0%, #1a1a2e 50%, #16213e 100%);
    }
    .main > div { background: transparent !important; padding: 1rem 2rem !important; }

    section[data-testid="stSidebar"] {
        background: rgba(10,10,25,0.85) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0,255,200,0.15);
    }
    section[data-testid="stSidebar"] * { color: white !important; }

    .welcome-box {
        background: rgba(30,30,50,0.9);
        border: 1px solid rgba(0,255,200,0.25);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0,255,200,0.1);
    }
    .chat-wrap { display:flex; align-items:flex-start; margin-bottom:16px; }
    .chat-wrap.user { flex-direction:row-reverse; }
    .bubble {
        padding: 13px 17px;
        border-radius: 18px;
        max-width: 72%;
        font-size: 15px;
        line-height: 1.6;
        word-wrap: break-word;
        white-space: pre-wrap;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .bubble.bot {
        background: rgba(30,30,55,0.85);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 4px 18px 18px 18px;
    }
    .bubble.user {
        background: linear-gradient(135deg,#667eea,#764ba2);
        border-radius: 18px 4px 18px 18px;
    }
    .avatar { font-size:1.8rem; margin:0 10px; align-self:flex-end; }

    [data-testid="stMetricValue"] { color: #00ffc8 !important; }

    .stTextInput > div > div > input {
        background: rgba(20,20,40,0.9) !important;
        border: 2px solid rgba(0,255,200,0.4) !important;
        border-radius: 30px !important;
        color: white !important;
        font-size: 15px !important;
        padding: 13px 22px !important;
        caret-color: #00ffc8;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00ffc8 !important;
        box-shadow: 0 0 20px rgba(0,255,200,0.3) !important;
    }
    .stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.4) !important; }

    .stButton > button {
        background: linear-gradient(135deg,#667eea,#764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 13px 20px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 0 20px rgba(102,126,234,0.6) !important;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg,#00ffc8,#667eea) !important;
    }
    [data-testid="stForm"] { background: transparent !important; border: none !important; }
    </style>
    """, unsafe_allow_html=True)

def labubu_anim(emotion=0):
    chars  = {0:'🦊',1:'🐱',2:'🐰',3:'🐯',4:'🐭',5:'🐨'}
    labels = {0:'Sedih',1:'Happy',2:'Love',3:'Marah',4:'Cemas',5:'Kaget'}
    char   = chars.get(emotion,'🐼')
    label  = labels.get(emotion,'Normal')
    return f"""
    <div style="display:flex;justify-content:center;align-items:center;
                padding:14px;background:rgba(255,255,255,0.04);
                border-radius:50px;margin:8px 0;border:1px solid rgba(255,255,255,0.08);">
        <span style="font-size:1.8rem;opacity:0.4;">🐼</span>
        <span style="font-size:2.6rem;margin:0 10px;
                     animation:bounce 2s infinite ease-in-out;">{char}</span>
        <span style="font-size:1.8rem;opacity:0.4;">🐼</span>
    </div>
    <div style="text-align:center;margin:-4px 0 10px 0;">
        <span style="background:rgba(255,255,255,0.08);padding:3px 14px;
                     border-radius:20px;font-size:0.83rem;color:rgba(255,255,255,0.65);">
            Labubu lagi {label}
        </span>
    </div>
    <style>
    @keyframes bounce {{
        0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-8px)}}
    }}
    </style>"""

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="PersonaTalk",
        page_icon="🐼",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_css()

    # ── Setup Gemini ──────────────────────────────────────────────────────────
    api_key = GEMINI_KEY or st.session_state.get('gemini_key', '')
    if api_key:
        genai.configure(api_key=api_key)

    # ── Load models ───────────────────────────────────────────────────────────
    with st.spinner("🧠 Memuat model PersonaTalk..."):
        emo_model, emo_vec, mbti_model, mbti_vec = load_models()

    # ── Header ────────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px 0;">
            <h1 style="background:linear-gradient(45deg,#00ffc8,#667eea);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       font-size:3rem;margin:0;">🐼 PersonaTalk</h1>
            <p style="color:rgba(255,255,255,0.7);font-size:1.05rem;margin-top:8px;">
                Teman curhat dengan deteksi emosi & analisis kepribadian MBTI
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-box">
        <div style="font-size:2.2rem;margin-bottom:8px;">🐼✨</div>
        <div style="font-size:1.3rem;font-weight:800;
                    background:linear-gradient(45deg,#00ffc8,#667eea);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Selamat datang di PersonaTalk!
        </div>
        <div style="color:rgba(255,255,255,0.75);margin-top:10px;line-height:1.7;">
            Halo! Aku di sini buat nemenin kamu ngobrol kapanpun kamu butuh.<br>
            Cerita aja — mau soal apapun, aku dengerin! 😊
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    defaults = {
        'messages':       [{'role':'bot','content':'Halo! Aku PersonaTalk 🐼\n\nAku siap dengerin cerita kamu. Mau curhat soal apa hari ini? 😊'}],
        'current_emotion': 0,
        'current_mbti':   None,
        'last_confidence': 0.5,
        'mbti_texts':     [],
        'gemini_key':     '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 Panel Analisis")

        # API Key input kalau belum ada di env
        if not GEMINI_KEY:
            st.markdown("### 🔑 Gemini API Key")
            key_input = st.text_input(
                "Masukkan API key:",
                type="password",
                placeholder="AIza...",
                value=st.session_state.get('gemini_key','')
            )
            if key_input:
                st.session_state['gemini_key'] = key_input
                genai.configure(api_key=key_input)
                st.success("✅ API Key tersambung!")
            st.markdown("---")

        # Mood
        st.markdown("### Mood Terdeteksi")
        mood_icons = {0:"😔",1:"😊",2:"❤️",3:"😠",4:"😨",5:"😲"}
        mood_names = {0:"Sedih",1:"Bahagia",2:"Cinta",3:"Marah",4:"Cemas",5:"Terkejut"}
        emo = st.session_state.current_emotion
        st.markdown(f"""
        <div style="background:rgba(0,255,200,0.08);border-left:4px solid #00ffc8;
                    padding:12px;border-radius:10px;margin:8px 0;">
            <span style="font-size:1.4rem;">{mood_icons.get(emo,'🐼')}</span>
            <strong style="font-size:1.05rem;margin-left:8px;">{mood_names.get(emo,'Normal')}</strong>
        </div>""", unsafe_allow_html=True)

        st.markdown(labubu_anim(emo), unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {st.session_state.last_confidence:.0%}")
        st.progress(st.session_state.last_confidence)
        st.markdown("---")

        # MBTI
        if st.session_state.current_mbti:
            st.markdown("### 🧬 Tipe MBTI Terdeteksi")
            st.markdown(f"""
            <div style="text-align:center;margin:10px 0;">
                <span style="background:linear-gradient(45deg,#00ffc8,#667eea);
                             padding:8px 24px;border-radius:30px;color:white;
                             font-weight:800;font-size:1.5rem;">
                    {st.session_state.current_mbti}
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("🧬 MBTI terdeteksi setelah beberapa pesan")

        st.markdown("---")
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages       = [{'role':'bot','content':'Halo! Aku PersonaTalk 🐼\n\nAku siap dengerin cerita kamu. Mau curhat soal apa hari ini? 😊'}]
            st.session_state.current_emotion = 0
            st.session_state.current_mbti   = None
            st.session_state.last_confidence = 0.5
            st.session_state.mbti_texts      = []
            st.rerun()

    # ── Chat display ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        is_user = msg['role'] == 'user'
        cls     = 'user' if is_user else 'bot'
        avatar  = '👤' if is_user else EMOTION_ICONS.get(st.session_state.current_emotion,'🐼')
        content = msg['content'].replace('<','&lt;').replace('>','&gt;')
        if is_user:
            st.markdown(f"""
            <div class="chat-wrap user">
                <div class="bubble user">{content}</div>
                <div class="avatar">{avatar}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-wrap bot">
                <div class="avatar">{avatar}</div>
                <div class="bubble bot">{content}</div>
            </div>""", unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "", placeholder="Ketik pesanmu di sini...",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("📤 Kirim", use_container_width=True)

    if submitted and user_input.strip():
        user_text = user_input.strip()

        # Deteksi emosi
        emotion, conf = predict_emotion(user_text, emo_model, emo_vec)
        st.session_state.current_emotion  = emotion
        st.session_state.last_confidence  = conf

        # Deteksi MBTI kumulatif
        mbti_pred, mbti_conf = predict_mbti(
            user_text, mbti_model, mbti_vec,
            st.session_state.mbti_texts
        )
        if mbti_pred and mbti_conf > 0.3:
            st.session_state.current_mbti = mbti_pred

        # Simpan pesan user
        st.session_state.messages.append({'role':'user','content':user_text})

        # Generate respons
        emo_name = EMOTION_NAMES_ID.get(emotion, 'netral')
        response = generate_gemini_response(user_text, emo_name, st.session_state.messages)
        if not response:
            response = fallback_response(user_text, emotion)

        st.session_state.messages.append({'role':'bot','content':response})
        st.rerun()


if __name__ == "__main__":
    main()
