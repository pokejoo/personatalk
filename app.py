"""
🐼 PersonaTalk — Streamlit App
Deploy ke Streamlit Cloud via GitHub
"""

import streamlit as st
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

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Gemini
try:
    import google.generativeai as genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

# Anthropic
try:
    import anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

# Hugging Face Hub
from huggingface_hub import hf_hub_download

# ============================================================================
# KONFIGURASI
# ============================================================================

HF_REPO_ID     = "Jooou139/personatalk"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
ANTHROPIC_KEY  = st.secrets.get("ANTHROPIC_API_KEY", "")
HF_TOKEN       = st.secrets.get("HF_TOKEN", "")

if _GENAI_OK and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
        token = HF_TOKEN if HF_TOKEN else None
        emo_model      = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="emo_model.pkl",       token=token))
        emo_vectorizer = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="emo_vectorizer.pkl",  token=token))
        mbti_model     = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="mbti_model.pkl",      token=token))
        mbti_vectorizer= joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="mbti_vectorizer.pkl", token=token))
        return emo_model, emo_vectorizer, mbti_model, mbti_vectorizer
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        st.stop()

# ============================================================================
# EMOTION DETECTION
# ============================================================================

EMOTION_NAMES_ID = {0:'Sedih',1:'Bahagia',2:'Cinta',3:'Marah',4:'Cemas',5:'Terkejut'}
EMOTION_ICONS    = {0:'🦊',1:'🐱',2:'🐰',3:'🐯',4:'🐭',5:'🐨'}
EMOTION_EMOJI    = {0:'😔',1:'😊',2:'❤️',3:'😠',4:'😨',5:'😲'}

EMOTION_LEXICON = {
    'sedih':0,'kecewa':0,'galau':0,'nangis':0,'sepi':0,'kesepian':0,
    'ditinggal':0,'diputus':0,'putus':0,'kehilangan':0,'patah hati':0,
    'down':0,'murung':0,'hopeless':0,'nyesel':0,'hampa':0,'kosong':0,
    'frustasi':0,'bingung':0,'capek banget':0,'sendirian':0,'sakit hati':0,
    'marah':3,'kesal':3,'benci':3,'jengkel':3,'muak':3,'emosi':3,
    'kesel':3,'sebel':3,'dongkol':3,'geram':3,'ngeselin':3,'dibohongin':3,
    'senang':1,'bahagia':1,'happy':1,'gembira':1,'lega':1,'bangga':1,
    'excited':1,'ceria':1,'bersyukur':1,'alhamdulillah':1,'semangat':1,
    'takut':4,'cemas':4,'khawatir':4,'panik':4,'nervous':4,'gelisah':4,
    'anxious':4,'anxiety':4,'stress':4,'overthinking':4,'insecure':4,'was-was':4,
    'cinta':2,'sayang':2,'rindu':2,'kangen':2,'suka':2,'naksir':2,
    'jatuh cinta':2,'pdkt':2,'gebetan':2,
    'kaget':5,'terkejut':5,'shock':5,'astaga':5,'ga nyangka':5,'nggak percaya':5,
}

def rule_based_emotion(text: str):
    t = text.lower()
    # Multi-word phrases dulu
    phrases = {
        0: ['harus apa','mau ngapain','ga ada dia','nggak ada dia','tanpa dia',
            'kehilangan dia','patah hati','putus sama','abis putus','diselingkuhin',
            'dikhianatin','ngerasa sepi','ga semangat','bingung banget','hilang arah'],
        3: ['nggak adil','gak adil','bikin kesel','dibohongin','ditipu'],
        4: ['overthinking','was-was','nggak tenang','deg degan','nggak yakin'],
        1: ['seneng banget','happy banget','lega banget','alhamdulillah'],
    }
    for emo, ph_list in phrases.items():
        if any(ph in t for ph in ph_list):
            return emo
    # Single keyword
    rules = {
        0: ['sedih','nangis','down','putus','ditinggal','patah hati','kehilangan',
            'sepi','galau','bingung','hampa','nyesel','hopeless','frustasi'],
        1: ['bahagia','senang','happy','gembira','excited','lega','bangga'],
        2: ['cinta','sayang','rindu','kangen','jatuh cinta','naksir'],
        3: ['marah','kesal','benci','emosi','jengkel','muak'],
        4: ['takut','cemas','khawatir','panik','nervous','gelisah','stress'],
        5: ['kaget','shock','terkejut','astaga']
    }
    for emo, kws in rules.items():
        if any(kw in t for kw in kws):
            return emo
    return None

def predict_emotion(text: str, emo_model, emo_vectorizer):
    rb = rule_based_emotion(text)
    if rb is not None:
        return rb, 0.95
    # Lexicon
    t = text.lower()
    scores = {}
    for word, emo in EMOTION_LEXICON.items():
        if word in t:
            scores[emo] = scores.get(emo, 0) + 1
    if scores:
        return max(scores, key=scores.get), 0.85
    # ML model
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
# PERSONALITY ANALYZER (MODE MBTI)
# ============================================================================

MBTI_QUESTIONS = [
    {"id":1,"q":"Ketika menghadapi masalah besar, kamu lebih suka:","A":"Langsung cari solusi praktis dan bertindak","B":"Merenung dan mikirin berbagai kemungkinan dulu","dim":"S/N","w":{"A":"S","B":"N"}},
    {"id":2,"q":"Di waktu luang, kamu lebih menikmati:","A":"Kumpul dan sosialisasi sama banyak orang","B":"Waktu sendiri untuk recharge energi","dim":"E/I","w":{"A":"E","B":"I"}},
    {"id":3,"q":"Saat ambil keputusan penting, kamu lebih andalkan:","A":"Logika dan analisis objektif","B":"Perasaan dan nilai-nilai pribadi","dim":"T/F","w":{"A":"T","B":"F"}},
    {"id":4,"q":"Gaya hidup sehari-harimu lebih ke mana?","A":"Terstruktur dengan jadwal dan rencana jelas","B":"Fleksibel dan ngikutin alur situasi","dim":"J/P","w":{"A":"J","B":"P"}},
    {"id":5,"q":"Waktu belajar hal baru, kamu lebih suka:","A":"Langsung praktek dan hands-on","B":"Baca teori dan pahami konsepnya dulu","dim":"S/N","w":{"A":"S","B":"N"}},
    {"id":6,"q":"Di grup chat atau diskusi, biasanya kamu:","A":"Aktif respon dan sering mulai topik baru","B":"Lebih banyak baca, sesekali baru komentar","dim":"E/I","w":{"A":"E","B":"I"}},
    {"id":7,"q":"Kalau teman curhat masalah, respons pertamamu:","A":"Langsung kasih solusi praktis","B":"Dengerin dulu dan kasih dukungan emosional","dim":"T/F","w":{"A":"T","B":"F"}},
    {"id":8,"q":"Menjelang deadline, kamu biasanya:","A":"Selesaikan jauh-jauh hari sebelumnya","B":"Paling produktif justru di menit-menit akhir","dim":"J/P","w":{"A":"J","B":"P"}},
    {"id":9,"q":"Kamu lebih tertarik pada:","A":"Fakta, detail konkret, dan pengalaman nyata","B":"Pola, kemungkinan besar, dan gambaran besar","dim":"S/N","w":{"A":"S","B":"N"}},
    {"id":10,"q":"Setelah seharian interaksi sama banyak orang, kamu merasa:","A":"Makin semangat dan energized","B":"Capek dan butuh waktu sendiri buat recharge","dim":"E/I","w":{"A":"E","B":"I"}},
]

MBTI_DESC = {
    "ISTJ":{"name":"The Logistician","desc":"Praktis, faktual, dan sangat terorganisir. Kamu orang yang bisa diandalkan."},
    "ISFJ":{"name":"The Defender","desc":"Penyayang, hangat, dan selalu siap melindungi orang yang kamu cintai."},
    "INFJ":{"name":"The Advocate","desc":"Idealistik, berprinsip, dan visioner. Kamu lihat dunia dengan cara yang unik."},
    "INTJ":{"name":"The Architect","desc":"Strategis, logis, dan selalu mikir jangka panjang. Masterplanner sejati."},
    "ISTP":{"name":"The Virtuoso","desc":"Pragmatis, fleksibel, dan jago banget mecahin masalah teknis."},
    "ISFP":{"name":"The Adventurer","desc":"Artistik, spontan, dan selalu hidup di saat ini dengan penuh warna."},
    "INFP":{"name":"The Mediator","desc":"Idealis, empatik, dan selalu cari makna di balik setiap hal."},
    "INTP":{"name":"The Logician","desc":"Inovatif, analitis, dan pencinta teori yang selalu ingin tahu."},
    "ESTP":{"name":"The Entrepreneur","desc":"Enerjik, action-oriented, dan persuasif. Kamu suka tantangan nyata."},
    "ESFP":{"name":"The Entertainer","desc":"Hangat, ramah, dan suka jadi pusat perhatian yang bikin suasana hidup."},
    "ENFP":{"name":"The Campaigner","desc":"Antusias, kreatif, dan inspiratif. Kamu bisa nyalaiin semangat orang lain."},
    "ENTP":{"name":"The Debater","desc":"Cerdas, suka tantangan intelektual, dan inovatif dalam berpikir."},
    "ESTJ":{"name":"The Executive","desc":"Tegas, terorganisir, dan efisien. Natural leader yang result-oriented."},
    "ESFJ":{"name":"The Consul","desc":"Penyayang, populer, dan selalu mau bantu. Sosok yang semua orang suka."},
    "ENFJ":{"name":"The Protagonist","desc":"Karismatik, inspiratif, dan suka bimbing orang menuju versi terbaiknya."},
    "ENTJ":{"name":"The Commander","desc":"Berwibawa, strategis, dan pemimpin alami yang penuh visi besar."},
}

DIM_EXPLAIN = {
    "E":"Ekstrovert — Energimu datang dari interaksi sosial",
    "I":"Introvert — Energimu datang dari waktu sendiri",
    "S":"Sensing — Kamu fokus pada fakta dan detail konkret",
    "N":"Intuition — Kamu fokus pada pola dan kemungkinan besar",
    "T":"Thinking — Keputusanmu berdasarkan logika dan analisis",
    "F":"Feeling — Keputusanmu berdasarkan nilai dan perasaan",
    "J":"Judging — Kamu suka struktur dan perencanaan yang jelas",
    "P":"Perceiving — Kamu fleksibel dan adaptif terhadap situasi",
}

def analyze_mbti(responses: list) -> str:
    scores = {"E":0,"I":0,"S":0,"N":0,"T":0,"F":0,"J":0,"P":0}
    for r in responses:
        q = next((x for x in MBTI_QUESTIONS if x["id"] == r["qid"]), None)
        if q and r["ans"] in q["w"]:
            scores[q["w"][r["ans"]]] += 1
    return (
        ("E" if scores["E"] >= scores["I"] else "I") +
        ("S" if scores["S"] >= scores["N"] else "N") +
        ("T" if scores["T"] >= scores["F"] else "F") +
        ("J" if scores["J"] >= scores["P"] else "P")
    )

def format_mbti_result(mbti_type: str) -> str:
    info = MBTI_DESC.get(mbti_type, {"name":"Unknown","desc":""})
    dims = "\n".join([f"• {d} → {DIM_EXPLAIN[d]}" for d in list(mbti_type) if d in DIM_EXPLAIN])
    return (
        f"Tipe Kepribadianmu: {mbti_type}\n"
        f"{info['name']}\n\n"
        f"{info['desc']}\n\n"
        f"Breakdown dimensimu:\n{dims}"
    )

# ============================================================================
# AI RESPONSE — Anthropic Claude + Gemini fallback
# ============================================================================

SYSTEM_PROMPT = """Kamu adalah PersonaTalk — sahabat curhat digital yang beneran peduli, bukan robot atau konselor formal.

KARAKTER KAMU:
Kamu seperti teman yang udah kenal lama — hangat, nyambung, dan nggak pernah ngehakimin. Kamu dengerin beneran sebelum kasih solusi.

CARA NGOBROL:
- Bahasa Indonesia sehari-hari, santai, campuran indo-english boleh
- SELALU validasi perasaan dulu sebelum kasih saran atau insight
- Empati harus terasa nyata — bukan template kayak "aku mengerti perasaanmu"
- Gunakan ekspresi natural: "duh", "hmm", "ya Allah", "aduh", "serius?"
- Panjang respons: 2-4 kalimat, nggak lebih
- Akhiri dengan SATU pertanyaan terbuka yang relevan dan spesifik
- Jangan pakai bullet point atau format kaku
- Jangan sebut diri sebagai AI atau bot
- Jangan ulangi kalimat yang sama dengan respons sebelumnya

KALAU USER SEDIH/GALAU: peluk dulu secara verbal, baru tanya
KALAU USER MARAH: acknowledge kemarahannya, jangan minta calm down
KALAU USER BINGUNG: temenin dulu, jangan langsung kasih solusi
KALAU USER CERITA TENTANG ORANG LAIN (gebetan, teman, dll): ikut excited/empati sesuai konteks

JANGAN:
- Diagnosis medis/psikologis
- Saran berbahaya
- Kalimat pembuka yang sama terus-menerus
- Kalimat: "Perasaan seperti ini sangat manusiawi kok" — terlalu template"""

def generate_ai_response(user_text: str, emotion_label: str, history: list) -> str:
    # Build context
    ctx = ""
    if len(history) > 1:
        lines = []
        for m in history[-6:-1]:
            role = "User" if m["role"] == "user" else "PersonaTalk"
            lines.append(f"{role}: {m['content']}")
        if lines:
            ctx = "Percakapan sebelumnya:\n" + "\n".join(lines) + "\n\n"

    user_prompt = (
        f"{ctx}Pesan user: \"{user_text}\"\n"
        f"Emosi terdeteksi: {emotion_label}\n\n"
        f"Balas sebagai PersonaTalk — natural, hangat, manusiawi, spesifik ke situasinya. "
        f"Jangan template. Max 3-4 kalimat."
    )

    # === ANTHROPIC CLAUDE (primary — lebih natural) ===
    if _ANTHROPIC_OK and ANTHROPIC_KEY:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            result = msg.content[0].text.strip().replace("**", "").replace("##", "")
            if result:
                return result
        except Exception:
            pass

    # === GEMINI (fallback) ===
    if _GENAI_OK and GEMINI_API_KEY:
        try:
            model  = genai.GenerativeModel("gemini-2.0-flash")
            resp   = model.generate_content(f"{SYSTEM_PROMPT}\n\n{user_prompt}")
            result = resp.text.strip().replace("**", "").replace("##", "")
            if result:
                return result
        except Exception:
            pass

    return None

def fallback_response(text: str, emotion: int) -> str:
    t = text.lower()
    if any(w in t for w in ['putus','selingkuh','ditinggal','diputus','diselingkuhin']):
        opts = [
            'Aduh, itu pasti nyakitin banget... putus ditambah selingkuh tuh double sakit ya. Gimana kondisi kamu sekarang, masih shock atau udah bisa napas dikit?',
            'Ya Allah, itu berat banget. Dikhianatin sama orang yang kamu percaya... wajar banget kalau sekarang rasanya hancur. Udah cerita ke siapa belum?',
        ]
    elif any(w in t for w in ['bingung','harus apa','mau ngapain','ga ada','nggak ada','tanpa dia']):
        opts = [
            'Hmm, ngerasa kayak tiba-tiba harus jalan sendiri dan bingung mulai dari mana ya? Itu wajar banget. Kamu udah lama sama dia?',
            'Kehilangan seseorang yang udah jadi bagian besar dari hidup tuh emang bikin kosong. Yang paling bikin kamu kepikiran sekarang apa?',
        ]
    elif any(w in t for w in ['capek','lelah','exhausted','burnout']):
        opts = [
            'Hmm, capek yang kayak gini beda sama capek biasa. Ini capek dari mana — fisik, pikiran, atau keduanya sekaligus?',
            'Ngerasa kelelahan kayak gini tuh tanda kamu udah ngasih banyak banget. Udah berapa lama ngerasa kayak gini?',
        ]
    elif any(w in t for w in ['suka','naksir','gebetan','pdkt','cantik','ganteng','kangen']):
        opts = [
            'Wah, ada yang spesial nih kayaknya! Cerita dong lebih — udah lama kenal?',
            'Ooh, menarik! Dia tau nggak kalau kamu naksir? Atau masih tahap ngira-ngira?',
        ]
    elif emotion == 3:
        opts = [
            'Iya aku ngerti, situasi kayak gitu emang bikin darah naik. Boleh cerita lebih — ini karena apa atau siapa?',
            'Wajar banget kesel. Kemarahan itu sering jadi sinyal kalau ada sesuatu yang beneran nggak beres. Udah lama nahan ini?',
        ]
    elif emotion == 4:
        opts = [
            'Hmm, ngerasa cemas itu nggak enak banget ya, kayak ada beban terus. Ini soal apa yang paling bikin kamu khawatir?',
            'Overthinking paling susah dimatiin emang. Kamu cemas soal hal yang udah terjadi atau yang belum?',
        ]
    elif emotion == 1:
        opts = [
            'Wah, energinya positif banget nih! Ada kabar baik? Cerita dong!',
            'Seneng banget denger kamu lagi happy! Ada hal spesial yang terjadi hari ini?',
        ]
    else:
        opts = [
            'Hmm, cerita lebih yuk. Aku dengerin kok — ada apa?',
            'Duh, kedengarannya ada yang lagi dipikirin nih. Mau cerita dari mana dulu?',
        ]
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
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #00ffc8 !important; }

    .welcome-box {
        background: rgba(30,30,50,0.9);
        border: 1px solid rgba(0,255,200,0.25);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0,255,200,0.1);
    }
    .chat-wrap { display:flex; align-items:flex-start; margin-bottom:16px; animation: floatIn 0.3s ease; }
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
        backdrop-filter: blur(8px);
    }
    .bubble.user {
        background: linear-gradient(135deg,#667eea,#764ba2);
        border-radius: 18px 4px 18px 18px;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .avatar { font-size:1.8rem; margin:0 10px; align-self:flex-end; }

    [data-testid="stMetricValue"] { color: #00ffc8 !important; }
    [data-testid="stMetricLabel"] { color: rgba(255,255,255,0.6) !important; }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg,#00ffc8,#667eea) !important;
    }
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
        color: white !important; border: none !important;
        border-radius: 30px !important;
        padding: 13px 20px !important;
        font-weight: 700 !important; font-size: 15px !important;
        width: 100% !important; transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 0 20px rgba(102,126,234,0.6) !important;
    }
    div[role="radiogroup"] { background: rgba(15,25,45,0.8); padding:8px; border-radius:12px; }
    div[role="radiogroup"] label { color: white !important; }
    [data-testid="stForm"] { background: transparent !important; border: none !important; }

    @keyframes floatIn {
        from { opacity:0; transform:translateY(12px); }
        to   { opacity:1; transform:translateY(0); }
    }
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
        <span style="font-size:2.6rem;margin:0 10px;animation:bounce 2s infinite ease-in-out;">{char}</span>
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

    # ── Load models ───────────────────────────────────────────────────────────
    with st.spinner("⚡ Memuat PersonaTalk..."):
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
            Pilih mode <b>Curhat</b> buat cerita santai, atau <b>Analisis MBTI</b> buat tau tipe kepribadianmu! 😊
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    defaults = {
        'messages':            [{'role':'bot','content':'Halo! Aku PersonaTalk 🐼\n\nAku siap dengerin cerita kamu. Mau curhat soal apa hari ini? 😊'}],
        'current_emotion':     0,
        'current_mbti':        None,
        'last_confidence':     0.5,
        'mbti_texts':          [],
        'mode':                '💬 Curhat',
        'q_index':             -1,
        'q_responses':         [],
        'last_bot_responses':  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 Panel Analisis")
        st.markdown("---")

        # Mode selector
        mode = st.radio("Mode Interaksi", ["💬 Curhat", "🧬 Analisis MBTI"], horizontal=True)
        if mode != st.session_state.mode:
            st.session_state.mode    = mode
            st.session_state.q_index = -1
            st.session_state.q_responses = []

        st.markdown("---")

        # Mood display
        st.markdown("### Mood Terdeteksi")
        emo = st.session_state.current_emotion
        st.markdown(f"""
        <div style="background:rgba(0,255,200,0.08);border-left:4px solid #00ffc8;
                    padding:12px;border-radius:10px;margin:8px 0;">
            <span style="font-size:1.4rem;">{EMOTION_EMOJI.get(emo,'🐼')}</span>
            <strong style="font-size:1.05rem;margin-left:8px;">{EMOTION_NAMES_ID.get(emo,'Normal')}</strong>
        </div>""", unsafe_allow_html=True)

        st.markdown(labubu_anim(emo), unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {st.session_state.last_confidence:.0%}")
        st.progress(st.session_state.last_confidence)
        st.markdown("---")

        # MBTI display
        if st.session_state.current_mbti:
            st.markdown("### 🧬 Tipe MBTI")
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

        # Progress bar untuk mode MBTI
        if st.session_state.q_index >= 0:
            st.markdown("---")
            total    = len(MBTI_QUESTIONS)
            progress = min((st.session_state.q_index + 1) / total, 1.0)
            st.progress(progress)
            st.caption(f"Pertanyaan {min(st.session_state.q_index + 1, total)} dari {total}")

        st.markdown("---")
        if st.button("🔄 Reset Chat", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ── Chat display ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        is_user = msg['role'] == 'user'
        cls     = 'user' if is_user else 'bot'
        avatar  = '👤' if is_user else EMOTION_ICONS.get(st.session_state.current_emotion, '🐼')
        content = msg['content'].replace('<', '&lt;').replace('>', '&gt;')
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
            placeholder = "Ketik pesanmu di sini..." if st.session_state.mode == "💬 Curhat" else "Jawab A atau B..."
            user_input  = st.text_input("", placeholder=placeholder, label_visibility="collapsed")
        with col2:
            submitted = st.form_submit_button("📤 Kirim", use_container_width=True)

    if submitted and user_input.strip():
        user_text = user_input.strip()
        st.session_state.messages.append({'role': 'user', 'content': user_text})

        # ── MODE CURHAT ───────────────────────────────────────────────────────
        if st.session_state.mode == "💬 Curhat":
            emotion, conf = predict_emotion(user_text, emo_model, emo_vec)
            st.session_state.current_emotion = emotion
            st.session_state.last_confidence = conf

            # Update MBTI dari chat
            mbti_pred, mbti_conf = predict_mbti(user_text, mbti_model, mbti_vec, st.session_state.mbti_texts)
            if mbti_pred and mbti_conf > 0.3:
                st.session_state.current_mbti = mbti_pred

            # Generate AI response
            emo_name = EMOTION_NAMES_ID.get(emotion, 'netral')
            response = generate_ai_response(user_text, emo_name, st.session_state.messages)

            # Hindari duplikat
            if response and response in st.session_state.last_bot_responses[-3:]:
                response = None
            if not response:
                response = fallback_response(user_text, emotion)

            st.session_state.last_bot_responses.append(response)
            if len(st.session_state.last_bot_responses) > 10:
                st.session_state.last_bot_responses.pop(0)

        # ── MODE ANALISIS MBTI ────────────────────────────────────────────────
        else:
            if st.session_state.q_index == -1:
                # Mulai kuesioner
                st.session_state.q_responses = []
                st.session_state.q_index     = 0
                q    = MBTI_QUESTIONS[0]
                response = (
                    f"Oke, yuk kita mulai analisis kepribadianmu! 🎯\n\n"
                    f"Pertanyaan {q['id']} dari {len(MBTI_QUESTIONS)}:\n\n"
                    f"{q['q']}\n\n"
                    f"A. {q['A']}\n"
                    f"B. {q['B']}\n\n"
                    f"Jawab dengan A atau B ya 😊"
                )
            else:
                ans = user_text.strip().upper()
                if ans in ['A', 'B']:
                    # Simpan jawaban
                    current_q = MBTI_QUESTIONS[st.session_state.q_index]
                    st.session_state.q_responses.append({"qid": current_q["id"], "ans": ans})
                    next_idx = st.session_state.q_index + 1

                    if next_idx < len(MBTI_QUESTIONS):
                        st.session_state.q_index = next_idx
                        q = MBTI_QUESTIONS[next_idx]
                        response = (
                            f"Pertanyaan {q['id']} dari {len(MBTI_QUESTIONS)}:\n\n"
                            f"{q['q']}\n\n"
                            f"A. {q['A']}\n"
                            f"B. {q['B']}\n\n"
                            f"Jawab dengan A atau B ya 😊"
                        )
                    else:
                        # Selesai — analisis
                        mbti_type = analyze_mbti(st.session_state.q_responses)
                        st.session_state.current_mbti = mbti_type
                        st.session_state.q_index      = -1
                        result = format_mbti_result(mbti_type)
                        response = (
                            f"✨ Analisis selesai!\n\n{result}\n\n"
                            f"---\nSekarang kamu bisa pindah ke mode Curhat kalau mau ngobrol santai 😊"
                        )
                else:
                    response = "Eh, jawabnya A atau B aja ya 😄 Coba lagi!"

        st.session_state.messages.append({'role': 'bot', 'content': response})
        st.rerun()


if __name__ == "__main__":
    main()
