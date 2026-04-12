"""
🐼 PersonaTalk — Custom UI (Groq Backend)
"""

import streamlit as st
import numpy as np
import re
import random
import math
import joblib

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',  quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    from groq import Groq
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False

from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID = "Jooou139/personatalk"
GROQ_KEY   = st.secrets.get("GROQ_API_KEY", "")
HF_TOKEN   = st.secrets.get("HF_TOKEN", "")

# ── Preprocessing ─────────────────────────────────────────────────────────────
STOPWORDS_EN = set(stopwords.words('english'))
lemmatizer   = WordNetLemmatizer()

def preprocess(text: str) -> str:
    if not text or not isinstance(text, str): return ''
    text = re.sub(r'http\S+|[^a-zA-Z\s]', ' ', text.lower())
    return ' '.join(lemmatizer.lemmatize(w) for w in text.split()
                    if w not in STOPWORDS_EN and len(w) > 2)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        tok = HF_TOKEN or None
        em  = joblib.load(hf_hub_download(HF_REPO_ID, "emo_model.pkl",       token=tok))
        ev  = joblib.load(hf_hub_download(HF_REPO_ID, "emo_vectorizer.pkl",  token=tok))
        mm  = joblib.load(hf_hub_download(HF_REPO_ID, "mbti_model.pkl",      token=tok))
        mv  = joblib.load(hf_hub_download(HF_REPO_ID, "mbti_vectorizer.pkl", token=tok))
        return em, ev, mm, mv
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        st.stop()

# ── Emotion ───────────────────────────────────────────────────────────────────
EMO_LABEL = {0:'Sedih', 1:'Bahagia', 2:'Cinta', 3:'Marah', 4:'Gelisah', 5:'Terkejut'}
EMO_EMOJI  = {0:'😔', 1:'😊', 2:'❤️', 3:'😠', 4:'😨', 5:'😲'}
EMO_COLOR  = {0:'#5b8dd9', 1:'#f59f00', 2:'#e64980', 3:'#f03e3e', 4:'#7950f2', 5:'#2ecc71'}

LEXICON = {
    'sedih':0,'kecewa':0,'galau':0,'nangis':0,'sepi':0,'kesepian':0,'putus':0,
    'ditinggal':0,'kehilangan':0,'patah hati':0,'down':0,'murung':0,'hampa':0,
    'marah':3,'kesal':3,'benci':3,'jengkel':3,'emosi':3,'kesel':3,'sebel':3,
    'senang':1,'bahagia':1,'happy':1,'excited':1,'lega':1,'bangga':1,'semangat':1,
    'takut':4,'cemas':4,'khawatir':4,'panik':4,'nervous':4,'gelisah':4,'stress':4,
    'cinta':2,'sayang':2,'rindu':2,'kangen':2,'suka':2,'naksir':2,
    'kaget':5,'shock':5,'terkejut':5,'astaga':5,
}

def rule_emotion(text: str):
    t = text.lower()
    for phrase, emo in [
        ('patah hati',0),('putus sama',0),('diselingkuhin',0),('dikhianatin',0),
        ('overthinking',4),('deg degan',4),('was-was',4),
        ('seneng banget',1),('alhamdulillah',1),('lega banget',1),
        ('jatuh cinta',2),('pdkt',2),('naksir',2),
    ]:
        if phrase in t: return emo
    for emo, kws in {
        0:['sedih','nangis','down','putus','ditinggal','sepi','galau','hampa'],
        1:['bahagia','senang','happy','gembira','excited','lega','bangga','yay'],
        2:['cinta','sayang','rindu','kangen','naksir','gebetan'],
        3:['marah','kesal','benci','emosi','jengkel','muak'],
        4:['takut','cemas','khawatir','panik','nervous','gelisah','stress'],
        5:['kaget','shock','terkejut','astaga'],
    }.items():
        if any(k in t for k in kws): return emo
    return None

def predict_emotion(text, em, ev):
    rb = rule_emotion(text)
    if rb is not None: return rb, 0.95
    t = text.lower()
    sc = {}
    for w, e in LEXICON.items():
        if w in t: sc[e] = sc.get(e,0)+1
    if sc: return max(sc, key=sc.get), 0.85
    cl = preprocess(text)
    if not cl: return 1, 0.5
    pr = em.predict_proba(ev.transform([cl]))[0]
    return int(np.argmax(pr)), float(pr.max())

def predict_mbti_from_text(text, mm, mv, hist):
    hist.append(text)
    if len(hist) < 2: return None, 0.0
    cl = preprocess(' '.join(hist[-10:]))
    if not cl: return None, 0.0
    X = mv.transform([cl])
    return mm.predict(X)[0], float(mm.predict_proba(X)[0].max())

# ── MBTI ──────────────────────────────────────────────────────────────────────
MBTI_Q = [
    {"id":1,"q":"Ketika menghadapi masalah besar, kamu lebih suka:","A":"Langsung cari solusi praktis","B":"Merenung dan mikirin berbagai kemungkinan","w":{"A":"S","B":"N"}},
    {"id":2,"q":"Di waktu luang, kamu lebih menikmati:","A":"Kumpul dan sosialisasi sama banyak orang","B":"Waktu sendiri untuk recharge","w":{"A":"E","B":"I"}},
    {"id":3,"q":"Saat ambil keputusan penting, kamu lebih andalkan:","A":"Logika dan analisis objektif","B":"Perasaan dan nilai-nilai pribadi","w":{"A":"T","B":"F"}},
    {"id":4,"q":"Gaya hidupmu sehari-hari:","A":"Terstruktur dengan jadwal jelas","B":"Fleksibel dan ngikutin situasi","w":{"A":"J","B":"P"}},
    {"id":5,"q":"Waktu belajar hal baru:","A":"Langsung praktek dan hands-on","B":"Baca teori dan pahami konsepnya dulu","w":{"A":"S","B":"N"}},
    {"id":6,"q":"Di grup diskusi, biasanya kamu:","A":"Aktif dan sering mulai topik baru","B":"Lebih banyak dengerin, sesekali komentar","w":{"A":"E","B":"I"}},
    {"id":7,"q":"Kalau teman curhat masalah, responsmu:","A":"Langsung kasih solusi praktis","B":"Dengerin dulu dan kasih dukungan emosional","w":{"A":"T","B":"F"}},
    {"id":8,"q":"Menjelang deadline, kamu biasanya:","A":"Selesaikan jauh-jauh hari","B":"Paling produktif di menit-menit akhir","w":{"A":"J","B":"P"}},
    {"id":9,"q":"Kamu lebih tertarik pada:","A":"Fakta, detail konkret, pengalaman nyata","B":"Pola, kemungkinan besar, gambaran besar","w":{"A":"S","B":"N"}},
    {"id":10,"q":"Setelah seharian interaksi sosial, kamu:","A":"Makin semangat dan energized","B":"Capek dan butuh waktu sendiri","w":{"A":"E","B":"I"}},
]

MBTI_DESC = {
    "ISTJ":("The Logistician","Praktis, faktual, terorganisir — orang yang bisa diandalkan."),
    "ISFJ":("The Defender","Penyayang, hangat, selalu siap melindungi orang yang dicintai."),
    "INFJ":("The Advocate","Idealistik, berprinsip, visioner — lihat dunia dengan cara unik."),
    "INTJ":("The Architect","Strategis, logis, selalu mikir jangka panjang. Masterplanner sejati."),
    "ISTP":("The Virtuoso","Pragmatis, fleksibel, jago banget mecahin masalah teknis."),
    "ISFP":("The Adventurer","Artistik, spontan, selalu hidup di saat ini dengan penuh warna."),
    "INFP":("The Mediator","Idealis, empatik, selalu cari makna di balik setiap hal."),
    "INTP":("The Logician","Inovatif, analitis, pencinta teori yang selalu ingin tahu."),
    "ESTP":("The Entrepreneur","Enerjik, action-oriented, persuasif. Suka tantangan nyata."),
    "ESFP":("The Entertainer","Hangat, ramah, suka jadi pusat perhatian yang bikin suasana hidup."),
    "ENFP":("The Campaigner","Antusias, kreatif, inspiratif — bisa nyalaiin semangat orang lain."),
    "ENTP":("The Debater","Cerdas, suka tantangan intelektual, inovatif dalam berpikir."),
    "ESTJ":("The Executive","Tegas, terorganisir, efisien. Natural leader yang result-oriented."),
    "ESFJ":("The Consul","Penyayang, populer, selalu mau bantu. Sosok yang semua orang suka."),
    "ENFJ":("The Protagonist","Karismatik, inspiratif, suka bimbing orang ke versi terbaiknya."),
    "ENTJ":("The Commander","Berwibawa, strategis, pemimpin alami yang penuh visi besar."),
}

DIM_EXP = {
    "E":"Ekstrovert — energi dari interaksi sosial",
    "I":"Introvert — energi dari waktu sendiri",
    "S":"Sensing — fokus pada fakta & detail",
    "N":"Intuition — fokus pada pola & kemungkinan",
    "T":"Thinking — keputusan berdasarkan logika",
    "F":"Feeling — keputusan berdasarkan nilai & perasaan",
    "J":"Judging — suka struktur & perencanaan",
    "P":"Perceiving — fleksibel & adaptif",
}

def analyze_mbti(responses):
    sc = {k:0 for k in "EISNTFJP"}
    for r in responses:
        q = next((x for x in MBTI_Q if x["id"]==r["qid"]), None)
        if q and r["ans"] in q["w"]: sc[q["w"][r["ans"]]] += 1
    return (("E" if sc["E"]>=sc["I"] else "I") +
            ("S" if sc["S"]>=sc["N"] else "N") +
            ("T" if sc["T"]>=sc["F"] else "F") +
            ("J" if sc["J"]>=sc["P"] else "P"))

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah PersonaTalk — teman curhat AI yang terasa seperti sahabat dekat, bukan chatbot generik atau terapis formal.

## Identitas & karakter
Kamu hangat, empatik, kadang playful, selalu genuine dan tidak menghakimi. Kamu tidak menyebut diri sebagai AI, bot, atau assistant.

## Cara berbicara — wajib diikuti
- Bahasa Indonesia sehari-hari, santai, boleh sedikit mix English kalau terasa natural
- Respons pendek: 2-4 kalimat, tidak lebih
- Pembuka harus bervariasi setiap respons: "Duh,", "Ooh,", "Wah,", "Hmm,", "Ya Allah,", "Aduh,", "Serius?", "Oof,"
- Akhiri dengan satu pertanyaan terbuka yang relevan dan BERBEDA dari pertanyaan sebelumnya
- Tidak pakai bullet point, list, atau format kaku

## Konteks — paling kritis
- Baca SELURUH riwayat percakapan sebelum menjawab
- Kalau user sudah menjawab pertanyaanmu → jangan tanya hal yang sama lagi
- Kalau user bercerita hal POSITIF → MATCH the energy, ikut excited
- Kalau topik sudah 4+ exchange → mulai berikan insight ringan atau saran praktis

## Pantangan keras
- JANGAN mengulang pertanyaan yang sudah dijawab user
- JANGAN mengulang kalimat pembuka yang sama dari respons sebelumnya
- JANGAN diagnosis medis atau psikologis
- JANGAN lebih dari 4 kalimat"""

# ── Duplicate Check ───────────────────────────────────────────────────────────
def is_dup(new: str, prev: list, thr=0.50) -> bool:
    if not prev or not new: return False
    nc = new.lower().strip()
    for old in prev[-3:]:
        oc = old.lower().strip()
        if nc == oc: return True
        nw, ow = set(nc.split()), set(oc.split())
        if len(nw & ow) / max(len(nw | ow), 1) > thr: return True
    return False

def clean_text(text: str) -> str:
    text = re.sub(r'\[Emosi user.*?\]', '', text, flags=re.DOTALL)
    text = re.sub(r'##\s+', '', text)
    return text.strip()

# ── Groq AI ───────────────────────────────────────────────────────────────────
def build_groq_messages(history, emotion_id, last_resp):
    emo_name  = EMO_LABEL.get(emotion_id, "Netral")
    no_repeat = ""
    if last_resp:
        no_repeat = " | Jangan ulangi: " + " // ".join(r[:70] for r in last_resp[-2:])
    msgs   = [{"role": "system", "content": SYSTEM_PROMPT}]
    recent = history[-14:-1] if len(history) > 1 else []
    hint   = f"[Emosi user: {emo_name}{no_repeat}]"
    first  = True
    for msg in recent:
        role    = "assistant" if msg['role'] == 'bot' else "user"
        content = msg['content']
        if first and role == "user":
            content = hint + "\n" + content
            first = False
        if msgs and msgs[-1]['role'] == role:
            msgs[-1]['content'] += "\n" + content
        else:
            msgs.append({"role": role, "content": content})
    cur = history[-1]['content'] if history and history[-1]['role'] == 'user' else ""
    if cur:
        if first: cur = hint + "\n" + cur
        if msgs and msgs[-1]['role'] == 'user':
            msgs[-1]['content'] += "\n" + cur
        else:
            msgs.append({"role": "user", "content": cur})
    if not msgs or msgs[-1]['role'] != 'user':
        msgs.append({"role": "user", "content": hint})
    return msgs

def get_ai_response(user_text, emotion_id, history, last_resp):
    last_resp = last_resp or []
    if not _GROQ_OK or not GROQ_KEY:
        st.session_state['_ai_err'] = "Groq key tidak ditemukan."
        return None
    try:
        client   = Groq(api_key=GROQ_KEY)
        msgs     = build_groq_messages(history, emotion_id, last_resp)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=msgs,
            max_tokens=400, temperature=0.85,
        )
        text = clean_text(response.choices[0].message.content)
        if text and len(text) > 10:
            if not is_dup(text, last_resp):
                st.session_state['_provider'] = 'groq'
                st.session_state['_ai_err']   = None
                return text
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": "Gunakan pembuka berbeda dan pertanyaan penutup beda topik."})
            retry = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=msgs,
                max_tokens=400, temperature=0.95,
            )
            text2 = clean_text(retry.choices[0].message.content)
            if text2 and len(text2) > 10:
                st.session_state['_provider'] = 'groq'
                st.session_state['_ai_err']   = None
                return text2
    except Exception as e:
        st.session_state['_ai_err'] = f"Groq error: {str(e)[:150]}"
    return None

def fallback_response(text, emotion, history):
    t  = text.lower()
    fc = ' '.join(m['content'].lower() for m in (history or [])[-5:] if m['role']=='user')
    for kws, opts in {
        ('putus','ditinggal'): ["Duh, berasa ada yang ilang tiba-tiba ya. Kamu lagi sendirian?","Aduh, ini pasti menyakitkan. Udah cerita ke orang terdekat belum?"],
        ('selingkuh','dikhianatin'): ["Oof, diselingkuhin itu rasa sakitnya berlapis. Dia udah tau ketahuan?","Ya Allah, dikhianatin itu beda levelnya. Kamu lagi gimana?"],
        ('rindu','kangen'): ["Kangen yang dalam kayak gini nyesek banget. Kamu kangen orangnya atau momennya?","Ooh, rindu kayak gini biasanya tanda ada hal penting yang kamu miss."],
        ('capek','lelah','burnout'): ["Capek yang kayak gini beda — bukan cuma fisik. Dari kerjaan atau hubungan?","Aduh, burnout kayak gini nyata dan berat. Kapan terakhir beneran istirahat?"],
        ('cemas','khawatir','stress'): ["Gelisah kayak gini nggak enak. Ini overthinking atau ada trigger nyata?","Cemas kayak gini biasanya ada trigger-nya. Soal apa yang paling bikin was-was?"],
        ('crush','naksir','gebetan'): ["Ooh, ada yang spesial nih! Dia udah tau kamu naksir?","Wah, ada yang bikin deg-degan! Gimana interaksi kalian sejauh ini?"],
    }.items():
        if any(k in fc or k in t for k in kws):
            return random.choice(opts)
    return random.choice({
        1:["Wah, ada yang bagus nih! Apaan yang terjadi?","Ooh, cerita dong dari awal!"],
        2:["Ooh, ada yang spesial nih! Gimana dia?","Wah, bikin deg-degan! Cerita lebih dong."],
        3:["Kemarahan kayak gini valid. Ini marah sama orangnya atau situasinya?","Oof, emosi banget. Apaan yang paling bikin gemas?"],
        4:["Gelisah kayak gini nggak enak. Soal apa yang paling bikin khawatir?","Cemas kayak gini berat. Udah berapa lama ngerasa gini?"],
        0:["Duh, kedengarannya berat. Mau cerita lebih? Aku dengerin.","Aduh, ada yang lagi berat dipikul. Dari mana mau mulai?"],
        5:["Serius?! Apaan yang bikin kaget banget?","Astaga, unexpected banget! Gimana ceritanya?"],
    }.get(emotion, ["Hmm, ada apa? Cerita yuk.","Duh, ada sesuatu nih. Aku di sini kok."]))

# ── Mood Donut SVG ────────────────────────────────────────────────────────────
def mood_donut_svg(emo_counts):
    total    = sum(emo_counts.values()) or 1
    pcts     = [emo_counts.get(i, 0) / total for i in range(6)]
    has_data = any(v > 0 for v in emo_counts.values())
    dominant = max(emo_counts, key=emo_counts.get) if has_data else 1
    dom_pct  = int(pcts[dominant] * 100)
    colors   = [EMO_COLOR[i] for i in range(6)]
    emojis   = ['😢','😊','😍','😤','😰','😲']

    r, cx, cy = 45, 50, 50
    inner_r   = 28
    paths, off = "", 0.0
    for i, pct in enumerate(pcts):
        if pct < 0.005:
            off += pct
            continue
        a1 = off * 360 - 90
        a2 = (off + pct) * 360 - 90
        large = 1 if pct > 0.5 else 0
        def pt(a, _r=r, _cx=cx, _cy=cy):
            rad = math.radians(a)
            return _cx + _r*math.cos(rad), _cy + _r*math.sin(rad)
        x1,y1 = pt(a1); x2,y2 = pt(a2)
        paths += f'<path d="M{cx},{cy}L{x1:.1f},{y1:.1f}A{r},{r},0,{large},1,{x2:.1f},{y2:.1f}Z" fill="{colors[i]}" opacity="0.9"/>'
        off += pct
    if not has_data:
        paths = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="rgba(255,255,255,0.2)"/>'

    svg = (f'<svg viewBox="0 0 100 100" width="110" height="110">'
           f'{paths}'
           f'<circle cx="{cx}" cy="{cy}" r="{inner_r}" fill="#1a3f7a"/>'
           f'<text x="{cx}" y="{cy-5}" text-anchor="middle" font-size="14">{emojis[dominant]}</text>'
           f'<text x="{cx}" y="{cy+10}" text-anchor="middle" fill="white" font-size="9" font-weight="bold">{dom_pct}%</text>'
           f'</svg>')

    # Mood list rows
    top3 = sorted([(i, pcts[i]) for i in range(6) if pcts[i] > 0.01], key=lambda x:-x[1])[:3]
    rows = ""
    for i, p in top3:
        rows += (f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
                 f'<span style="font-size:14px;">{emojis[i]}</span>'
                 f'<span style="color:white;font-size:13px;flex:1;">{EMO_LABEL[i]}</span>'
                 f'<span style="color:rgba(255,255,255,0.8);font-size:13px;font-weight:600;">{int(p*100)}%</span>'
                 f'</div>')
    if not rows:
        rows = '<span style="color:rgba(255,255,255,0.6);font-size:12px;">Mulai chat...</span>'

    # Mood label
    dom = dominant
    mood_label = "Mood bagus! 🌟" if dom == 1 else "Mood oke 👍" if dom in [2,5] else "Perlu perhatian 💙"

    return svg, rows, mood_label

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
    margin: 0; padding: 0;
}

.stApp {
    background: linear-gradient(135deg, #1a3a6b 0%, #2a5ba8 50%, #4a90d9 100%) !important;
    min-height: 100vh;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important; visibility: hidden !important;
}

/* Remove default padding */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Streamlit sidebar — hide it, we use custom */
section[data-testid="stSidebar"] { display: none !important; }

/* Button resets */
.stButton > button {
    font-family: 'Poppins', sans-serif !important;
}

/* Chat input hidden — we use custom */
[data-testid="stChatInput"] { display: none !important; }

/* Text input styling */
.stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 14px !important;
    color: #1e293b !important;
    padding: 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.3); border-radius: 4px; }

/* MBTI buttons */
.mbti-btn {
    display: inline-block;
    width: 100%;
    padding: 14px 20px;
    margin-bottom: 10px;
    border-radius: 12px;
    background: rgba(255,255,255,0.15);
    border: 2px solid rgba(255,255,255,0.3);
    color: white;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
    font-family: 'Poppins', sans-serif;
}
.mbti-btn:hover {
    background: rgba(255,255,255,0.25);
    border-color: rgba(255,255,255,0.6);
}
</style>
""", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="PersonaTalk",
        page_icon="✳️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    with st.spinner("Memuat PersonaTalk..."):
        emo_model, emo_vec, mbti_model, mbti_vec = load_models()

    # Session state defaults
    defaults = {
        'messages':     [{'role':'bot','content':'Halo, Talk Friend! 👋 Aku PersonaTalk.\nCerita apa aja, aku dengerin ya!'}],
        'emotion':      1, 'confidence': 0.5,
        'mbti':         None, 'mbti_texts': [],
        'last_bot':     [], 'emo_counts': {i:0 for i in range(6)},
        '_provider':    None, '_ai_err': None,
        'sidebar_open': True,
        'mode':         'curhat',       # 'curhat' | 'mbti'
        'nav':          'dashboard',    # 'dashboard' | 'riwayat' | 'tentang' | 'setelan'
        'mbti_step':    0,
        'mbti_resp':    [],
        'mbti_result':  None,
        'chat_input':   '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    emo_id   = st.session_state.emotion
    sidebar  = st.session_state.sidebar_open
    mode     = st.session_state.mode

    # ── LAYOUT: sidebar + main ────────────────────────────────────────────────
    if sidebar:
        left_w, main_w = 220, 980
        col_sidebar, col_main = st.columns([220, 980])
    else:
        col_sidebar = None
        col_main    = st.columns([1])[0]

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    if sidebar and col_sidebar:
        with col_sidebar:
            nav   = st.session_state.nav
            _mode = st.session_state.mode

            def nav_style(key):
                active = (nav == key) or (_mode == key)
                return (
                    "background:#3b82f6;color:white;border-radius:40px;padding:8px 14px;"
                    "margin-bottom:4px;cursor:pointer;display:block;font-size:13px;"
                    "font-weight:600;border:none;width:100%;text-align:left;"
                ) if active else (
                    "background:transparent;color:#334155;border-radius:40px;padding:8px 14px;"
                    "margin-bottom:4px;cursor:pointer;display:block;font-size:13px;"
                    "font-weight:500;border:none;width:100%;text-align:left;"
                )

            st.markdown(f"""
            <div style="
                background:#f0f4ff;
                border-radius:0 20px 20px 0;
                padding:20px 16px;
                height:100vh;
                position:fixed;
                top:0; left:0;
                width:220px;
                box-shadow:4px 0 20px rgba(0,0,0,0.15);
                display:flex;
                flex-direction:column;
                z-index:100;
                overflow-y:auto;
            ">
                <!-- Logo -->
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">
                    <span style="color:#e53e3e;font-size:22px;font-weight:800;">✳</span>
                    <div>
                        <div style="font-size:15px;font-weight:800;color:#1e293b;line-height:1.1;">persona</div>
                        <div style="font-size:15px;font-weight:800;color:#1e293b;line-height:1.1;">talk</div>
                    </div>
                </div>

                <!-- Search -->
                <div style="
                    background:white;
                    border-radius:20px;
                    padding:8px 12px;
                    display:flex;
                    align-items:center;
                    gap:8px;
                    margin-bottom:20px;
                    border:1px solid #e2e8f0;
                ">
                    <span style="color:#94a3b8;font-size:14px;">🔍</span>
                    <span style="color:#94a3b8;font-size:13px;">Cari...</span>
                </div>

                <!-- Nav -->
                <div style="font-size:11px;color:#94a3b8;font-weight:600;margin-bottom:8px;padding-left:4px;">MENU</div>
            </div>
            """, unsafe_allow_html=True)

            # Nav buttons — menggunakan st.button untuk interaktivitas
            st.markdown('<div style="height:190px;"></div>', unsafe_allow_html=True)

            if st.button("🏠  Dashboard", key="nav_dash", use_container_width=True):
                st.session_state.nav = 'dashboard'
                st.rerun()
            if st.button("🕐  Riwayat Chat", key="nav_riwayat", use_container_width=True):
                st.session_state.nav = 'riwayat'
                st.rerun()
            if st.button("ℹ️  Tentang", key="nav_tentang", use_container_width=True):
                st.session_state.nav = 'tentang'
                st.rerun()

            st.markdown('<div style="margin:12px 0 6px;font-size:11px;color:#94a3b8;font-weight:600;padding-left:4px;">PANEL ANALISIS</div>', unsafe_allow_html=True)

            if st.button("💬  Curhat", key="mode_curhat", use_container_width=True):
                st.session_state.mode = 'curhat'
                st.session_state.nav  = 'dashboard'
                st.rerun()
            if st.button("🧠  Analisis MBTI", key="mode_mbti", use_container_width=True):
                st.session_state.mode    = 'mbti'
                st.session_state.nav     = 'dashboard'
                st.session_state.mbti_step   = 0
                st.session_state.mbti_resp   = []
                st.session_state.mbti_result = None
                st.rerun()

            st.markdown('<div style="flex:1;"></div>', unsafe_allow_html=True)
            st.markdown('<hr style="border-color:#e2e8f0;margin:12px 0;">', unsafe_allow_html=True)

            if st.button("⚙️  Setelan", key="nav_setelan", use_container_width=True):
                st.session_state.nav = 'setelan'
                st.rerun()

            # Style the sidebar buttons
            st.markdown(f"""
            <style>
            [data-testid="stButton"] > button {{
                font-family: 'Poppins', sans-serif !important;
                border-radius: 40px !important;
                border: none !important;
                font-size: 13px !important;
                font-weight: 500 !important;
                text-align: left !important;
                padding: 8px 14px !important;
                margin-bottom: 2px !important;
                color: #334155 !important;
                background: transparent !important;
                transition: all 0.15s !important;
            }}
            [data-testid="stButton"] > button:hover {{
                background: #e8f0fe !important;
                color: #1e40af !important;
            }}
            /* Active mode button highlight */
            {"" }
            </style>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN AREA
    # ══════════════════════════════════════════════════════════════════════════
    with col_main:
        main_pad = "220px" if sidebar else "0px"

        st.markdown(f"""
        <div style="
            margin-left:{main_pad};
            padding:24px 28px;
            min-height:100vh;
        ">
        """, unsafe_allow_html=True)

        # ── TOP BAR ──────────────────────────────────────────────────────────
        topbar_col1, topbar_col2 = st.columns([1, 3])
        with topbar_col1:
            toggle_label = "✕" if sidebar else "☰"
            if st.button(toggle_label, key="sidebar_toggle"):
                st.session_state.sidebar_open = not sidebar
                st.rerun()

        st.markdown("""
        <style>
        div[data-testid="stButton"]:has(button[kind="secondary"]) button {
            background: rgba(255,255,255,0.2) !important;
            color: white !important;
            border-radius: 10px !important;
            font-size: 18px !important;
            width: 42px !important;
            height: 42px !important;
            padding: 0 !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """, unsafe_allow_html=True)

        nav = st.session_state.nav

        # ── DASHBOARD ─────────────────────────────────────────────────────────
        if nav == 'dashboard':
            # Header + Mood Panel
            svg, mood_rows, mood_label = mood_donut_svg(st.session_state.emo_counts)
            h_col1, h_col2 = st.columns([3, 2])

            with h_col1:
                st.markdown("""
                <div style="padding:10px 0 20px;">
                    <div style="font-size:2.4rem;font-weight:800;color:white;line-height:1.15;">
                        Halo, Talk Friend! 👋
                    </div>
                    <div style="font-size:1rem;color:rgba(255,255,255,0.8);margin-top:6px;">
                        Bagaimana kabar mu hari ini?
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with h_col2:
                st.markdown(f"""
                <div style="
                    background:rgba(255,255,255,0.12);
                    border-radius:20px;
                    padding:16px 20px;
                    backdrop-filter:blur(10px);
                    border:1px solid rgba(255,255,255,0.2);
                ">
                    <div style="color:white;font-weight:700;font-size:13px;margin-bottom:10px;">
                        🎯 Detektor Mood Hari Ini
                    </div>
                    <div style="display:flex;align-items:center;gap:14px;">
                        <div>{svg}</div>
                        <div style="flex:1;">
                            {mood_rows}
                            <div style="margin-top:8px;color:rgba(255,255,255,0.9);
                                        font-size:12px;font-weight:600;">{mood_label}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

            # ── MODE: CURHAT ──────────────────────────────────────────────────
            if mode == 'curhat':
                st.markdown("""
                <div style="
                    background:white;
                    border-radius:20px;
                    padding:20px 24px 80px;
                    box-shadow:0 8px 32px rgba(0,0,0,0.12);
                    min-height:420px;
                    position:relative;
                ">
                <div style="font-size:15px;font-weight:700;color:#1e293b;margin-bottom:16px;">
                    💬 Curhat
                </div>
                """, unsafe_allow_html=True)

                # Chat messages
                msgs = st.session_state.messages
                for msg in msgs:
                    if msg['role'] == 'user':
                        st.markdown(f"""
                        <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
                            <div style="
                                background:#c5cff5;
                                border-radius:18px 18px 4px 18px;
                                padding:10px 16px;
                                max-width:70%;
                                font-size:14px;
                                color:#1e293b;
                                line-height:1.5;
                            ">{msg['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="display:flex;justify-content:flex-start;margin-bottom:10px;">
                            <div style="
                                background:#f0f4ff;
                                border-radius:18px 18px 18px 4px;
                                padding:10px 16px;
                                max-width:70%;
                                font-size:14px;
                                color:#1e293b;
                                line-height:1.5;
                                border:1px solid #e2e8f0;
                            ">{msg['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Chat Input Bar ────────────────────────────────────────────
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                inp_col1, inp_col2, inp_col3 = st.columns([0.5, 8, 1])

                with inp_col1:
                    st.markdown("""
                    <div style="
                        width:38px;height:38px;
                        background:rgba(255,255,255,0.2);
                        border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-size:18px;
                        cursor:pointer;margin-top:4px;
                    ">+</div>
                    """, unsafe_allow_html=True)

                with inp_col2:
                    user_input = st.text_input(
                        label="",
                        placeholder="Ketik pesan...",
                        key="chat_text_input",
                        label_visibility="collapsed",
                    )
                    st.markdown("""
                    <style>
                    div[data-testid="stTextInput"] > div {
                        background: white !important;
                        border-radius: 24px !important;
                        padding: 6px 16px !important;
                        border: none !important;
                        box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
                    }
                    div[data-testid="stTextInput"] > div > div > input {
                        color: #1e293b !important;
                        font-size: 14px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                with inp_col3:
                    send_btn = st.button("✈️", key="send_btn")

                # Process input
                if (send_btn or (user_input and user_input.strip() and
                    st.session_state.get('_last_input') != user_input)) and user_input and user_input.strip():

                    st.session_state['_last_input'] = user_input
                    user_text = user_input.strip()
                    st.session_state.messages.append({'role':'user','content':user_text})

                    emo_id, conf = predict_emotion(user_text, emo_model, emo_vec)
                    st.session_state.emotion    = emo_id
                    st.session_state.confidence = conf
                    st.session_state.emo_counts[emo_id] = st.session_state.emo_counts.get(emo_id,0)+1

                    mbti_p, mbti_c = predict_mbti_from_text(user_text, mbti_model, mbti_vec, st.session_state.mbti_texts)
                    if mbti_p and mbti_c > 0.3:
                        st.session_state.mbti = mbti_p

                    response = get_ai_response(user_text, emo_id, st.session_state.messages, st.session_state.last_bot)
                    if not response:
                        response = fallback_response(user_text, emo_id, st.session_state.messages)

                    st.session_state.last_bot.append(response)
                    st.session_state.messages.append({'role':'bot','content':response})
                    st.rerun()

            # ── MODE: MBTI ────────────────────────────────────────────────────
            elif mode == 'mbti':
                st.markdown("""
                <div style="
                    background:white;
                    border-radius:20px;
                    padding:28px 32px;
                    box-shadow:0 8px 32px rgba(0,0,0,0.12);
                    min-height:420px;
                ">
                <div style="font-size:15px;font-weight:700;color:#1e293b;margin-bottom:20px;">
                    🧠 Analisis MBTI
                </div>
                """, unsafe_allow_html=True)

                step   = st.session_state.mbti_step
                result = st.session_state.mbti_result

                if result:
                    # Show result
                    name, desc = MBTI_DESC.get(result, ("Unknown", "Tipe kepribadian unik."))
                    dims = "".join(
                        f'<div style="padding:8px 14px;background:#f0f4ff;border-radius:10px;'
                        f'margin-bottom:8px;font-size:13px;color:#334155;">'
                        f'<b style="color:#3b82f6;">{d}</b> — {DIM_EXP.get(d,"")}</div>'
                        for d in result if d in DIM_EXP
                    )
                    st.markdown(f"""
                    <div style="text-align:center;padding:20px 0;">
                        <div style="font-size:3rem;font-weight:800;color:#3b82f6;margin-bottom:8px;">{result}</div>
                        <div style="font-size:1.1rem;font-weight:700;color:#1e293b;margin-bottom:6px;">{name}</div>
                        <div style="font-size:14px;color:#64748b;margin-bottom:24px;">{desc}</div>
                        <div style="text-align:left;max-width:500px;margin:0 auto;">{dims}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    if st.button("🔄 Ulangi Analisis", key="reset_mbti"):
                        st.session_state.mbti_step   = 0
                        st.session_state.mbti_resp   = []
                        st.session_state.mbti_result = None
                        st.rerun()

                elif step < len(MBTI_Q):
                    q = MBTI_Q[step]
                    progress = (step) / len(MBTI_Q)

                    st.markdown(f"""
                    <div style="margin-bottom:16px;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:12px;color:#94a3b8;margin-bottom:6px;">
                            <span>Pertanyaan {step+1} dari {len(MBTI_Q)}</span>
                            <span>{int(progress*100)}%</span>
                        </div>
                        <div style="background:#e2e8f0;border-radius:10px;height:6px;">
                            <div style="background:#3b82f6;border-radius:10px;height:6px;
                                        width:{int(progress*100)}%;transition:width 0.3s;"></div>
                        </div>
                    </div>
                    <div style="font-size:15px;font-weight:600;color:#1e293b;
                                margin-bottom:24px;line-height:1.6;">
                        {q['q']}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    btn_a, btn_b = st.columns(2)
                    with btn_a:
                        if st.button(f"A. {q['A']}", key=f"mbti_a_{step}", use_container_width=True):
                            st.session_state.mbti_resp.append({"qid": q["id"], "ans": "A"})
                            st.session_state.mbti_step += 1
                            if st.session_state.mbti_step >= len(MBTI_Q):
                                st.session_state.mbti_result = analyze_mbti(st.session_state.mbti_resp)
                                st.session_state.mbti = st.session_state.mbti_result
                            st.rerun()
                    with btn_b:
                        if st.button(f"B. {q['B']}", key=f"mbti_b_{step}", use_container_width=True):
                            st.session_state.mbti_resp.append({"qid": q["id"], "ans": "B"})
                            st.session_state.mbti_step += 1
                            if st.session_state.mbti_step >= len(MBTI_Q):
                                st.session_state.mbti_result = analyze_mbti(st.session_state.mbti_resp)
                                st.session_state.mbti = st.session_state.mbti_result
                            st.rerun()

                    # Style MBTI buttons
                    st.markdown("""
                    <style>
                    div[data-testid="stButton"] > button {
                        background: #f0f4ff !important;
                        color: #1e293b !important;
                        border: 2px solid #e2e8f0 !important;
                        border-radius: 14px !important;
                        padding: 14px 16px !important;
                        font-size: 13px !important;
                        font-weight: 500 !important;
                        text-align: left !important;
                        height: auto !important;
                        white-space: normal !important;
                        line-height: 1.5 !important;
                        transition: all 0.15s !important;
                    }
                    div[data-testid="stButton"] > button:hover {
                        background: #3b82f6 !important;
                        color: white !important;
                        border-color: #3b82f6 !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

        # ── RIWAYAT ───────────────────────────────────────────────────────────
        elif nav == 'riwayat':
            st.markdown("""
            <div style="background:white;border-radius:20px;padding:28px;
                        box-shadow:0 8px 32px rgba(0,0,0,0.12);">
            <div style="font-size:18px;font-weight:700;color:#1e293b;margin-bottom:20px;">
                🕐 Riwayat Chat
            </div>
            """, unsafe_allow_html=True)
            msgs = [m for m in st.session_state.messages if m['role']=='user']
            if msgs:
                for i, m in enumerate(msgs[-10:], 1):
                    st.markdown(f"""
                    <div style="padding:10px 14px;background:#f8fafc;border-radius:10px;
                                margin-bottom:8px;font-size:13px;color:#334155;
                                border-left:3px solid #3b82f6;">
                        <b>#{i}</b> {m['content'][:80]}{'...' if len(m['content'])>80 else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#94a3b8;font-size:14px;">Belum ada riwayat chat.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── TENTANG ───────────────────────────────────────────────────────────
        elif nav == 'tentang':
            st.markdown("""
            <div style="background:white;border-radius:20px;padding:28px;
                        box-shadow:0 8px 32px rgba(0,0,0,0.12);">
            <div style="font-size:18px;font-weight:700;color:#1e293b;margin-bottom:16px;">
                ℹ️ Tentang PersonaTalk
            </div>
            <div style="font-size:14px;color:#475569;line-height:1.8;">
                <b>PersonaTalk</b> adalah teman curhat AI yang dirancang untuk mendengarkan
                tanpa menghakimi. Aplikasi ini menggunakan model ML untuk mendeteksi emosi
                dan kepribadian MBTI dari percakapan, serta AI Groq (LLaMA 3.3 70B) untuk
                menghasilkan respons yang hangat dan empatik.<br><br>
                <b>Fitur:</b><br>
                • 🎯 Deteksi emosi real-time<br>
                • 🧠 Analisis kepribadian MBTI<br>
                • 💬 Curhat dengan AI yang memahami konteks<br>
                • 📊 Mood tracker harian
            </div>
            </div>
            """, unsafe_allow_html=True)

        # ── SETELAN ───────────────────────────────────────────────────────────
        elif nav == 'setelan':
            st.markdown("""
            <div style="background:white;border-radius:20px;padding:28px;
                        box-shadow:0 8px 32px rgba(0,0,0,0.12);">
            <div style="font-size:18px;font-weight:700;color:#1e293b;margin-bottom:20px;">
                ⚙️ Setelan
            </div>
            """, unsafe_allow_html=True)

            provider = st.session_state.get('_provider') or 'Belum aktif'
            err      = st.session_state.get('_ai_err')
            mbti_r   = st.session_state.get('mbti') or 'Belum dianalisis'

            st.markdown(f"""
            <div style="font-size:13px;color:#475569;">
                <div style="padding:10px 0;border-bottom:1px solid #f1f5f9;">
                    <b>AI Provider:</b> {provider}
                </div>
                <div style="padding:10px 0;border-bottom:1px solid #f1f5f9;">
                    <b>MBTI Kamu:</b> {mbti_r}
                </div>
                <div style="padding:10px 0;border-bottom:1px solid #f1f5f9;">
                    <b>Total Pesan:</b> {len([m for m in st.session_state.messages if m['role']=='user'])}
                </div>
                {'<div style="padding:10px 0;color:#ef4444;"><b>Error:</b> '+err+'</div>' if err else ''}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Reset Semua Chat", key="reset_chat"):
                st.session_state.messages  = [{'role':'bot','content':'Halo, Talk Friend! 👋 Cerita apa aja, aku dengerin ya!'}]
                st.session_state.emo_counts = {i:0 for i in range(6)}
                st.session_state.last_bot   = []
                st.session_state.mbti_texts = []
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
