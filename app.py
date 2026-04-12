"""
🐼 PersonaTalk v3
- Chat bubbles rendered INSIDE white card (one HTML string)
- Dark sidebar (native st.sidebar) with smooth Streamlit toggle
- Sidebar: mode switch (Curhat / MBTI), mood panel, MBTI badge
- Models loaded from HuggingFace
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

# ── Config ─────────────────────────────────────────────────────────────────────
HF_REPO_ID = "Jooou139/personatalk"
GROQ_KEY   = st.secrets.get("GROQ_API_KEY", "")
HF_TOKEN   = st.secrets.get("HF_TOKEN", "")

# ── Preprocessing ──────────────────────────────────────────────────────────────
STOPWORDS_EN = set(stopwords.words('english'))
lemmatizer   = WordNetLemmatizer()

def preprocess(text: str) -> str:
    if not text or not isinstance(text, str): return ''
    text = re.sub(r'http\S+|[^a-zA-Z\s]', ' ', text.lower())
    return ' '.join(lemmatizer.lemmatize(w) for w in text.split()
                    if w not in STOPWORDS_EN and len(w) > 2)

# ── Load Models ────────────────────────────────────────────────────────────────
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

# ── Emotion ────────────────────────────────────────────────────────────────────
EMO_LABEL = {0:'Sedih', 1:'Bahagia', 2:'Cinta', 3:'Marah', 4:'Gelisah', 5:'Terkejut'}
EMO_EMOJI = {0:'😔', 1:'😊', 2:'❤️',  3:'😠', 4:'😨', 5:'😲'}
EMO_COLOR = {0:'#5b8dd9', 1:'#f59f00', 2:'#e64980', 3:'#f03e3e', 4:'#7950f2', 5:'#2ecc71'}

LEXICON = {
    'sedih':0,'kecewa':0,'galau':0,'nangis':0,'sepi':0,'kesepian':0,'putus':0,
    'ditinggal':0,'kehilangan':0,'patah hati':0,'down':0,'murung':0,'hampa':0,
    'nyesel':0,'frustasi':0,'hopeless':0,'bingung':0,'sendirian':0,
    'marah':3,'kesal':3,'benci':3,'jengkel':3,'emosi':3,'kesel':3,'sebel':3,
    'muak':3,'tersinggung':3,'dibohongi':3,'ditipu':3,
    'senang':1,'bahagia':1,'happy':1,'excited':1,'lega':1,'bangga':1,'semangat':1,
    'alhamdulillah':1,'syukur':1,'gembira':1,
    'takut':4,'cemas':4,'khawatir':4,'panik':4,'nervous':4,'gelisah':4,'stress':4,
    'anxious':4,'anxiety':4,'overthinking':4,'was-was':4,'resah':4,
    'cinta':2,'sayang':2,'rindu':2,'kangen':2,'suka':2,'naksir':2,
    'kaget':5,'shock':5,'terkejut':5,'astaga':5,
}

def rule_emotion(text: str):
    t = text.lower()
    for phrase, emo in [
        ('patah hati',0),('putus sama',0),('diselingkuhin',0),('dikhianatin',0),
        ('sakit hati',0),('ditinggal pergi',0),
        ('overthinking',4),('deg degan',4),('was-was',4),('ga tenang',4),
        ('nggak tenang',4),('khawatir banget',4),
        ('seneng banget',1),('alhamdulillah',1),('lega banget',1),
        ('jatuh cinta',2),('pdkt',2),('naksir',2),
        ('marah banget',3),('kesel banget',3),('nggak adil',3),
        ('kaget banget',5),('ga nyangka',5),('nggak nyangka',5),
    ]:
        if phrase in t: return emo
    for emo, kws in {
        0:['sedih','nangis','down','putus','ditinggal','sepi','galau','hampa','nyesel'],
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

def predict_mbti_passive(text, mm, mv, hist):
    hist.append(text)
    if len(hist) < 2: return None, 0.0
    cl = preprocess(' '.join(hist[-10:]))
    if not cl: return None, 0.0
    X = mv.transform([cl])
    return mm.predict(X)[0], float(mm.predict_proba(X)[0].max())

# ── MBTI ───────────────────────────────────────────────────────────────────────
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

# ── Groq AI ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah PersonaTalk — teman curhat AI yang terasa seperti sahabat dekat.

Kamu hangat, empatik, kadang playful, tidak menghakimi. Bukan AI/bot/assistant.

Cara berbicara:
- Bahasa Indonesia santai, boleh mix English
- 2-4 kalimat, tidak lebih
- Pembuka bervariasi: "Duh,", "Ooh,", "Wah,", "Hmm,", "Ya Allah,", "Aduh,", "Serius?", "Oof,"
- Akhiri dengan satu pertanyaan terbuka relevan
- Tidak pakai bullet point

Pantangan: jangan ulangi pertanyaan yang sudah dijawab, jangan diagnosis, jangan lebih dari 4 kalimat."""

def is_dup(new, prev, thr=0.50):
    if not prev or not new: return False
    nc = new.lower().strip()
    for old in prev[-3:]:
        oc = old.lower().strip()
        if nc == oc: return True
        nw, ow = set(nc.split()), set(oc.split())
        if len(nw & ow) / max(len(nw | ow), 1) > thr: return True
    return False

def clean_text(text):
    text = re.sub(r'\[Emosi user.*?\]', '', text, flags=re.DOTALL)
    return re.sub(r'##\s+', '', text).strip()

def get_ai_response(user_text, emotion_id, history, last_resp):
    if not _GROQ_OK or not GROQ_KEY: return None
    emo_name  = EMO_LABEL.get(emotion_id, "Netral")
    no_repeat = (" | Jangan ulangi: " + " // ".join(r[:70] for r in last_resp[-2:])) if last_resp else ""
    hint      = f"[Emosi user: {emo_name}{no_repeat}]"
    msgs      = [{"role":"system","content":SYSTEM_PROMPT}]
    recent    = history[-14:-1] if len(history) > 1 else []
    first     = True
    for msg in recent:
        role    = "assistant" if msg['role']=='bot' else "user"
        content = msg['content']
        if first and role=="user": content = hint+"\n"+content; first=False
        if msgs and msgs[-1]['role']==role: msgs[-1]['content'] += "\n"+content
        else: msgs.append({"role":role,"content":content})
    cur = history[-1]['content'] if history and history[-1]['role']=='user' else ""
    if cur:
        if first: cur = hint+"\n"+cur
        if msgs and msgs[-1]['role']=='user': msgs[-1]['content'] += "\n"+cur
        else: msgs.append({"role":"user","content":cur})
    if not msgs or msgs[-1]['role']!='user':
        msgs.append({"role":"user","content":hint})
    try:
        client = Groq(api_key=GROQ_KEY)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=msgs, max_tokens=400, temperature=0.85)
        text   = clean_text(resp.choices[0].message.content)
        if text and len(text) > 10:
            if not is_dup(text, last_resp): return text
            msgs += [{"role":"assistant","content":text},
                     {"role":"user","content":"Gunakan pembuka berbeda dan pertanyaan penutup beda topik."}]
            r2 = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=msgs, max_tokens=400, temperature=0.95)
            t2 = clean_text(r2.choices[0].message.content)
            if t2 and len(t2) > 10: return t2
    except Exception as e:
        st.session_state['_ai_err'] = str(e)[:150]
    return None

def fallback_response(text, emotion, history):
    t  = text.lower()
    fc = ' '.join(m['content'].lower() for m in (history or [])[-5:] if m['role']=='user')
    for kws, opts in {
        ('putus','ditinggal'):    ["Duh, berasa ada yang ilang tiba-tiba ya. Kamu lagi sendirian?","Aduh, ini pasti menyakitkan. Udah cerita ke orang terdekat belum?"],
        ('selingkuh','khianatin'):["Oof, diselingkuhin itu rasa sakitnya berlapis. Dia udah tau ketahuan?","Ya Allah, dikhianatin itu beda levelnya. Kamu lagi gimana?"],
        ('rindu','kangen'):       ["Kangen yang dalam kayak gini nyesek banget. Kamu kangen orangnya atau momennya?"],
        ('capek','lelah'):        ["Capek yang kayak gini beda — bukan cuma fisik. Dari kerjaan atau hubungan?"],
        ('cemas','khawatir'):     ["Gelisah kayak gini nggak enak. Ini overthinking atau ada trigger nyata?"],
        ('naksir','gebetan'):     ["Ooh, ada yang spesial nih! Dia udah tau kamu naksir?"],
    }.items():
        if any(k in fc or k in t for k in kws): return random.choice(opts)
    return random.choice({
        1:["Wah, ada yang bagus nih! Apaan yang terjadi?"],
        2:["Ooh, ada yang spesial nih! Cerita dong."],
        3:["Kemarahan kayak gini valid. Ini marah sama orangnya atau situasinya?"],
        4:["Gelisah kayak gini nggak enak. Soal apa yang paling bikin khawatir?"],
        0:["Duh, kedengarannya berat. Mau cerita lebih? Aku dengerin."],
        5:["Serius?! Apaan yang bikin kaget banget?"],
    }.get(emotion, ["Hmm, ada apa? Cerita yuk."]))

# ── Mood Donut SVG ─────────────────────────────────────────────────────────────
def mood_donut_svg(emo_counts, size=80):
    total    = sum(emo_counts.values()) or 1
    pcts     = [emo_counts.get(i,0)/total for i in range(6)]
    has_data = any(v>0 for v in emo_counts.values())
    dominant = max(emo_counts, key=emo_counts.get) if has_data else 1
    dom_pct  = int(pcts[dominant]*100)
    colors   = [EMO_COLOR[i] for i in range(6)]
    emojis   = ['😢','😊','😍','😤','😰','😲']
    r, cx, cy, inner_r = 45, 50, 50, 28
    paths, off = "", 0.0
    for i, pct in enumerate(pcts):
        if pct < 0.005: off += pct; continue
        a1, a2 = off*360-90, (off+pct)*360-90
        large  = 1 if pct>0.5 else 0
        def pt(a, _r=r, _cx=cx, _cy=cy):
            rad = math.radians(a)
            return _cx+_r*math.cos(rad), _cy+_r*math.sin(rad)
        x1,y1 = pt(a1); x2,y2 = pt(a2)
        paths += f'<path d="M{cx},{cy}L{x1:.1f},{y1:.1f}A{r},{r},0,{large},1,{x2:.1f},{y2:.1f}Z" fill="{colors[i]}" opacity="0.9"/>'
        off += pct
    if not has_data:
        paths = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="rgba(255,255,255,0.12)"/>'
    return f"""<svg viewBox="0 0 100 100" width="{size}" height="{size}">
        {paths}
        <circle cx="{cx}" cy="{cy}" r="{inner_r}" fill="#0f172a"/>
        <text x="{cx}" y="{cy-5}" text-anchor="middle" font-size="14">{emojis[dominant]}</text>
        <text x="{cx}" y="{cy+10}" text-anchor="middle" fill="white" font-size="9" font-weight="bold">{dom_pct}%</text>
    </svg>"""

# ── Build ALL chat bubbles as ONE HTML string ──────────────────────────────────
def build_chat_html(messages):
    html = ""
    for msg in messages:
        # Escape HTML special chars, preserve newlines as <br>
        content = (msg['content']
                   .replace('&','&amp;')
                   .replace('<','&lt;')
                   .replace('>','&gt;')
                   .replace('\n','<br>'))
        if msg['role'] == 'user':
            html += f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
                <div style="
                    background:linear-gradient(135deg,#3b82f6,#2a5ba8);
                    border-radius:18px 18px 4px 18px;
                    padding:11px 16px;max-width:70%;
                    font-size:14px;color:white;line-height:1.6;
                    box-shadow:0 2px 10px rgba(59,130,246,0.35);
                    word-wrap:break-word;
                ">{content}</div>
            </div>"""
        else:
            html += f"""
            <div style="display:flex;justify-content:flex-start;margin-bottom:10px;">
                <div style="
                    background:linear-gradient(135deg,#4a90d9,#2a5ba8);
                    border-radius:18px 18px 18px 4px;
                    padding:11px 16px;max-width:70%;
                    font-size:14px;color:white;line-height:1.6;
                    box-shadow:0 2px 8px rgba(42,91,168,0.2);
                    border:1px solid rgba(255,255,255,0.15);
                    word-wrap:break-word;
                ">{content}</div>
            </div>"""
    return html

# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }

/* App background */
.stApp { background: linear-gradient(135deg,#1a3a6b 0%,#2a5ba8 50%,#4a90d9 100%) !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

/* Main block */
.main .block-container { padding:1.5rem 2rem 2rem !important; max-width:100% !important; }

/* ── SIDEBAR (dark) ── */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
    min-width: 250px !important;
    max-width: 250px !important;
}
section[data-testid="stSidebar"] > div { padding: 1.4rem 1rem !important; }

/* All sidebar text white */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.85) !important; }

/* Sidebar toggle arrow */
[data-testid="collapsedControl"] {
    background: #1e3a5f !important;
    border-radius: 0 10px 10px 0 !important;
    color: white !important;
    transition: background 0.25s !important;
}
[data-testid="collapsedControl"]:hover { background: #2a5ba8 !important; }

/* Sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.65) !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 10px 14px !important;
    width: 100% !important;
    justify-content: flex-start !important;
    transition: all 0.18s ease !important;
    margin-bottom: 3px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    transform: translateX(2px) !important;
}

/* Active nav button */
.nav-active > .stButton > button {
    background: #1d4ed8 !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(29,78,216,0.4) !important;
}
.nav-active > .stButton > button:hover {
    background: #1e40af !important;
    color: white !important;
    transform: none !important;
}

/* Send button */
.send-btn .stButton > button {
    background: #3b82f6 !important;
    border: none !important;
    border-radius: 50% !important;
    color: white !important;
    font-size: 18px !important;
    width: 46px !important;
    height: 46px !important;
    padding: 0 !important;
    min-height: unset !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.5) !important;
}
.send-btn .stButton > button:hover {
    background: #2563eb !important;
    transform: scale(1.08) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.6) !important;
}

/* MBTI choice buttons */
.mbti-btn .stButton > button {
    background: white !important;
    color: #1e293b !important;
    border: 2px solid #e2e8f0 !important;
    border-radius: 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 56px !important;
    padding: 13px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    transition: all 0.18s !important;
}
.mbti-btn .stButton > button:hover {
    background: #3b82f6 !important;
    color: white !important;
    border-color: #3b82f6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.35) !important;
}

/* Action / reset buttons */
.action-btn .stButton > button {
    background: rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    font-size: 13px !important;
    padding: 10px 18px !important;
    transition: all 0.18s !important;
}
.action-btn .stButton > button:hover {
    background: rgba(255,255,255,0.14) !important;
    color: white !important;
    transform: none !important;
    box-shadow: none !important;
}

/* White card action buttons */
.action-btn-white .stButton > button {
    background: white !important;
    color: #334155 !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    font-size: 13px !important;
    padding: 10px 18px !important;
    transition: all 0.18s !important;
}
.action-btn-white .stButton > button:hover {
    background: #dbeafe !important;
    color: #1d4ed8 !important;
    border-color: #bfdbfe !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: white !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 28px !important;
    color: #1e293b !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 20px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
    transition: all 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }
.stTextInput label { display: none !important; }

/* Form */
[data-testid="stForm"] { background:transparent !important; border:none !important; padding:0 !important; }

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#3b82f6,#60a5fa) !important;
    border-radius: 10px !important;
}
.stProgress > div > div { background:rgba(255,255,255,0.1) !important; border-radius:10px !important; }

/* Sidebar progress (confidence bar) */
section[data-testid="stSidebar"] .stProgress > div > div > div > div {
    background: linear-gradient(90deg,#f59f00,#fbbf24) !important;
}
section[data-testid="stSidebar"] .stProgress > div > div {
    background: rgba(255,255,255,0.12) !important;
}

/* Metric */
[data-testid="stMetricValue"] { color: #60a5fa !important; font-weight:700 !important; }
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.5) !important; font-size:12px !important; }

/* HR */
hr { border-color: rgba(255,255,255,0.08) !important; margin:12px 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(59,130,246,0.4); border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="PersonaTalk",
        page_icon="✳️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    with st.spinner("Memuat PersonaTalk..."):
        emo_model, emo_vec, mbti_model, mbti_vec = load_models()

    defaults = {
        'messages':    [{'role':'bot','content':'Halo, Talk Friend! 👋 Aku PersonaTalk.\nCerita apa aja, aku dengerin ya!'}],
        'emotion':     1, 'confidence': 0.5,
        'mbti':        None, 'mbti_texts': [],
        'last_bot':    [], 'emo_counts': {i:0 for i in range(6)},
        '_ai_err':     None,
        'mode':        'curhat',
        'mbti_step':   0, 'mbti_resp': [], 'mbti_result': None,
        '_last_input': '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    mode = st.session_state.mode
    emo  = st.session_state.emotion

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:

        # Logo
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:22px;padding-left:2px;">
            <span style="color:#ef4444;font-size:22px;font-weight:900;line-height:1;">✳</span>
            <div style="line-height:1.2;">
                <div style="font-size:15px;font-weight:800;color:white;">persona</div>
                <div style="font-size:15px;font-weight:800;color:white;">talk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Mode Switch ───────────────────────────────────────────────────────
        st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.35);font-weight:700;letter-spacing:0.1em;margin-bottom:8px;padding-left:2px;">MODE INTERAKSI</div>', unsafe_allow_html=True)

        for key, icon, label in [('curhat','💬','Curhat'), ('mbti','🧠','Analisis MBTI')]:
            is_active = (mode == key)
            st.markdown('<div class="nav-active">' if is_active else '<div>', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"mode_{key}", use_container_width=True):
                st.session_state.mode = key
                if key == 'mbti':
                    st.session_state.mbti_step   = 0
                    st.session_state.mbti_resp   = []
                    st.session_state.mbti_result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)

        # ── Mood Terdeteksi ───────────────────────────────────────────────────
        st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.35);font-weight:700;letter-spacing:0.1em;margin-bottom:10px;padding-left:2px;">MOOD TERDETEKSI</div>', unsafe_allow_html=True)

        emo_name  = EMO_LABEL.get(emo, 'Netral')
        emo_emoji = EMO_EMOJI.get(emo, '😊')
        emo_color = EMO_COLOR.get(emo, '#3b82f6')

        # Mood badge with colored left border
        st.markdown(f"""
        <div style="
            background:rgba(255,255,255,0.06);
            border-left:4px solid {emo_color};
            border-radius:0 12px 12px 0;
            padding:12px 14px;
            margin-bottom:14px;
            display:flex;align-items:center;gap:10px;
        ">
            <span style="font-size:1.8rem;">{emo_emoji}</span>
            <span style="font-size:1rem;font-weight:700;color:white;">{emo_name}</span>
        </div>
        """, unsafe_allow_html=True)

        # Donut chart + confidence number
        donut    = mood_donut_svg(st.session_state.emo_counts, size=78)
        conf_pct = int(st.session_state.confidence * 100)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
            <div>{donut}</div>
            <div style="flex:1;">
                <div style="color:rgba(255,255,255,0.45);font-size:11px;margin-bottom:3px;">Confidence</div>
                <div style="font-size:1.4rem;font-weight:800;color:white;">{conf_pct}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(st.session_state.confidence)

        st.markdown('<hr>', unsafe_allow_html=True)

        # ── MBTI Badge ────────────────────────────────────────────────────────
        st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.35);font-weight:700;letter-spacing:0.1em;margin-bottom:10px;padding-left:2px;">KEPRIBADIAN MBTI</div>', unsafe_allow_html=True)

        mbti_val = st.session_state.mbti
        if mbti_val:
            mbti_name = MBTI_DESC.get(mbti_val, ("?",""))[0]
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,#1d4ed8,#3b82f6);
                border-radius:14px;padding:14px 16px;text-align:center;
            ">
                <div style="font-size:2rem;font-weight:800;color:white;letter-spacing:0.05em;">{mbti_val}</div>
                <div style="font-size:11px;color:rgba(255,255,255,0.75);margin-top:3px;">{mbti_name}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background:rgba(255,255,255,0.04);
                border:1.5px dashed rgba(255,255,255,0.12);
                border-radius:12px;padding:14px;text-align:center;
            ">
                <div style="color:rgba(255,255,255,0.35);font-size:12px;">Belum terdeteksi</div>
                <div style="color:rgba(255,255,255,0.25);font-size:11px;margin-top:2px;">
                    Mulai curhat atau ikuti quiz!
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)

        # ── Reset button ──────────────────────────────────────────────────────
        st.markdown('<div class="action-btn">', unsafe_allow_html=True)
        if st.button("🗑️  Reset Chat", key="reset_chat", use_container_width=True):
            st.session_state.messages   = [{'role':'bot','content':'Halo, Talk Friend! 👋 Cerita apa aja, aku dengerin ya!'}]
            st.session_state.emo_counts = {i:0 for i in range(6)}
            st.session_state.last_bot   = []
            st.session_state.mbti_texts = []
            st.session_state._ai_err    = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN AREA
    # ══════════════════════════════════════════════════════════════════════════

    # Page header
    st.markdown("""
    <div style="padding:4px 0 18px;">
        <div style="font-size:2rem;font-weight:800;color:white;line-height:1.2;">Halo, Talk Friend! 👋</div>
        <div style="font-size:0.9rem;color:rgba(255,255,255,0.6);margin-top:4px;">Bagaimana kabar mu hari ini?</div>
    </div>
    """, unsafe_allow_html=True)

    mode_title = "💬 Curhat" if mode == "curhat" else "🧠 Analisis MBTI"

    # ══════════════════════════════════════════════════════════════════════════
    # CURHAT MODE
    # ══════════════════════════════════════════════════════════════════════════
    if mode == 'curhat':

        # Build ALL messages into ONE html string → render in ONE st.markdown call
        # This is the key: everything inside the white card div in one shot.
        chat_html = build_chat_html(st.session_state.messages)

        st.markdown(f"""
        <div style="
            background:white;
            border-radius:20px;
            padding:20px 22px 18px;
            box-shadow:0 8px 32px rgba(0,0,0,0.12);
            margin-bottom:14px;
        ">
            <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:14px;">
                {mode_title}
            </div>
            <div style="
                min-height:260px;
                max-height:400px;
                overflow-y:auto;
                padding-right:4px;
            ">{chat_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # Input row below card
        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([9, 1])
            with c1:
                user_input = st.text_input("", placeholder="Ketik pesanmu...", label_visibility="collapsed")
            with c2:
                st.markdown('<div class="send-btn">', unsafe_allow_html=True)
                submitted = st.form_submit_button("✈️", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        if submitted and user_input and user_input.strip():
            user_text = user_input.strip()
            if user_text != st.session_state._last_input:
                st.session_state._last_input = user_text
                st.session_state.messages.append({'role':'user','content':user_text})

                emo_id, conf = predict_emotion(user_text, emo_model, emo_vec)
                st.session_state.emotion    = emo_id
                st.session_state.confidence = conf
                st.session_state.emo_counts[emo_id] = st.session_state.emo_counts.get(emo_id,0)+1

                mbti_p, mbti_c = predict_mbti_passive(user_text, mbti_model, mbti_vec, st.session_state.mbti_texts)
                if mbti_p and mbti_c > 0.3:
                    st.session_state.mbti = mbti_p

                response = get_ai_response(user_text, emo_id, st.session_state.messages, st.session_state.last_bot)
                if not response:
                    response = fallback_response(user_text, emo_id, st.session_state.messages)

                st.session_state.last_bot.append(response)
                st.session_state.messages.append({'role':'bot','content':response})
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # MBTI QUIZ MODE
    # ══════════════════════════════════════════════════════════════════════════
    elif mode == 'mbti':
        step   = st.session_state.mbti_step
        result = st.session_state.mbti_result

        if result:
            name, desc = MBTI_DESC.get(result, ("Unknown","Tipe kepribadian unik."))
            dims_html  = "".join(f"""
                <div style="background:#f0f4ff;border-left:3px solid #3b82f6;
                            border-radius:0 8px 8px 0;padding:9px 14px;
                            margin-bottom:8px;font-size:13px;color:#334155;">
                    <b style="color:#3b82f6;">{d}</b> — {DIM_EXP[d]}
                </div>""" for d in result if d in DIM_EXP)

            st.markdown(f"""
            <div style="background:white;border-radius:20px;padding:24px 26px;
                        box-shadow:0 8px 32px rgba(0,0,0,0.12);margin-bottom:14px;">
                <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:18px;">{mode_title}</div>
                <div style="text-align:center;padding:8px 0 22px;">
                    <div style="font-size:3rem;font-weight:800;
                                background:linear-gradient(135deg,#3b82f6,#2a5ba8);
                                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                                margin-bottom:6px;">{result}</div>
                    <div style="font-size:1rem;font-weight:700;color:#1e293b;margin-bottom:6px;">{name}</div>
                    <div style="font-size:13px;color:#64748b;max-width:400px;margin:0 auto;">{desc}</div>
                </div>
                <div style="max-width:460px;margin:0 auto;">{dims_html}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="action-btn-white">', unsafe_allow_html=True)
            if st.button("🔄 Ulangi Analisis", key="reset_mbti"):
                st.session_state.mbti_step   = 0
                st.session_state.mbti_resp   = []
                st.session_state.mbti_result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        elif step < len(MBTI_Q):
            q        = MBTI_Q[step]
            progress = step / len(MBTI_Q)

            # Question card
            st.markdown(f"""
            <div style="background:white;border-radius:20px;padding:22px 24px 18px;
                        box-shadow:0 8px 32px rgba(0,0,0,0.12);margin-bottom:14px;">
                <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:14px;">{mode_title}</div>
                <div style="display:flex;justify-content:space-between;
                            font-size:12px;color:#94a3b8;margin-bottom:6px;">
                    <span>Pertanyaan {step+1} dari {len(MBTI_Q)}</span>
                    <span>{int(progress*100)}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(progress)

            st.markdown(f"""
            <div style="font-size:15px;font-weight:600;color:white;
                        margin:14px 0 18px;line-height:1.65;">{q['q']}</div>
            """, unsafe_allow_html=True)

            ba, bb = st.columns(2)
            with ba:
                st.markdown('<div class="mbti-btn">', unsafe_allow_html=True)
                if st.button(f"A.  {q['A']}", key=f"mbti_a_{step}", use_container_width=True):
                    st.session_state.mbti_resp.append({"qid":q["id"],"ans":"A"})
                    st.session_state.mbti_step += 1
                    if st.session_state.mbti_step >= len(MBTI_Q):
                        st.session_state.mbti_result = analyze_mbti(st.session_state.mbti_resp)
                        st.session_state.mbti = st.session_state.mbti_result
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            with bb:
                st.markdown('<div class="mbti-btn">', unsafe_allow_html=True)
                if st.button(f"B.  {q['B']}", key=f"mbti_b_{step}", use_container_width=True):
                    st.session_state.mbti_resp.append({"qid":q["id"],"ans":"B"})
                    st.session_state.mbti_step += 1
                    if st.session_state.mbti_step >= len(MBTI_Q):
                        st.session_state.mbti_result = analyze_mbti(st.session_state.mbti_resp)
                        st.session_state.mbti = st.session_state.mbti_result
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
