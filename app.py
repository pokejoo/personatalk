"""
🐼 PersonaTalk — Merged UI
Native Streamlit sidebar + HuggingFace models + Blue gradient design
"""

import streamlit as st
import numpy as np
import re
import random
import math
import joblib
import os

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
EMO_EMOJI = {0:'😔', 1:'😊', 2:'❤️',  3:'😠', 4:'😨', 5:'😲'}
EMO_COLOR = {0:'#5b8dd9', 1:'#f59f00', 2:'#e64980', 3:'#f03e3e', 4:'#7950f2', 5:'#2ecc71'}
EMO_BG    = {0:'rgba(91,141,217,0.15)', 1:'rgba(245,159,0,0.15)',
             2:'rgba(230,73,128,0.15)',  3:'rgba(240,62,62,0.15)',
             4:'rgba(121,80,242,0.15)',  5:'rgba(46,204,113,0.15)'}

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
    'cinta':2,'sayang':2,'rindu':2,'kangen':2,'suka':2,'naksir':2,'jatuh cinta':2,
    'kaget':5,'shock':5,'terkejut':5,'astaga':5,'gak nyangka':5,
}

def rule_emotion(text: str):
    t = text.lower()
    for phrase, emo in [
        ('patah hati',0),('putus sama',0),('diselingkuhin',0),('dikhianatin',0),
        ('sakit hati',0),('ditinggal pergi',0),('abis putus',0),
        ('overthinking',4),('deg degan',4),('was-was',4),('ga tenang',4),
        ('nggak tenang',4),('khawatir banget',4),
        ('seneng banget',1),('alhamdulillah',1),('lega banget',1),('bangga banget',1),
        ('jatuh cinta',2),('pdkt',2),('naksir',2),('kangen banget',2),
        ('marah banget',3),('kesel banget',3),('nggak adil',3),('gak adil',3),
        ('kaget banget',5),('ga nyangka',5),('nggak nyangka',5),
    ]:
        if phrase in t: return emo
    for emo, kws in {
        0:['sedih','nangis','down','putus','ditinggal','sepi','galau','hampa','nyesel','bingung'],
        1:['bahagia','senang','happy','gembira','excited','lega','bangga','yay'],
        2:['cinta','sayang','rindu','kangen','naksir','gebetan'],
        3:['marah','kesal','benci','emosi','jengkel','muak'],
        4:['takut','cemas','khawatir','panik','nervous','gelisah','stress','anxious'],
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
    """Passive MBTI prediction from chat history (curhat mode)."""
    hist.append(text)
    if len(hist) < 2: return None, 0.0
    cl = preprocess(' '.join(hist[-10:]))
    if not cl: return None, 0.0
    X = mv.transform([cl])
    return mm.predict(X)[0], float(mm.predict_proba(X)[0].max())

# ── MBTI Quiz ─────────────────────────────────────────────────────────────────
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

## Konteks
- Baca SELURUH riwayat percakapan sebelum menjawab
- Kalau user sudah menjawab pertanyaanmu → jangan tanya hal yang sama lagi
- Kalau user bercerita hal POSITIF → MATCH the energy, ikut excited
- Kalau topik sudah 4+ exchange → mulai berikan insight ringan atau saran praktis

## Pantangan keras
- JANGAN mengulang pertanyaan yang sudah dijawab user
- JANGAN mengulang kalimat pembuka yang sama dari respons sebelumnya
- JANGAN diagnosis medis atau psikologis
- JANGAN lebih dari 4 kalimat"""

# ── Groq AI ───────────────────────────────────────────────────────────────────
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
                return text
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": "Gunakan pembuka berbeda dan pertanyaan penutup beda topik."})
            retry = client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=msgs,
                max_tokens=400, temperature=0.95,
            )
            text2 = clean_text(retry.choices[0].message.content)
            if text2 and len(text2) > 10:
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
def mood_donut_svg(emo_counts, size=90):
    total    = sum(emo_counts.values()) or 1
    pcts     = [emo_counts.get(i, 0) / total for i in range(6)]
    has_data = any(v > 0 for v in emo_counts.values())
    dominant = max(emo_counts, key=emo_counts.get) if has_data else 1
    dom_pct  = int(pcts[dominant] * 100)
    colors   = [EMO_COLOR[i] for i in range(6)]
    emojis_list = ['😢','😊','😍','😤','😰','😲']

    r, cx, cy, inner_r = 45, 50, 50, 28
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
        paths = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="rgba(255,255,255,0.15)"/>'

    return f"""
    <svg viewBox="0 0 100 100" width="{size}" height="{size}">
        {paths}
        <circle cx="{cx}" cy="{cy}" r="{inner_r}" fill="#1a3f7a"/>
        <text x="{cx}" y="{cy-5}" text-anchor="middle" font-size="14">{emojis_list[dominant]}</text>
        <text x="{cx}" y="{cy+10}" text-anchor="middle" fill="white" font-size="9" font-weight="bold">{dom_pct}%</text>
    </svg>"""

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #1a3a6b 0%, #2a5ba8 50%, #4a90d9 100%) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* ── Main content padding ── */
.main .block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Native sidebar ── */
section[data-testid="stSidebar"] {
    background: #f0f4ff !important;
    border-right: 1px solid #e2e8f0 !important;
    min-width: 230px !important;
    max-width: 230px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem !important;
}
section[data-testid="stSidebar"] * {
    color: #334155 !important;
}

/* ── Sidebar buttons (nav items) ── */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 40px !important;
    color: #334155 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 9px 14px !important;
    width: 100% !important;
    justify-content: flex-start !important;
    transition: all 0.18s ease !important;
    margin-bottom: 2px !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #dbeafe !important;
    color: #1d4ed8 !important;
    transform: translateX(3px) !important;
}

/* ── Active nav button ── */
.nav-active > .stButton > button {
    background: #3b82f6 !important;
    color: white !important;
    font-weight: 600 !important;
}
.nav-active > .stButton > button:hover {
    background: #2563eb !important;
    color: white !important;
    transform: none !important;
}

/* ── Main area buttons (send, MBTI choices, action) ── */
.main .stButton > button {
    background: #3b82f6 !important;
    border: none !important;
    border-radius: 30px !important;
    color: white !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.4) !important;
}
.main .stButton > button:hover {
    background: #2563eb !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
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
    padding: 12px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    line-height: 1.5 !important;
}
.mbti-btn .stButton > button:hover {
    background: #3b82f6 !important;
    color: white !important;
    border-color: #3b82f6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.35) !important;
}

/* Reset / action buttons */
.action-btn .stButton > button {
    background: white !important;
    color: #334155 !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    font-size: 13px !important;
    padding: 10px 18px !important;
}
.action-btn .stButton > button:hover {
    background: #dbeafe !important;
    color: #1d4ed8 !important;
    border-color: #bfdbfe !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Text input (chat box) ── */
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
.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;
}
.stTextInput label { display: none !important; }

/* ── Form ── */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #4a90d9) !important;
    border-radius: 10px !important;
}
.stProgress > div > div {
    background: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── Metric ── */
[data-testid="stMetricValue"] { color: #3b82f6 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 12px !important; }

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; margin: 10px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 4px; }

/* ── Radio (mode selector) ── */
section[data-testid="stSidebar"] [role="radiogroup"] {
    background: transparent !important;
    gap: 0 !important;
}
section[data-testid="stSidebar"] [role="radiogroup"] label {
    color: #334155 !important;
    font-size: 13px !important;
    padding: 6px 10px !important;
}

/* ── White card ── */
.pt-card {
    background: white;
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.10);
}
</style>
""", unsafe_allow_html=True)


# ── Chat bubble renderer ──────────────────────────────────────────────────────
def render_messages(messages):
    for msg in messages:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
                <div style="
                    background:linear-gradient(135deg,#3b82f6,#2a5ba8);
                    border-radius:18px 18px 4px 18px;
                    padding:10px 16px;
                    max-width:70%;
                    font-size:14px;
                    color:white;
                    line-height:1.6;
                    box-shadow:0 2px 10px rgba(59,130,246,0.3);
                ">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start;margin-bottom:10px;">
                <div style="
                    background:linear-gradient(135deg,#4a90d9,#2a5ba8);
                    border-radius:18px 18px 18px 4px;
                    padding:10px 16px;
                    max-width:70%;
                    font-size:14px;
                    color:white;
                    line-height:1.6;
                    box-shadow:0 2px 10px rgba(42,91,168,0.2);
                    border:1px solid rgba(255,255,255,0.15);
                ">{msg['content']}</div>
            </div>
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

    # Load models
    with st.spinner("Memuat PersonaTalk..."):
        emo_model, emo_vec, mbti_model, mbti_vec = load_models()

    # Session state defaults
    defaults = {
        'messages':      [{'role':'bot','content':'Halo, Talk Friend! 👋 Aku PersonaTalk.\nCerita apa aja, aku dengerin ya!'}],
        'emotion':       1,
        'confidence':    0.5,
        'mbti':          None,
        'mbti_texts':    [],
        'last_bot':      [],
        'emo_counts':    {i:0 for i in range(6)},
        '_ai_err':       None,
        'nav':           'dashboard',   # dashboard | riwayat | tentang | setelan
        'mode':          'curhat',      # curhat | mbti
        'mbti_step':     0,
        'mbti_resp':     [],
        'mbti_result':   None,
        '_last_input':   '',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    nav  = st.session_state.nav
    mode = st.session_state.mode
    emo  = st.session_state.emotion

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR — native st.sidebar
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        # ── Logo ──────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;padding-left:4px;">
            <span style="color:#e53e3e;font-size:24px;font-weight:900;line-height:1;">✳</span>
            <div style="line-height:1.15;">
                <div style="font-size:16px;font-weight:900;color:#1e293b;">persona</div>
                <div style="font-size:16px;font-weight:900;color:#1e293b;">talk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Mood Donut ────────────────────────────────────────────────────────
        donut_svg = mood_donut_svg(st.session_state.emo_counts, size=72)
        emo_name  = EMO_LABEL.get(emo, 'Netral')
        emo_emoji = EMO_EMOJI.get(emo, '😊')
        mood_label = "Mood bagus! 🌟" if emo==1 else "Mood oke 👍" if emo in [2,5] else "Perlu perhatian 💙"

        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#1a3a6b,#2a5ba8);
            border-radius:16px;
            padding:14px 16px;
            margin-bottom:16px;
        ">
            <div style="color:rgba(255,255,255,0.8);font-size:11px;font-weight:700;
                        letter-spacing:0.06em;margin-bottom:10px;">🎯 DETEKTOR MOOD</div>
            <div style="display:flex;align-items:center;gap:12px;">
                <div>{donut_svg}</div>
                <div>
                    <div style="color:white;font-size:15px;font-weight:700;">
                        {emo_emoji} {emo_name}
                    </div>
                    <div style="color:rgba(255,255,255,0.7);font-size:11px;margin-top:2px;">
                        {mood_label}
                    </div>
                    <div style="margin-top:8px;">
                        <div style="background:rgba(255,255,255,0.2);border-radius:6px;
                                    height:4px;width:100%;overflow:hidden;">
                            <div style="background:#f59f00;border-radius:6px;height:4px;
                                        width:{int(st.session_state.confidence*100)}%;
                                        transition:width 0.4s;"></div>
                        </div>
                        <div style="color:rgba(255,255,255,0.6);font-size:10px;margin-top:3px;">
                            Confidence {st.session_state.confidence:.0%}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── MBTI badge ────────────────────────────────────────────────────────
        mbti_val = st.session_state.mbti
        if mbti_val:
            mbti_name = MBTI_DESC.get(mbti_val, ("?",""))[0]
            st.markdown(f"""
            <div style="
                background:linear-gradient(135deg,#3b82f6,#2563eb);
                border-radius:12px;
                padding:10px 14px;
                margin-bottom:16px;
                text-align:center;
            ">
                <div style="color:rgba(255,255,255,0.8);font-size:10px;font-weight:700;
                            letter-spacing:0.06em;">🧠 MBTI KAMU</div>
                <div style="color:white;font-size:22px;font-weight:800;margin-top:2px;">{mbti_val}</div>
                <div style="color:rgba(255,255,255,0.8);font-size:11px;">{mbti_name}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background:rgba(59,130,246,0.1);
                border:1.5px dashed rgba(59,130,246,0.3);
                border-radius:12px;
                padding:10px 14px;
                margin-bottom:16px;
                text-align:center;
                color:#64748b;
                font-size:12px;
            ">🧠 MBTI belum terdeteksi<br><span style="font-size:11px;">Mulai curhat atau ikuti quiz!</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="font-size:11px;color:#94a3b8;font-weight:700;letter-spacing:0.06em;margin-bottom:6px;padding-left:4px;">MENU</div>', unsafe_allow_html=True)

        # ── Nav buttons ───────────────────────────────────────────────────────
        for key, icon, label in [
            ('dashboard','🏠','Dashboard'),
            ('riwayat',  '🕐','Riwayat Chat'),
            ('tentang',  'ℹ️', 'Tentang'),
        ]:
            is_active = (nav == key)
            wrap = 'nav-active' if is_active else ''
            st.markdown(f'<div class="{wrap}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.nav = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#94a3b8;font-weight:700;letter-spacing:0.06em;margin-bottom:6px;padding-left:4px;">PANEL ANALISIS</div>', unsafe_allow_html=True)

        for key, icon, label in [
            ('curhat','💬','Curhat'),
            ('mbti',  '🧠','Analisis MBTI'),
        ]:
            is_active = (mode == key and nav == 'dashboard')
            wrap = 'nav-active' if is_active else ''
            st.markdown(f'<div class="{wrap}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"mode_{key}", use_container_width=True):
                st.session_state.mode = key
                st.session_state.nav  = 'dashboard'
                if key == 'mbti':
                    st.session_state.mbti_step   = 0
                    st.session_state.mbti_resp   = []
                    st.session_state.mbti_result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Spacer + divider + settings ───────────────────────────────────────
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)

        is_active = (nav == 'setelan')
        wrap = 'nav-active' if is_active else ''
        st.markdown(f'<div class="{wrap}">', unsafe_allow_html=True)
        if st.button("⚙️  Setelan", key="nav_setelan", use_container_width=True):
            st.session_state.nav = 'setelan'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN AREA
    # ══════════════════════════════════════════════════════════════════════════

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:6px 0 18px;">
        <div style="font-size:2rem;font-weight:800;color:white;line-height:1.2;">
            Halo, Talk Friend! 👋
        </div>
        <div style="font-size:0.95rem;color:rgba(255,255,255,0.7);margin-top:4px;">
            Bagaimana kabar mu hari ini?
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    if nav == 'dashboard':

        # ── Mode title ────────────────────────────────────────────────────────
        mode_title = "💬 Curhat" if mode == "curhat" else "🧠 Analisis MBTI"

        # ══════════════════════════════════════════════════════════════════════
        # CURHAT MODE
        # ══════════════════════════════════════════════════════════════════════
        if mode == 'curhat':
            st.markdown(f"""
            <div class="pt-card" style="margin-bottom:14px;">
                <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:14px;">
                    {mode_title}
                </div>
                <div style="
                    min-height:300px;
                    max-height:420px;
                    overflow-y:auto;
                    padding-right:4px;
                " id="chat-area">
            """, unsafe_allow_html=True)

            render_messages(st.session_state.messages)

            st.markdown("</div></div>", unsafe_allow_html=True)

            # ── Input form ────────────────────────────────────────────────────
            with st.form("chat_form", clear_on_submit=True):
                c1, c2 = st.columns([8, 1])
                with c1:
                    user_input = st.text_input(
                        "", placeholder="Ketik pesanmu...",
                        label_visibility="collapsed"
                    )
                with c2:
                    submitted = st.form_submit_button("✈️", use_container_width=True)

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

        # ══════════════════════════════════════════════════════════════════════
        # MBTI QUIZ MODE
        # ══════════════════════════════════════════════════════════════════════
        elif mode == 'mbti':
            step   = st.session_state.mbti_step
            result = st.session_state.mbti_result

            if result:
                # ── Result card ───────────────────────────────────────────────
                name, desc = MBTI_DESC.get(result, ("Unknown", "Tipe kepribadian unik."))
                st.markdown(f"""
                <div class="pt-card">
                    <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:20px;">
                        {mode_title}
                    </div>
                    <div style="text-align:center;padding:10px 0 20px;">
                        <div style="font-size:3.5rem;font-weight:800;
                                    background:linear-gradient(135deg,#3b82f6,#2a5ba8);
                                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                                    margin-bottom:6px;">{result}</div>
                        <div style="font-size:1.05rem;font-weight:700;color:#1e293b;margin-bottom:6px;">{name}</div>
                        <div style="font-size:13px;color:#64748b;max-width:420px;margin:0 auto;">{desc}</div>
                    </div>
                    <div style="max-width:480px;margin:0 auto;">
                """, unsafe_allow_html=True)

                for d in result:
                    if d in DIM_EXP:
                        st.markdown(f"""
                        <div style="
                            background:#f0f4ff;
                            border-left:3px solid #3b82f6;
                            border-radius:8px;
                            padding:9px 14px;
                            margin-bottom:8px;
                            font-size:13px;
                            color:#334155;
                        "><b style="color:#3b82f6;">{d}</b> — {DIM_EXP[d]}</div>
                        """, unsafe_allow_html=True)

                st.markdown("</div></div>", unsafe_allow_html=True)

                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                st.markdown('<div class="action-btn">', unsafe_allow_html=True)
                if st.button("🔄 Ulangi Analisis", key="reset_mbti"):
                    st.session_state.mbti_step   = 0
                    st.session_state.mbti_resp   = []
                    st.session_state.mbti_result = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            elif step < len(MBTI_Q):
                # ── Question card ─────────────────────────────────────────────
                q        = MBTI_Q[step]
                progress = step / len(MBTI_Q)

                st.markdown(f"""
                <div class="pt-card">
                    <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:16px;">
                        {mode_title}
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                font-size:12px;color:#94a3b8;margin-bottom:6px;">
                        <span>Pertanyaan {step+1} dari {len(MBTI_Q)}</span>
                        <span>{int(progress*100)}%</span>
                    </div>
                """, unsafe_allow_html=True)

                st.progress(progress)

                st.markdown(f"""
                    <div style="font-size:15px;font-weight:600;color:#1e293b;
                                margin:16px 0 20px;line-height:1.6;">
                        {q['q']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

                ba, bb = st.columns(2)
                with ba:
                    st.markdown('<div class="mbti-btn">', unsafe_allow_html=True)
                    if st.button(f"A.  {q['A']}", key=f"mbti_a_{step}", use_container_width=True):
                        st.session_state.mbti_resp.append({"qid": q["id"], "ans": "A"})
                        st.session_state.mbti_step += 1
                        if st.session_state.mbti_step >= len(MBTI_Q):
                            st.session_state.mbti_result = analyze_mbti(st.session_state.mbti_resp)
                            st.session_state.mbti = st.session_state.mbti_result
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                with bb:
                    st.markdown('<div class="mbti-btn">', unsafe_allow_html=True)
                    if st.button(f"B.  {q['B']}", key=f"mbti_b_{step}", use_container_width=True):
                        st.session_state.mbti_resp.append({"qid": q["id"], "ans": "B"})
                        st.session_state.mbti_step += 1
                        if st.session_state.mbti_step >= len(MBTI_Q):
                            st.session_state.mbti_result = analyze_mbti(st.session_state.mbti_resp)
                            st.session_state.mbti = st.session_state.mbti_result
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RIWAYAT
    # ══════════════════════════════════════════════════════════════════════════
    elif nav == 'riwayat':
        st.markdown('<div class="pt-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:16px;font-weight:700;color:#1e293b;margin-bottom:16px;">🕐 Riwayat Chat</div>', unsafe_allow_html=True)

        user_msgs = [m for m in st.session_state.messages if m['role']=='user']
        if user_msgs:
            for i, m in enumerate(user_msgs[-20:], 1):
                preview = m['content'][:90] + ('...' if len(m['content'])>90 else '')
                st.markdown(f"""
                <div style="
                    padding:10px 14px;
                    background:#f8fafc;
                    border-radius:10px;
                    margin-bottom:8px;
                    font-size:13px;
                    color:#334155;
                    border-left:3px solid #3b82f6;
                    display:flex;
                    align-items:flex-start;
                    gap:10px;
                ">
                    <span style="color:#94a3b8;font-size:11px;min-width:20px;">#{i}</span>
                    <span>{preview}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#94a3b8;font-size:14px;text-align:center;padding:30px 0;">Belum ada riwayat chat.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TENTANG
    # ══════════════════════════════════════════════════════════════════════════
    elif nav == 'tentang':
        st.markdown("""
        <div class="pt-card">
            <div style="font-size:16px;font-weight:700;color:#1e293b;margin-bottom:14px;">ℹ️ Tentang PersonaTalk</div>
            <div style="font-size:14px;color:#475569;line-height:1.85;">
                <b style="color:#1e293b;">PersonaTalk</b> adalah teman curhat AI yang dirancang
                untuk mendengarkan tanpa menghakimi. Aplikasi ini menggunakan model ML untuk
                mendeteksi emosi dan kepribadian MBTI dari percakapan, serta AI Groq
                (LLaMA 3.3 70B) untuk menghasilkan respons yang hangat dan empatik.
                <br><br>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px;">
                    <div style="background:#f0f4ff;border-radius:10px;padding:12px 14px;">
                        <div style="font-size:18px;margin-bottom:4px;">🎯</div>
                        <div style="font-weight:600;color:#1e293b;font-size:13px;">Deteksi Emosi</div>
                        <div style="color:#64748b;font-size:12px;margin-top:2px;">Real-time dari pesanmu</div>
                    </div>
                    <div style="background:#f0f4ff;border-radius:10px;padding:12px 14px;">
                        <div style="font-size:18px;margin-bottom:4px;">🧠</div>
                        <div style="font-weight:600;color:#1e293b;font-size:13px;">Analisis MBTI</div>
                        <div style="color:#64748b;font-size:12px;margin-top:2px;">Quiz 10 pertanyaan</div>
                    </div>
                    <div style="background:#f0f4ff;border-radius:10px;padding:12px 14px;">
                        <div style="font-size:18px;margin-bottom:4px;">💬</div>
                        <div style="font-weight:600;color:#1e293b;font-size:13px;">Curhat AI</div>
                        <div style="color:#64748b;font-size:12px;margin-top:2px;">Paham konteks percakapan</div>
                    </div>
                    <div style="background:#f0f4ff;border-radius:10px;padding:12px 14px;">
                        <div style="font-size:18px;margin-bottom:4px;">📊</div>
                        <div style="font-weight:600;color:#1e293b;font-size:13px;">Mood Tracker</div>
                        <div style="color:#64748b;font-size:12px;margin-top:2px;">Pantau emosi harianmu</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SETELAN
    # ══════════════════════════════════════════════════════════════════════════
    elif nav == 'setelan':
        st.markdown('<div class="pt-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:16px;font-weight:700;color:#1e293b;margin-bottom:18px;">⚙️ Setelan</div>', unsafe_allow_html=True)

        provider = "Groq (LLaMA 3.3 70B)" if (GROQ_KEY and _GROQ_OK) else "Belum aktif"
        err      = st.session_state.get('_ai_err')
        mbti_r   = st.session_state.get('mbti') or 'Belum dianalisis'
        total_msg = len([m for m in st.session_state.messages if m['role']=='user'])

        rows = [
            ("🤖 AI Provider",   provider),
            ("🧠 MBTI Kamu",      mbti_r),
            ("💬 Total Pesan",    str(total_msg)),
            ("😊 Emosi Sekarang", f"{EMO_EMOJI.get(emo,'')} {EMO_LABEL.get(emo,'Netral')}"),
        ]
        for label, val in rows:
            st.markdown(f"""
            <div style="
                display:flex;justify-content:space-between;align-items:center;
                padding:11px 0;border-bottom:1px solid #f1f5f9;
                font-size:13px;color:#475569;
            ">
                <span>{label}</span>
                <span style="font-weight:600;color:#1e293b;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

        if err:
            st.markdown(f'<div style="padding:10px 0;font-size:12px;color:#ef4444;"><b>Error:</b> {err}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("🗑️ Reset Chat", key="reset_chat", use_container_width=True):
                st.session_state.messages   = [{'role':'bot','content':'Halo, Talk Friend! 👋 Cerita apa aja, aku dengerin ya!'}]
                st.session_state.emo_counts = {i:0 for i in range(6)}
                st.session_state.last_bot   = []
                st.session_state.mbti_texts = []
                st.session_state._ai_err    = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("🔄 Reset MBTI", key="reset_mbti_set", use_container_width=True):
                st.session_state.mbti        = None
                st.session_state.mbti_step   = 0
                st.session_state.mbti_resp   = []
                st.session_state.mbti_result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
