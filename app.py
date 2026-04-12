"""
PersonaTalk — UI custom (screenshot) + HuggingFace models + Groq AI
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

from huggingface_hub import hf_hub_download

try:
    from groq import Groq
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False

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
    'sedih':0,'kecewa':0,'galau':0,'nangis':0,'sepi':0,'putus':0,'ditinggal':0,
    'kehilangan':0,'patah hati':0,'down':0,'murung':0,'hampa':0,'nyesel':0,
    'frustasi':0,'hopeless':0,'bingung':0,'sendirian':0,'sakit hati':0,
    'dikhianati':0,'selingkuh':0,
    'marah':3,'kesal':3,'benci':3,'jengkel':3,'emosi':3,'kesel':3,'sebel':3,
    'muak':3,'dibohongi':3,'ditipu':3,'nggak adil':3,'gak adil':3,
    'senang':1,'bahagia':1,'happy':1,'excited':1,'lega':1,'bangga':1,'semangat':1,
    'alhamdulillah':1,'syukur':1,'gembira':1,'berhasil':1,'sukses':1,
    'takut':4,'cemas':4,'khawatir':4,'panik':4,'nervous':4,'gelisah':4,'stress':4,
    'anxious':4,'overthinking':4,'was-was':4,'resah':4,
    'cinta':2,'sayang':2,'rindu':2,'kangen':2,'suka':2,'naksir':2,
    'kaget':5,'shock':5,'terkejut':5,'astaga':5,'ga nyangka':5,
}

def rule_emotion(text: str):
    t = text.lower()
    for phrase, emo in [
        ('patah hati',0),('putus sama',0),('diselingkuhin',0),('dikhianatin',0),
        ('sakit hati',0),('capek banget',0),('lelah banget',0),('bingung banget',0),
        ('overthinking',4),('deg degan',4),('was-was',4),('ga tenang',4),
        ('nggak tenang',4),('khawatir banget',4),
        ('seneng banget',1),('alhamdulillah',1),('lega banget',1),
        ('jatuh cinta',2),('pdkt',2),('naksir',2),
        ('marah banget',3),('kesel banget',3),('nggak adil',3),
        ('kaget banget',5),('ga nyangka',5),
    ]:
        if phrase in t: return emo
    for emo, kws in {
        0:['sedih','nangis','down','putus','ditinggal','sepi','galau','hampa'],
        1:['bahagia','senang','happy','gembira','excited','lega','bangga'],
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
    t  = text.lower()
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
    X  = mv.transform([cl])
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
SYSTEM_PROMPT = """Kamu adalah PersonaTalk — sahabat curhat digital yang beneran peduli, bukan robot atau konselor formal.

Kamu seperti teman yang udah kenal lama, nyambung, hangat, dan nggak pernah ngehakimin.

CARA NGOBROL:
- Bahasa Indonesia sehari-hari, santai, boleh mix english
- Mulai SELALU dengan validasi perasaan dulu
- Empati harus berasa nyata, bukan template
- Gunakan ekspresi natural: "duh", "hmm", "ya Allah", "aduh", "serius?", "waduh"
- 2-4 kalimat aja
- Di akhir selalu tanya SATU pertanyaan terbuka yang relevan
- Jangan pakai bullet point atau format kaku
- Jangan bilang "sebagai AI"

PANTANGAN:
- Jangan diagnosis medis/psikologis
- Jangan jawaban template generik
- Jangan lebih dari 4 kalimat"""

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
            model="llama-3.3-70b-versatile", messages=msgs,
            max_tokens=400, temperature=0.85)
        text = resp.choices[0].message.content.strip()
        text = re.sub(r'\[Emosi user.*?\]','',text,flags=re.DOTALL).strip()
        return text if text and len(text)>10 else None
    except Exception as e:
        st.session_state['_ai_err'] = str(e)[:150]
    return None

def fallback_response(text, emotion):
    t = text.lower()
    for kws, opts in {
        ('putus','ditinggal','selingkuh'):
            ["Aduh, itu pasti nyakitin banget... wajar banget kalau sekarang rasanya hancur. Udah cerita ke siapa belum?",
             "Duh, itu pasti bikin kamu ngerasa down banget. Mau cerita lebih?"],
        ('capek','lelah','burnout'):
            ["Hmm, capek yang kayak gini tuh beda sama capek biasa. Ini dari mana — fisik, pikiran, atau keduanya?"],
        ('marah','kesal','benci','kesel'):
            ["Iya, situasi kayak gitu emang bikin darah naik. Boleh cerita lebih — ini karena apa?"],
        ('takut','cemas','khawatir','anxious'):
            ["Hmm, ngerasa cemas itu nggak enak banget ya. Ini soal apa yang bikin kamu khawatir?"],
        ('naksir','gebetan','crush'):
            ["Ooh ada yang spesial nih! Dia udah tau kamu naksir?"],
    }.items():
        if any(k in t for k in kws): return random.choice(opts)
    return random.choice({
        0:["Duh, kedengarannya berat banget... cerita lebih dong, aku dengerin."],
        1:["Wah, ada kabar baik nih kayaknya! Cerita dong!"],
        2:["Ada yang spesial nih 😊 Cerita yuk!"],
        3:["Kayaknya ada yang bikin kamu kesel. Boleh cerita — ini soal apa?"],
        4:["Kayaknya ada yang lagi bikin kamu khawatir. Cerita yuk, biar agak lega."],
        5:["Wah, ada kejutan nih! Cerita dong lebih lengkapnya!"],
    }.get(emotion, ["Hmm, cerita lebih yuk. Aku dengerin kok."]))

# ── Mood Donut SVG ─────────────────────────────────────────────────────────────
def mood_donut_svg(emo_counts, size=110):
    total    = sum(emo_counts.values()) or 1
    pcts     = [emo_counts.get(i,0)/total for i in range(6)]
    has_data = any(v>0 for v in emo_counts.values())
    dominant = max(emo_counts, key=emo_counts.get) if has_data else 1
    dom_pct  = int(pcts[dominant]*100)
    colors   = [EMO_COLOR[i] for i in range(6)]
    emojis   = ['😢','😊','😍','😤','😰','😲']
    r,cx,cy,ir = 45,50,50,28
    paths, off = "", 0.0
    for i, pct in enumerate(pcts):
        if pct < 0.005: off+=pct; continue
        a1,a2 = off*360-90,(off+pct)*360-90
        large = 1 if pct>0.5 else 0
        def pt(a,_r=r,_cx=cx,_cy=cy):
            rad=math.radians(a); return _cx+_r*math.cos(rad),_cy+_r*math.sin(rad)
        x1,y1=pt(a1); x2,y2=pt(a2)
        paths+=f'<path d="M{cx},{cy}L{x1:.1f},{y1:.1f}A{r},{r},0,{large},1,{x2:.1f},{y2:.1f}Z" fill="{colors[i]}" opacity="0.9"/>'
        off+=pct
    if not has_data:
        paths=f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="rgba(255,255,255,0.2)"/>'
    return f"""<svg viewBox="0 0 100 100" width="{size}" height="{size}">
        {paths}
        <circle cx="{cx}" cy="{cy}" r="{ir}" fill="#1a3f7a"/>
        <text x="{cx}" y="{cy-5}" text-anchor="middle" font-size="14">{emojis[dominant]}</text>
        <text x="{cx}" y="{cy+10}" text-anchor="middle" fill="white" font-size="9" font-weight="bold">{dom_pct}%</text>
    </svg>"""

# ── Build chat HTML (ONE string → renders inside white card) ───────────────────
def build_chat_html(messages):
    if not messages:
        return '<div style="color:#94a3b8;font-size:14px;text-align:center;padding:40px 0;">Mulai percakapan...</div>'
    html = ""
    for msg in messages:
        content = (msg['content']
                   .replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
                   .replace('\n','<br>').replace('**',''))
        if msg['role'] == 'user':
            html += f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:12px;">
                <div style="
                    background:linear-gradient(135deg,#a5b4fc,#818cf8);
                    border-radius:20px 20px 4px 20px;
                    padding:12px 18px;max-width:65%;
                    font-size:14px;color:white;line-height:1.6;
                    box-shadow:0 2px 12px rgba(129,140,248,0.3);
                    word-wrap:break-word;
                ">{content}</div>
            </div>"""
        else:
            html += f"""
            <div style="display:flex;justify-content:flex-start;margin-bottom:12px;">
                <div style="
                    background:rgba(241,245,249,0.95);
                    border-radius:20px 20px 20px 4px;
                    padding:12px 18px;max-width:65%;
                    font-size:14px;color:#1e293b;line-height:1.6;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);
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
.stApp {
    background: linear-gradient(135deg,#1a3a6b 0%,#2a5ba8 50%,#4a90d9 100%) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

/* Main block */
.main .block-container { padding:1.5rem 2rem 2rem !important; max-width:100% !important; }

/* ── SIDEBAR (white) ── */
section[data-testid="stSidebar"] {
    background: white !important;
    border-right: 1px solid #e2e8f0 !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.2rem 0.8rem !important;
}

/* Override teks sidebar jadi gelap */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {
    color: #334155 !important;
}

/* Sidebar toggle arrow */
[data-testid="collapsedControl"] {
    background: white !important;
    border-radius: 0 8px 8px 0 !important;
    color: #334155 !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.1) !important;
}

/* ── Sidebar nav buttons ── */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 30px !important;
    color: #64748b !important;
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
    height: auto !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #f1f5f9 !important;
    color: #1e293b !important;
}

/* Active nav */
.nav-active > .stButton > button {
    background: #3b82f6 !important;
    color: white !important;
    font-weight: 600 !important;
}
.nav-active > .stButton > button:hover {
    background: #2563eb !important;
    color: white !important;
}

/* ── Main send button ── */
.send-btn .stButton > button {
    background: #3b82f6 !important;
    border: none !important;
    border-radius: 50% !important;
    color: white !important;
    font-size: 16px !important;
    width: 42px !important;
    height: 42px !important;
    min-height: unset !important;
    padding: 0 !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.4) !important;
    transition: all 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.send-btn .stButton > button:hover {
    background: #2563eb !important;
    transform: scale(1.08) !important;
}

/* MBTI buttons */
.mbti-btn .stButton > button {
    background: #f8fafc !important;
    color: #1e293b !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 54px !important;
    padding: 12px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    transition: all 0.18s !important;
    box-shadow: none !important;
}
.mbti-btn .stButton > button:hover {
    background: #3b82f6 !important;
    color: white !important;
    border-color: #3b82f6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.3) !important;
}

/* Action buttons */
.action-btn .stButton > button {
    background: white !important;
    color: #334155 !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-size: 13px !important;
    padding: 10px 18px !important;
    height: auto !important;
    box-shadow: none !important;
    transition: all 0.18s !important;
}
.action-btn .stButton > button:hover {
    background: #dbeafe !important;
    color: #1d4ed8 !important;
    border-color: #bfdbfe !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: white !important;
    border: none !important;
    border-radius: 28px !important;
    color: #1e293b !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 14px !important;
    padding: 11px 18px !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #94a3b8 !important; }
.stTextInput > div { background: transparent !important; border: none !important; box-shadow: none !important; }
.stTextInput label { display: none !important; }

/* Form */
[data-testid="stForm"] { background:transparent !important; border:none !important; padding:0 !important; }

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#3b82f6,#60a5fa) !important;
    border-radius: 10px !important;
}
.stProgress > div > div { background:#e2e8f0 !important; border-radius:10px !important; }

/* HR */
hr { border-color: #e2e8f0 !important; margin: 10px 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 4px; }
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
        'nav':         'dashboard',
        'mode':        'curhat',
        'mbti_step':   0, 'mbti_resp': [], 'mbti_result': None,
        '_last_input': '',
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    nav  = st.session_state.nav
    mode = st.session_state.mode
    emo  = st.session_state.emotion

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:

        # Logo + hamburger row
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    margin-bottom:18px;padding:0 4px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="color:#ef4444;font-size:20px;font-weight:900;">✳</span>
                <div style="line-height:1.15;">
                    <div style="font-size:14px;font-weight:800;color:#1e293b;">persona</div>
                    <div style="font-size:14px;font-weight:800;color:#1e293b;">talk</div>
                </div>
            </div>
            <span style="font-size:18px;color:#64748b;cursor:pointer;">≡</span>
        </div>
        """, unsafe_allow_html=True)

        # Search bar (decorative)
        st.markdown("""
        <div style="background:#f1f5f9;border-radius:20px;padding:8px 14px;
                    display:flex;align-items:center;gap:8px;margin-bottom:18px;">
            <span style="color:#94a3b8;font-size:13px;">🔍</span>
            <span style="color:#94a3b8;font-size:13px;">Cari...</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Nav ───────────────────────────────────────────────────────────────
        st.markdown('<div style="font-size:11px;color:#94a3b8;font-weight:700;letter-spacing:0.08em;margin-bottom:6px;padding-left:4px;">MENU</div>', unsafe_allow_html=True)

        for key, icon, label in [
            ('dashboard','🏠','Dashboard'),
            ('riwayat',  '🕐','Riwayat Chat'),
            ('tentang',  'ℹ️', 'Tentang'),
        ]:
            active = (nav == key)
            st.markdown('<div class="nav-active">' if active else '<div>', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                st.session_state.nav = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;color:#94a3b8;font-weight:700;letter-spacing:0.08em;margin-bottom:6px;padding-left:4px;">PANEL ANALISIS</div>', unsafe_allow_html=True)

        for key, icon, label in [
            ('curhat','💬','Curhat'),
            ('mbti',  '🧠','Analisis MBTI'),
        ]:
            active = (mode == key and nav == 'dashboard')
            st.markdown('<div class="nav-active">' if active else '<div>', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"mode_{key}", use_container_width=True):
                st.session_state.mode = key
                st.session_state.nav  = 'dashboard'
                if key == 'mbti':
                    st.session_state.mbti_step   = 0
                    st.session_state.mbti_resp   = []
                    st.session_state.mbti_result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Spacer + divider + setelan
        st.markdown('<div style="flex:1;min-height:20px;"></div>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)

        active = (nav == 'setelan')
        st.markdown('<div class="nav-active">' if active else '<div>', unsafe_allow_html=True)
        if st.button("⚙️  Setelan", key="nav_setelan", use_container_width=True):
            st.session_state.nav = 'setelan'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN AREA
    # ══════════════════════════════════════════════════════════════════════════

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown("""
        <div style="padding:6px 0 18px;">
            <div style="font-size:2.2rem;font-weight:800;color:white;line-height:1.2;">
                Halo, Talk Friend! 👋
            </div>
            <div style="font-size:0.95rem;color:rgba(255,255,255,0.7);margin-top:4px;">
                Bagaimana kabar mu hari ini?
            </div>
        </div>
        """, unsafe_allow_html=True)

    with h2:
        # Mood donut panel — kanan atas
        total     = sum(st.session_state.emo_counts.values()) or 1
        pcts      = [st.session_state.emo_counts.get(i,0)/total for i in range(6)]
        has_data  = any(v>0 for v in st.session_state.emo_counts.values())
        dominant  = max(st.session_state.emo_counts, key=st.session_state.emo_counts.get) if has_data else 1
        mood_label = "Mood bagus! 🌟" if dominant==1 else "Mood oke 👍" if dominant in [2,5] else "Perlu perhatian 💙"
        donut_svg  = mood_donut_svg(st.session_state.emo_counts, size=110)

        # Top 3 mood rows
        top3 = sorted([(i,pcts[i]) for i in range(6) if pcts[i]>0.01], key=lambda x:-x[1])[:3]
        mood_rows = ""
        emojis_list = ['😢','😊','😍','😤','😰','😲']
        for i,p in top3:
            mood_rows += f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                <span style="font-size:13px;">{emojis_list[i]}</span>
                <span style="color:white;font-size:13px;flex:1;">{EMO_LABEL[i]}</span>
                <span style="color:rgba(255,255,255,0.85);font-size:13px;font-weight:600;">{int(p*100)}%</span>
            </div>"""
        if not mood_rows:
            mood_rows = '<span style="color:rgba(255,255,255,0.5);font-size:12px;">Mulai chat...</span>'

        st.markdown(f"""
        <div style="
            background:rgba(255,255,255,0.12);
            border-radius:18px;padding:14px 18px;
            backdrop-filter:blur(10px);
            border:1px solid rgba(255,255,255,0.2);
            margin-bottom:14px;
        ">
            <div style="color:rgba(255,255,255,0.8);font-size:11px;font-weight:700;
                        letter-spacing:0.06em;margin-bottom:10px;">🎯 DETEKTOR MOOD HARI INI</div>
            <div style="display:flex;align-items:center;gap:14px;">
                <div>{donut_svg}</div>
                <div style="flex:1;">
                    {mood_rows}
                    <div style="color:rgba(255,255,255,0.9);font-size:12px;
                                font-weight:600;margin-top:8px;">{mood_label}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    if nav == 'dashboard':
        mode_title = "💬 Curhat" if mode=="curhat" else "🧠 Analisis MBTI"

        # ── CURHAT ────────────────────────────────────────────────────────────
        if mode == 'curhat':
            # White card dengan chat di dalamnya (ONE html string)
            chat_html = build_chat_html(st.session_state.messages)
            st.markdown(f"""
            <div style="
                background:white;border-radius:20px;
                padding:20px 22px 16px;
                box-shadow:0 8px 32px rgba(0,0,0,0.12);
                margin-bottom:12px;
            ">
                <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:14px;">
                    {mode_title}
                </div>
                <div style="
                    min-height:240px;max-height:380px;
                    overflow-y:auto;padding-right:4px;
                ">{chat_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # Input bar
            st.markdown("""
            <div style="
                background:white;border-radius:28px;
                padding:6px 8px 6px 16px;
                box-shadow:0 4px 20px rgba(0,0,0,0.10);
                display:flex;align-items:center;gap:8px;
            ">
            """, unsafe_allow_html=True)

            with st.form("chat_form", clear_on_submit=True):
                ic1, ic2, ic3 = st.columns([0.5, 9, 0.8])
                with ic1:
                    st.markdown("""
                    <div style="width:34px;height:34px;background:#f1f5f9;border-radius:50%;
                                display:flex;align-items:center;justify-content:center;
                                color:#64748b;font-size:16px;cursor:pointer;margin-top:4px;">+</div>
                    """, unsafe_allow_html=True)
                with ic2:
                    user_input = st.text_input(
                        "", placeholder="Ketik pesanmu...",
                        label_visibility="collapsed"
                    )
                with ic3:
                    st.markdown('<div class="send-btn">', unsafe_allow_html=True)
                    submitted = st.form_submit_button("➤", use_container_width=False)
                    st.markdown('</div>', unsafe_allow_html=True)

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

                    mbti_p, mbti_c = predict_mbti_passive(
                        user_text, mbti_model, mbti_vec, st.session_state.mbti_texts)
                    if mbti_p and mbti_c > 0.3:
                        st.session_state.mbti = mbti_p

                    response = get_ai_response(
                        user_text, emo_id,
                        st.session_state.messages, st.session_state.last_bot)
                    if not response:
                        response = fallback_response(user_text, emo_id)

                    st.session_state.last_bot.append(response)
                    if len(st.session_state.last_bot) > 10:
                        st.session_state.last_bot.pop(0)
                    st.session_state.messages.append({'role':'bot','content':response})
                    st.rerun()

        # ── MBTI QUIZ ─────────────────────────────────────────────────────────
        elif mode == 'mbti':
            step   = st.session_state.mbti_step
            result = st.session_state.mbti_result

            if result:
                name, desc = MBTI_DESC.get(result, ("Unknown","Tipe kepribadian unik."))
                dims_html  = "".join(f"""
                    <div style="background:#f0f4ff;border-left:3px solid #3b82f6;
                                border-radius:0 8px 8px 0;padding:9px 14px;
                                margin-bottom:8px;font-size:13px;color:#334155;">
                        <b style="color:#3b82f6;">{d}</b> — {DIM_EXP.get(d,'')}
                    </div>""" for d in result if d in DIM_EXP)

                st.markdown(f"""
                <div style="background:white;border-radius:20px;padding:26px;
                            box-shadow:0 8px 32px rgba(0,0,0,0.12);margin-bottom:14px;">
                    <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:18px;">{mode_title}</div>
                    <div style="text-align:center;padding:8px 0 22px;">
                        <div style="font-size:3rem;font-weight:800;
                                    background:linear-gradient(135deg,#3b82f6,#2a5ba8);
                                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                                    margin-bottom:6px;">{result}</div>
                        <div style="font-size:1rem;font-weight:700;color:#1e293b;margin-bottom:6px;">{name}</div>
                        <div style="font-size:13px;color:#64748b;max-width:420px;margin:0 auto;">{desc}</div>
                    </div>
                    <div style="max-width:480px;margin:0 auto;">{dims_html}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="action-btn">', unsafe_allow_html=True)
                if st.button("🔄 Ulangi Analisis", key="reset_mbti"):
                    st.session_state.mbti_step   = 0
                    st.session_state.mbti_resp   = []
                    st.session_state.mbti_result = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            elif step < len(MBTI_Q):
                q        = MBTI_Q[step]
                progress = step / len(MBTI_Q)

                st.markdown(f"""
                <div style="background:white;border-radius:20px;padding:22px 24px 18px;
                            box-shadow:0 8px 32px rgba(0,0,0,0.12);margin-bottom:14px;">
                    <div style="font-size:14px;font-weight:700;color:#1e293b;margin-bottom:12px;">{mode_title}</div>
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

    # ══════════════════════════════════════════════════════════════════════════
    # RIWAYAT
    # ══════════════════════════════════════════════════════════════════════════
    elif nav == 'riwayat':
        st.markdown('<div style="background:white;border-radius:20px;padding:24px;box-shadow:0 8px 32px rgba(0,0,0,0.12);">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:16px;font-weight:700;color:#1e293b;margin-bottom:16px;">🕐 Riwayat Chat</div>', unsafe_allow_html=True)
        user_msgs = [m for m in st.session_state.messages if m['role']=='user']
        if user_msgs:
            for i, m in enumerate(user_msgs[-20:], 1):
                preview = m['content'][:90]+('...' if len(m['content'])>90 else '')
                st.markdown(f"""
                <div style="padding:10px 14px;background:#f8fafc;border-radius:10px;
                            margin-bottom:8px;font-size:13px;color:#334155;
                            border-left:3px solid #3b82f6;">
                    <span style="color:#94a3b8;font-size:11px;">#{i}</span>
                    <span style="margin-left:8px;">{preview}</span>
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
        <div style="background:white;border-radius:20px;padding:26px;box-shadow:0 8px 32px rgba(0,0,0,0.12);">
            <div style="font-size:16px;font-weight:700;color:#1e293b;margin-bottom:14px;">ℹ️ Tentang PersonaTalk</div>
            <div style="font-size:14px;color:#475569;line-height:1.85;">
                <b style="color:#1e293b;">PersonaTalk</b> adalah teman curhat AI yang dirancang untuk
                mendengarkan tanpa menghakimi. Menggunakan model ML untuk mendeteksi emosi dan
                kepribadian MBTI, serta Groq AI (LLaMA 3.3 70B) untuk respons yang hangat dan empatik.
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
        provider  = "Groq (LLaMA 3.3 70B)" if (GROQ_KEY and _GROQ_OK) else "Belum aktif"
        err       = st.session_state.get('_ai_err')
        mbti_r    = st.session_state.get('mbti') or 'Belum dianalisis'
        total_msg = len([m for m in st.session_state.messages if m['role']=='user'])

        st.markdown(f"""
        <div style="background:white;border-radius:20px;padding:26px;box-shadow:0 8px 32px rgba(0,0,0,0.12);">
            <div style="font-size:16px;font-weight:700;color:#1e293b;margin-bottom:18px;">⚙️ Setelan</div>
            <div style="font-size:13px;color:#475569;">
                {''.join(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                    padding:11px 0;border-bottom:1px solid #f1f5f9;">
                    <span>{lbl}</span>
                    <span style="font-weight:600;color:#1e293b;">{val}</span>
                </div>""" for lbl,val in [
                    ("🤖 AI Provider", provider),
                    ("🧠 MBTI Kamu", mbti_r),
                    ("💬 Total Pesan", str(total_msg)),
                    ("😊 Emosi Sekarang", f"{EMO_EMOJI.get(emo,'')} {EMO_LABEL.get(emo,'Netral')}"),
                ])}
                {'<div style="padding:10px 0;font-size:12px;color:#ef4444;"><b>Error:</b> '+err+'</div>' if err else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
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
            if st.button("🔄 Reset MBTI", key="reset_mbti_s", use_container_width=True):
                st.session_state.mbti        = None
                st.session_state.mbti_step   = 0
                st.session_state.mbti_resp   = []
                st.session_state.mbti_result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
