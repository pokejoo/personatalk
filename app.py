"""
PersonaTalk — UI dari referensi + HuggingFace models (Jooou139/personatalk)
"""

import streamlit as st
import numpy as np
import re
import os
import random

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',  quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import joblib
from huggingface_hub import hf_hub_download

import plotly.graph_objects as go

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

# ── Load Models dari HuggingFace ───────────────────────────────────────────────
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
emotion_labels = {0:'Sedih', 1:'Bahagia', 2:'Cinta', 3:'Marah', 4:'Takut', 5:'Terkejut'}
mood_icons     = {0:'😔', 1:'😊', 2:'❤️', 3:'😠', 4:'😨', 5:'😲'}

emotion_lexicon = {
    "sedih":0,"kecewa":0,"patah hati":0,"galau":0,"murung":0,"terpukul":0,
    "hancur":0,"terluka":0,"down":0,"nangis":0,"menangis":0,"ditinggal":0,
    "diputus":0,"putus":0,"kehilangan":0,"sepi":0,"kesepian":0,"bingung":0,
    "sendirian":0,"hampa":0,"kosong":0,"hopeless":0,"nyesel":0,"frustasi":0,
    "sakit hati":0,"dikhianati":0,"selingkuh":0,"diselingkuhi":0,
    "marah":3,"kesal":3,"sebal":3,"benci":3,"jengkel":3,"emosi":3,"kesel":3,
    "muak":3,"tersinggung":3,"dibohongi":3,"ditipu":3,"nggak adil":3,"gak adil":3,
    "senang":1,"bahagia":1,"gembira":1,"happy":1,"lega":1,"bangga":1,"excited":1,
    "alhamdulillah":1,"syukur":1,"semangat":1,"berhasil":1,"sukses":1,
    "takut":4,"cemas":4,"khawatir":4,"panik":4,"gelisah":4,"nervous":4,
    "overthinking":4,"anxiety":4,"anxious":4,"was-was":4,"stress":4,"resah":4,
    "cinta":2,"sayang":2,"suka":2,"jatuh cinta":2,"kangen":2,"rindu":2,"naksir":2,
    "kaget":5,"terkejut":5,"shock":5,"astaga":5,"ga nyangka":5,"nggak nyangka":5,
}

def rule_based_emotion(text: str):
    t = text.lower()
    phrase_rules = {
        0: ["patah hati","putus sama","diselingkuhin","dikhianatin","sakit hati",
            "ditinggal pergi","capek banget","lelah banget","bingung banget",
            "ga tau harus","nggak tau harus","hilang arah","ga semangat"],
        3: ["nggak adil","gak adil","dibohongin","ditipu","dimanfaatin"],
        4: ["overthinking","was-was","nggak tenang","ga tenang","takut banget",
            "khawatir banget","deg degan","nggak yakin","ga yakin"],
        1: ["seneng banget","happy banget","lega banget","alhamdulillah","bangga banget"],
    }
    for emo, phrases in phrase_rules.items():
        if any(ph in t for ph in phrases): return emo
    rules = {
        0: ["sedih","nangis","down","putus","ditinggal","sepi","galau","hampa","nyesel"],
        1: ["bahagia","senang","happy","gembira","excited","lega","bangga"],
        2: ["cinta","sayang","rindu","kangen","jatuh cinta","naksir"],
        3: ["marah","kesal","benci","emosi","jengkel","muak","sebal"],
        4: ["takut","cemas","khawatir","panik","nervous","gelisah","anxious","stress"],
        5: ["kaget","shock","terkejut","astaga"],
    }
    for emo, kws in rules.items():
        if any(k in t for k in kws): return emo
    return None

def lexicon_emotion(text: str):
    t = text.lower()
    sc = {}
    for w, e in emotion_lexicon.items():
        if w in t: sc[e] = sc.get(e,0)+1
    return max(sc, key=sc.get) if sc else None

def predict_emotion_ml(text, em, ev):
    cl = preprocess(text)
    if not cl: return 0, 0.5
    pr = em.predict_proba(ev.transform([cl]))[0]
    return int(np.argmax(pr)), float(pr.max())

def predict_emotion(text, em, ev):
    rb = rule_based_emotion(text)
    if rb is not None: return rb, 0.95
    lx = lexicon_emotion(text)
    if lx is not None: return lx, 0.85
    return predict_emotion_ml(text, em, ev)

def predict_mbti_passive(text, mm, mv, hist):
    hist.append(text)
    if len(hist) < 2: return None, 0.0
    cl = preprocess(' '.join(hist[-10:]))
    if not cl: return None, 0.0
    X  = mv.transform([cl])
    return mm.predict(X)[0], float(mm.predict_proba(X)[0].max())

# ── MBTI ───────────────────────────────────────────────────────────────────────
MBTI_QUESTIONS = [
    {"id":1,"question":"Ketika menghadapi masalah besar, kamu lebih suka:","A":"Mencari solusi praktis dan langsung bertindak","B":"Merenung dan memikirkan berbagai kemungkinan dulu","weight":{"A":"S","B":"N"}},
    {"id":2,"question":"Di waktu luang, kamu lebih menikmati:","A":"Berkumpul dan bersosialisasi dengan banyak orang","B":"Waktu sendiri untuk mengisi ulang energi","weight":{"A":"E","B":"I"}},
    {"id":3,"question":"Saat mengambil keputusan penting, kamu lebih mengandalkan:","A":"Logika dan analisis objektif","B":"Perasaan dan nilai-nilai pribadi","weight":{"A":"T","B":"F"}},
    {"id":4,"question":"Gaya hidup sehari-harimu lebih ke mana?","A":"Terstruktur dengan jadwal dan rencana jelas","B":"Fleksibel dan mengikuti alur situasi","weight":{"A":"J","B":"P"}},
    {"id":5,"question":"Ketika belajar hal baru, kamu lebih suka:","A":"Langsung praktek dan hands-on","B":"Baca teori dan pahami konsepnya dulu","weight":{"A":"S","B":"N"}},
    {"id":6,"question":"Dalam group chat atau diskusi, biasanya kamu:","A":"Aktif merespon dan sering mulai topik baru","B":"Lebih banyak membaca, sesekali baru komentar","weight":{"A":"E","B":"I"}},
    {"id":7,"question":"Kalau teman curhat masalah, respons pertamamu biasanya:","A":"Langsung kasih solusi praktis","B":"Dengerin dulu dan kasih dukungan emosional","weight":{"A":"T","B":"F"}},
    {"id":8,"question":"Menjelang deadline, kamu biasanya:","A":"Selesaikan tugas jauh-jauh hari sebelumnya","B":"Paling produktif justru di menit-menit akhir","weight":{"A":"J","B":"P"}},
    {"id":9,"question":"Kamu lebih tertarik pada:","A":"Fakta, detail konkret, dan pengalaman nyata","B":"Pola, kemungkinan besar, dan gambaran besar","weight":{"A":"S","B":"N"}},
    {"id":10,"question":"Setelah seharian interaksi dengan banyak orang, kamu merasa:","A":"Makin bersemangat dan energized","B":"Lelah dan butuh waktu sendiri untuk recharge","weight":{"A":"E","B":"I"}},
]

MBTI_DESC = {
    "ISTJ":{"name":"The Logistician","desc":"Praktis, faktual, dan sangat terorganisir. Kamu adalah orang yang bisa diandalkan."},
    "ISFJ":{"name":"The Defender","desc":"Penyayang, hangat, dan selalu siap melindungi orang yang kamu cintai."},
    "INFJ":{"name":"The Advocate","desc":"Idealistik, berprinsip, dan visioner. Kamu melihat dunia dengan cara yang unik."},
    "INTJ":{"name":"The Architect","desc":"Strategis, logis, dan selalu berpikir jangka panjang. Masterplanner sejati."},
    "ISTP":{"name":"The Virtuoso","desc":"Pragmatis, fleksibel, dan jago banget memecahkan masalah teknis."},
    "ISFP":{"name":"The Adventurer","desc":"Artistik, spontan, dan selalu hidup di saat ini dengan penuh warna."},
    "INFP":{"name":"The Mediator","desc":"Idealis, empatik, dan selalu mencari makna di balik setiap hal."},
    "INTP":{"name":"The Logician","desc":"Inovatif, analitis, dan pencinta teori yang selalu ingin tahu."},
    "ESTP":{"name":"The Entrepreneur","desc":"Enerjik, action-oriented, dan persuasif. Kamu suka tantangan nyata."},
    "ESFP":{"name":"The Entertainer","desc":"Hangat, ramah, dan suka jadi pusat perhatian yang bikin suasana hidup."},
    "ENFP":{"name":"The Campaigner","desc":"Antusias, kreatif, dan inspiratif. Kamu bisa menyalakan semangat orang lain."},
    "ENTP":{"name":"The Debater","desc":"Cerdas, suka tantangan intelektual, dan inovatif dalam berpikir."},
    "ESTJ":{"name":"The Executive","desc":"Tegas, terorganisir, dan efisien. Natural leader yang hasil-oriented."},
    "ESFJ":{"name":"The Consul","desc":"Penyayang, populer, dan selalu mau membantu. Sosok yang semua orang suka."},
    "ENFJ":{"name":"The Protagonist","desc":"Karismatik, inspiratif, dan suka membimbing orang menuju versi terbaiknya."},
    "ENTJ":{"name":"The Commander","desc":"Berwibawa, strategis, dan pemimpin alami yang penuh visi besar."},
}

DIM_EXP = {
    "E":"Ekstrovert — Energimu datang dari interaksi sosial",
    "I":"Introvert — Energimu datang dari waktu sendiri",
    "S":"Sensing — Kamu fokus pada fakta dan detail konkret",
    "N":"Intuition — Kamu fokus pada pola dan kemungkinan besar",
    "T":"Thinking — Keputusanmu berdasarkan logika dan analisis",
    "F":"Feeling — Keputusanmu berdasarkan nilai dan perasaan",
    "J":"Judging — Kamu suka struktur dan perencanaan yang jelas",
    "P":"Perceiving — Kamu fleksibel dan adaptif terhadap situasi",
}

def analyze_mbti(responses):
    sc = {"E":0,"I":0,"S":0,"N":0,"T":0,"F":0,"J":0,"P":0}
    for r in responses:
        q = next((x for x in MBTI_QUESTIONS if x["id"]==r["question_id"]), None)
        if q and r["answer"] in q["weight"]: sc[q["weight"][r["answer"]]] += 1
    return (("E" if sc["E"]>=sc["I"] else "I") +
            ("S" if sc["S"]>=sc["N"] else "N") +
            ("T" if sc["T"]>=sc["F"] else "F") +
            ("J" if sc["J"]>=sc["P"] else "P"))

def format_mbti_result(mbti_type):
    info = MBTI_DESC.get(mbti_type, {"name":"Unknown","desc":""})
    dims = [f"• {d} → {DIM_EXP.get(d,'')}" for d in mbti_type if d in DIM_EXP]
    return (f"🧠 **Tipe Kepribadianmu: {mbti_type}**\n"
            f"📋 *{info['name']}*\n\n{info['desc']}\n\n"
            f"**Breakdown dimensimu:**\n" + "\n".join(dims))

# ── AI Response ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah PersonaTalk — sahabat curhat digital yang beneran peduli, bukan robot atau konselor formal.

KARAKTER KAMU:
Kamu seperti teman SMA/kuliah yang udah kenal lama, nyambung, hangat, dan nggak pernah ngehakimin.

CARA NGOBROL:
- Bahasa Indonesia sehari-hari, santai, campuran indo-english boleh
- Mulai SELALU dengan validasi perasaan dulu
- Empati harus berasa nyata, bukan template
- Gunakan ekspresi natural: "duh", "hmm", "ya Allah", "aduh", "serius?", "waduh"
- 2-4 kalimat aja, jangan panjang
- Di akhir selalu tanya SATU pertanyaan terbuka yang relevan
- Jangan pakai bullet point atau format kaku
- Jangan bilang "sebagai AI"

PANTANGAN:
- Jangan diagnosis medis/psikologis
- Jangan jawaban template generik
- Jangan lebih dari 4 kalimat"""

def get_ai_response(user_text, emotion_id, history, last_resp):
    if not _GROQ_OK or not GROQ_KEY: return None
    emo_name  = emotion_labels.get(emotion_id, "Netral")
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
             "Duh, itu pasti bikin kamu ngerasa down banget. Mau cerita lebih detail nggak?"],
        ('bingung','harus apa','mau ngapain'):
            ["Hmm, ngerasa kayak tiba-tiba harus jalan sendiri dan bingung mulai dari mana ya? Yang paling bikin kamu kepikiran sekarang apa?"],
        ('capek','lelah','burnout'):
            ["Hmm, capek yang kayak gini tuh beda sama capek biasa. Ini capek dari mana — fisik, pikiran, atau keduanya?"],
        ('marah','kesal','benci','kesel'):
            ["Iya, situasi kayak gitu emang bikin darah naik. Boleh cerita lebih — ini karena apa atau siapa?"],
        ('takut','cemas','khawatir','anxious'):
            ["Hmm, ngerasa cemas itu nggak enak banget ya. Ini soal apa yang bikin kamu khawatir?"],
    }.items():
        if any(k in t for k in kws): return random.choice(opts)
    openers = {
        0: ["Duh, kedengarannya berat banget... cerita lebih dong, aku dengerin."],
        1: ["Wah, ada kabar baik nih kayaknya! Cerita dong!"],
        2: ["Ada yang spesial nih kayaknya 😊 Cerita yuk!"],
        3: ["Kayaknya ada yang bikin kamu kesel nih. Boleh cerita — ini soal apa?"],
        4: ["Kayaknya ada yang lagi bikin kamu khawatir. Cerita yuk, biar agak lega."],
        5: ["Wah, ada kejutan nih! Cerita dong lebih lengkapnya!"],
    }
    return random.choice(openers.get(emotion, ["Hmm, cerita lebih yuk. Aku dengerin kok."]))

# ── Visualization ──────────────────────────────────────────────────────────────
def create_radar_chart(mbti: str):
    if not mbti or len(mbti) != 4: return None
    dimensions  = ['E vs I', 'S vs N', 'T vs F', 'J vs P']
    char_scores = {'E':75,'I':25,'S':75,'N':25,'T':75,'F':25,'J':75,'P':25}
    values = [char_scores.get(mbti[0],50), char_scores.get(mbti[1],50),
              char_scores.get(mbti[2],50), char_scores.get(mbti[3],50)]
    fig = go.Figure(data=go.Scatterpolar(
        r=values+[values[0]], theta=dimensions+[dimensions[0]],
        fill='toself',
        marker=dict(color='rgba(0,255,200,0.8)'),
        line=dict(color='#00ffc8', width=2)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], color='white'),
            angularaxis=dict(color='white'),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=250,
        margin=dict(l=30,r=30,t=30,b=30),
        showlegend=False
    )
    return fig

def get_labubu_avatar(emotion=0):
    mood_map = {
        0:{"emoji":"🦊","color":"#4a90e2"},
        1:{"emoji":"🐱","color":"#f1c40f"},
        2:{"emoji":"🐰","color":"#ff69b4"},
        3:{"emoji":"🐯","color":"#e74c3c"},
        4:{"emoji":"🐭","color":"#9b59b6"},
        5:{"emoji":"🐨","color":"#f39c12"},
    }
    mood = mood_map.get(emotion, {"emoji":"🐼","color":"#00ffff"})
    return f'<span style="font-size:2rem;filter:drop-shadow(0 0 8px {mood["color"]});">{mood["emoji"]}</span>'

def labubu_animation(emotion=0):
    chars  = {0:"🦊",1:"🐱",2:"🐰",3:"🐯",4:"🐭",5:"🐨"}
    labels = {0:"Sedih",1:"Happy",2:"Love",3:"Marah",4:"Cemas",5:"Kaget"}
    char  = chars.get(emotion,"🐼")
    label = labels.get(emotion,"Normal")
    return f"""
    <div style="display:flex;justify-content:center;align-items:center;
                padding:15px;background:rgba(255,255,255,0.04);
                border-radius:50px;margin:10px 0;border:1px solid rgba(255,255,255,0.08);">
        <span style="font-size:1.8rem;margin:0 8px;opacity:0.5;">🐼</span>
        <span style="font-size:2.8rem;margin:0 8px;animation:bounce 2s infinite ease-in-out;">{char}</span>
        <span style="font-size:1.8rem;margin:0 8px;opacity:0.5;">🐼</span>
    </div>
    <div style="text-align:center;margin:-5px 0 12px 0;">
        <span style="background:rgba(255,255,255,0.08);padding:4px 14px;
                     border-radius:20px;font-size:0.85rem;color:rgba(255,255,255,0.7);">
            Labubu lagi {label}
        </span>
    </div>
    <style>
    @keyframes bounce {{
        0%,100% {{ transform:translateY(0); }}
        50% {{ transform:translateY(-8px); }}
    }}
    </style>"""

# ── CSS ────────────────────────────────────────────────────────────────────────
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

.main > div {
    background: transparent !important;
    padding: 1rem 2rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10,10,25,0.95) !important;
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(0,255,200,0.15);
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #00ffc8 !important; }

/* ── Chat bubbles ── */
.chat-wrap {
    display: flex;
    align-items: flex-start;
    margin-bottom: 18px;
    animation: floatIn 0.3s ease;
}
.chat-wrap.user { flex-direction: row-reverse; }
.bubble {
    padding: 14px 18px;
    border-radius: 18px;
    max-width: 72%;
    font-size: 15px;
    line-height: 1.6;
    word-wrap: break-word;
    white-space: pre-wrap;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.bubble.bot {
    background: rgba(30,30,55,0.8);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 4px 18px 18px 18px;
    backdrop-filter: blur(8px);
}
.bubble.user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 18px 4px 18px 18px;
    border: 1px solid rgba(255,255,255,0.15);
}
.avatar { font-size: 1.8rem; margin: 0 10px; align-self: flex-end; }

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #00ffc8 !important; }
[data-testid="stMetricLabel"] { color: rgba(255,255,255,0.6) !important; }

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00ffc8, #667eea) !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    background: rgba(20,20,40,0.9) !important;
    border: 2px solid rgba(0,255,200,0.4) !important;
    border-radius: 30px !important;
    color: white !important;
    font-size: 15px !important;
    padding: 14px 22px !important;
    caret-color: #00ffc8;
}
.stTextInput > div > div > input:focus {
    border-color: #00ffc8 !important;
    box-shadow: 0 0 20px rgba(0,255,200,0.3) !important;
}
.stTextInput > div > div > input::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 14px 20px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    width: 100% !important;
    height: 52px !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 0 20px rgba(102,126,234,0.6) !important;
}

/* ── Radio ── */
div[role="radiogroup"] {
    background: rgba(15,25,45,0.8);
    padding: 8px;
    border-radius: 12px;
}
div[role="radiogroup"] label { color: white !important; }

/* ── Alert ── */
.stAlert {
    background: rgba(20,20,40,0.9) !important;
    border: 1px solid rgba(0,255,200,0.3) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ── Form ── */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}

@keyframes floatIn {
    from { opacity: 0; transform: translateY(15px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Welcome box ── */
.welcome-box {
    background: rgba(30,30,50,0.9);
    border: 1px solid rgba(0,255,200,0.25);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,255,200,0.1);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="PersonaTalk",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🐼"
    )
    inject_css()

    # Load models
    with st.spinner("⚡ Memuat PersonaTalk..."):
        emo_model, emo_vec, mbti_model, mbti_vec = load_models()

    # Session state
    defaults = {
        'messages': [{'role':'bot','content':"Halo! Aku PersonaTalk 🐼\n\nAku bisa jadi temen curhat kamu dan juga bantu analisis kepribadian MBTI-mu.\n\nMau ngobrol santai dulu atau langsung tes kepribadian? Cerita aja ya! 😊"}],
        'current_emotion':         1,
        'current_mbti':            None,
        'last_confidence':         0.5,
        'question_responses':      [],
        'current_question_index':  -1,
        'mode':                    '💬 Curhat',
        'last_bot':                [],
        'mbti_texts':              [],
        '_ai_err':                 None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Header ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style="text-align:center;padding:20px 0 10px 0;">
            <h1 style="background:linear-gradient(45deg,#00ffc8,#667eea);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       font-size:3rem;margin:0;">🐼 PersonaTalk</h1>
            <p style="color:rgba(255,255,255,0.7);font-size:1.1rem;margin-top:8px;">
                Teman curhat dengan deteksi emosi & analisis kepribadian MBTI
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Welcome box ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="welcome-box">
        <div style="font-size:2.5rem;margin-bottom:8px;">🐼✨</div>
        <div style="font-size:1.4rem;font-weight:800;
                    background:linear-gradient(45deg,#00ffc8,#667eea);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Selamat datang di PersonaTalk!
        </div>
        <div style="color:rgba(255,255,255,0.75);margin-top:10px;line-height:1.7;">
            Halo! Aku di sini buat nemenin kamu ngobrol dan ngebantu kenali kepribadianmu.<br>
            Pilih mode <b>Curhat</b> buat cerita santai, atau <b>Analisis MBTI</b>
            buat tau tipe kepribadianmu! 😊
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 🧠 Panel Analisis")
        st.markdown("---")

        # Mode switch
        mode = st.radio(
            "Mode Interaksi",
            ["💬 Curhat", "🧬 Analisis MBTI"],
            horizontal=True
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            if mode == "💬 Curhat":
                st.session_state.current_question_index = -1
                st.session_state.question_responses     = []

        st.markdown("---")

        # Mood panel
        st.markdown("### Mood Terdeteksi")
        emo = st.session_state.current_emotion
        current_mood = emotion_labels.get(emo, "Normal")
        st.markdown(f"""
        <div style="background:rgba(0,255,200,0.08);border-left:4px solid #00ffc8;
                    padding:12px;border-radius:10px;margin:8px 0;">
            <span style="font-size:1.5rem;">{mood_icons.get(emo,'🐼')}</span>
            <strong style="font-size:1.1rem;margin-left:8px;">{current_mood}</strong>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(labubu_animation(emo), unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {st.session_state.last_confidence:.0%}")
        st.progress(st.session_state.last_confidence)

        st.markdown("---")

        # MBTI panel
        if st.session_state.current_mbti:
            st.markdown("### 🧬 Tipe MBTI")
            fig = create_radar_chart(st.session_state.current_mbti)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="text-align:center;margin-top:8px;">
                <span style="background:linear-gradient(45deg,#00ffc8,#667eea);
                             padding:8px 22px;border-radius:30px;color:white;
                             font-weight:800;font-size:1.4rem;">
                    {st.session_state.current_mbti}
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("💬 Mulai percakapan untuk prediksi MBTI")

        st.markdown("---")

        # MBTI quiz progress
        if st.session_state.current_question_index >= 0:
            total    = len(MBTI_QUESTIONS)
            progress = min((st.session_state.current_question_index + 1) / total, 1.0)
            st.progress(progress)
            st.caption(f"Pertanyaan {min(st.session_state.current_question_index+1, total)} dari {total}")
            st.markdown("---")

        # Reset
        if st.button("🔄 Reset Chat", use_container_width=True):
            for key in ['messages','current_emotion','current_mbti','last_confidence',
                        'question_responses','current_question_index','last_bot',
                        'mbti_texts','_ai_err']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAT DISPLAY
    # ══════════════════════════════════════════════════════════════════════════
    for msg in st.session_state.messages:
        is_user      = msg['role'] == 'user'
        wrap_class   = "user" if is_user else "bot"
        bubble_class = "user" if is_user else "bot"
        avatar       = "👤" if is_user else get_labubu_avatar(st.session_state.current_emotion)

        if is_user:
            st.markdown(f"""
            <div class="chat-wrap {wrap_class}">
                <div class="bubble {bubble_class}">{msg['content']}</div>
                <div class="avatar">{avatar}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-wrap {wrap_class}">
                <div class="avatar">{avatar}</div>
                <div class="bubble {bubble_class}">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # INPUT FORM
    # ══════════════════════════════════════════════════════════════════════════
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([5,1])
        with c1:
            user_input = st.text_input(
                "", placeholder="Ketik pesanmu di sini...",
                label_visibility="collapsed"
            )
        with c2:
            submitted = st.form_submit_button("📤 Kirim", use_container_width=True)

    if submitted and user_input and user_input.strip():
        user_text = user_input.strip()
        st.session_state.messages.append({'role':'user','content':user_text})

        # ── CURHAT MODE ───────────────────────────────────────────────────────
        if st.session_state.mode == "💬 Curhat":
            emo_id, conf = predict_emotion(user_text, emo_model, emo_vec)
            st.session_state.current_emotion  = emo_id
            st.session_state.last_confidence  = conf

            mbti_p, mbti_c = predict_mbti_passive(
                user_text, mbti_model, mbti_vec, st.session_state.mbti_texts)
            if mbti_p and mbti_c > 0.3:
                st.session_state.current_mbti = mbti_p

            response = get_ai_response(
                user_text, emo_id,
                st.session_state.messages,
                st.session_state.last_bot
            )
            if not response:
                response = fallback_response(user_text, emo_id)

            st.session_state.last_bot.append(response)
            if len(st.session_state.last_bot) > 10:
                st.session_state.last_bot.pop(0)
            st.session_state.messages.append({'role':'bot','content':response})

        # ── MBTI MODE ─────────────────────────────────────────────────────────
        elif st.session_state.mode == "🧬 Analisis MBTI":
            if st.session_state.current_question_index == -1:
                st.session_state.question_responses     = []
                st.session_state.current_question_index = 0
                q        = MBTI_QUESTIONS[0]
                response = (f"Oke, yuk kita mulai analisis kepribadianmu! 🎯\n\n"
                            f"**Pertanyaan {q['id']} dari 10:**\n\n{q['question']}\n\n"
                            f"**A.** {q['A']}\n**B.** {q['B']}\n\n"
                            f"Jawab dengan **A** atau **B** ya 😊")
            else:
                answer = user_text.strip().upper()
                if answer in ['A','B']:
                    idx = st.session_state.current_question_index
                    q   = MBTI_QUESTIONS[idx]
                    st.session_state.question_responses.append({
                        "question_id": q["id"], "answer": answer
                    })
                    next_idx = idx + 1
                    st.session_state.current_question_index = next_idx

                    if next_idx < len(MBTI_QUESTIONS):
                        nq       = MBTI_QUESTIONS[next_idx]
                        response = (f"**Pertanyaan {nq['id']} dari 10:**\n\n"
                                    f"{nq['question']}\n\n"
                                    f"**A.** {nq['A']}\n**B.** {nq['B']}\n\n"
                                    f"Jawab dengan **A** atau **B** ya 😊")
                    else:
                        mbti_type = analyze_mbti(st.session_state.question_responses)
                        st.session_state.current_mbti            = mbti_type
                        st.session_state.current_question_index  = -1
                        result_text = format_mbti_result(mbti_type)
                        response = (f"✨ **Analisis selesai!**\n\n{result_text}\n\n---\n"
                                    f"Sekarang kamu bisa pindah ke mode **Curhat** kalau mau ngobrol santai 😊")
                else:
                    response = "Eh, jawabnya A atau B aja ya 😄 Coba lagi!"

            st.session_state.messages.append({'role':'bot','content':response})

        st.rerun()


if __name__ == "__main__":
    main()
