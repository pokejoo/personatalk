"""
🐼 PersonaTalk — Streamlit App
UI persis seperti referensi: light blue gradient, white sidebar, donut chart
Claude PRIMARY · Gemini FALLBACK
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
    import anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

try:
    import google.generativeai as genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO_ID    = "Jooou139/personatalk"
ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY    = st.secrets.get("GEMINI_API_KEY", "")
HF_TOKEN      = st.secrets.get("HF_TOKEN", "")

if _GENAI_OK and GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# ── Preprocessing ─────────────────────────────────────────────────────────────
STOPWORDS_EN = set(stopwords.words('english'))
lemmatizer   = WordNetLemmatizer()

def preprocess(text: str) -> str:
    if not text or not isinstance(text, str):
        return ''
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
EMO_FACE   = {0:'😢', 1:'😊', 2:'😍', 3:'😤', 4:'😰', 5:'😲'}
EMO_COLOR  = {0:'#5b8dd9', 1:'#f4a435', 2:'#e64980', 3:'#f03e3e', 4:'#9b59b6', 5:'#2ecc71'}
EMO_ICON   = {0:'🦊', 1:'🐱', 2:'🐰', 3:'🐯', 4:'🐭', 5:'🐨'}

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
    phrases = [
        ('patah hati',0),('putus sama',0),('diselingkuhin',0),('dikhianatin',0),
        ('overthinking',4),('deg degan',4),('was-was',4),
        ('seneng banget',1),('alhamdulillah',1),('lega banget',1),
        ('jatuh cinta',2),('pdkt',2),('naksir',2),
    ]
    for phrase, emo in phrases:
        if phrase in t: return emo
    kw_map = {
        0:['sedih','nangis','down','putus','ditinggal','sepi','galau','hampa','nyesel'],
        1:['bahagia','senang','happy','gembira','excited','lega','bangga','yay'],
        2:['cinta','sayang','rindu','kangen','naksir','gebetan'],
        3:['marah','kesal','benci','emosi','jengkel','muak','dongkol'],
        4:['takut','cemas','khawatir','panik','nervous','gelisah','stress','insecure'],
        5:['kaget','shock','terkejut','astaga'],
    }
    for emo, kws in kw_map.items():
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

def predict_mbti(text, mm, mv, hist):
    hist.append(text)
    if len(hist) < 2: return None, 0.0
    cl = preprocess(' '.join(hist[-10:]))
    if not cl: return None, 0.0
    X    = mv.transform([cl])
    pred = mm.predict(X)[0]
    conf = float(mm.predict_proba(X)[0].max())
    return pred, conf

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
    "S":"Sensing — fokus pada fakta & detail konkret",
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

def fmt_mbti(t):
    name, desc = MBTI_DESC.get(t, ("Unknown",""))
    dims = "\n".join(f"  {d} → {DIM_EXP[d]}" for d in t if d in DIM_EXP)
    return f"Tipe kamu: {t} — {name}\n\n{desc}\n\nDimensi:\n{dims}"

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah PersonaTalk — teman curhat AI yang terasa seperti sahabat dekat, bukan chatbot generik atau terapis formal.

## Identitas & karakter
Kamu hangat, empatik, kadang playful, selalu genuine dan tidak menghakimi. Kamu tidak menyebut diri sebagai AI, bot, atau assistant.

## Cara berbicara — wajib diikuti
- Bahasa Indonesia sehari-hari, santai, boleh sedikit mix English kalau terasa natural
- Respons pendek: 2-4 kalimat, tidak lebih
- Pembuka harus bervariasi setiap respons: "Duh,", "Ooh,", "Wah,", "Hmm,", "Ya Allah,", "Aduh,", "Serius?", "Oof," — TIDAK boleh pakai pembuka yang persis sama dua kali berturut-turut
- Akhiri dengan satu pertanyaan terbuka yang relevan dan BERBEDA dari pertanyaan sebelumnya
- Tidak pakai bullet point, list, atau format kaku

## Konteks — paling kritis
- Baca SELURUH riwayat percakapan sebelum menjawab
- Kalau user sudah menjawab pertanyaanmu → jangan tanya hal yang sama lagi, lanjutkan dari jawaban mereka
- Kalau user bercerita hal POSITIF (ketemu teman baru, senang, excited, crush) → MATCH the energy, ikut excited, jangan balas dengan nada negatif atau neutral yang datar
- Kalau topik sudah 4+ exchange dan masih sama → mulai berikan satu insight ringan atau saran praktis sambil tetap empati
- Kalau baru pertama cerita sesuatu → validasi dulu, gali lebih dalam, jangan langsung kasih solusi

## Panduan per emosi
- Sedih/galau: Validasi rasa sakitnya secara spesifik. Tanya yang konkret dan personal.
- Marah: Validasi bahwa kemarahan itu wajar. Tanya apa yang paling menyakitkan/frustasi.
- Cemas: Normalkan perasaannya. Bedakan overthinking vs ada trigger nyata.
- Senang/excited: Ikut antusias dengan tulus! Gali ceritanya, tanya detail.
- Cinta/crush: Playful tapi genuine. Tanya gimana dia orangnya atau progress-nya.
- Capek/burnout: Empati dulu, tanya ini lelah fisik, emosional, atau keduanya.

## Pantangan keras
- JANGAN mengulang pertanyaan yang sudah dijawab user
- JANGAN mengulang kalimat pembuka yang sama dari respons sebelumnya
- JANGAN diagnosis medis atau psikologis
- JANGAN lebih dari 4 kalimat
- JANGAN pakai template korporat seperti "Saya memahami perasaan Anda"
- JANGAN balas cerita senang dengan nada sedih atau kekhawatiran yang tidak relevan"""

# ── Duplicate Check ───────────────────────────────────────────────────────────
def is_dup(new: str, prev: list, thr=0.50) -> bool:
    if not prev or not new: return False
    nc = new.lower().strip()
    for old in prev[-3:]:
        oc = old.lower().strip()
        if nc == oc: return True
        nw, ow = set(nc.split()), set(oc.split())
        if len(nw & ow) / max(len(nw | ow), 1) > thr: return True
        nf = nc.split('?')[0].split('.')[0].strip()
        of = oc.split('?')[0].split('.')[0].strip()
        if nf and of and len(nf) > 12 and nf == of: return True
    return False

# ── Claude message builder ────────────────────────────────────────────────────
def build_messages(history: list, emotion_id: int, last_resp: list) -> list:
    emo_name  = EMO_LABEL.get(emotion_id, "Netral")
    no_repeat = ""
    if last_resp:
        no_repeat = " | Jangan ulangi pola: " + " // ".join(r[:70] for r in last_resp[-2:])

    msgs   = []
    recent = history[-14:-1] if len(history) > 1 else []

    for msg in recent:
        role    = "assistant" if msg['role'] == 'bot' else "user"
        content = msg['content']
        if msgs and msgs[-1]['role'] == role:
            msgs[-1]['content'] += "\n" + content
        else:
            msgs.append({'role': role, 'content': content})

    hint = f"[Emosi user saat ini: {emo_name}{no_repeat}]"
    if msgs and msgs[0]['role'] == 'user':
        msgs[0]['content'] = hint + "\n" + msgs[0]['content']
    else:
        msgs.insert(0, {'role': 'user', 'content': hint})
        if len(msgs) > 1 and msgs[1]['role'] != 'assistant':
            msgs.insert(1, {'role': 'assistant', 'content': 'Oke, aku dengerin.'})

    cur = history[-1]['content'] if history and history[-1]['role'] == 'user' else ""
    if cur:
        if msgs and msgs[-1]['role'] == 'user':
            msgs[-1]['content'] += "\n" + cur
        else:
            msgs.append({'role': 'user', 'content': cur})

    if not msgs or msgs[0]['role'] != 'user':
        msgs.insert(0, {'role': 'user', 'content': hint})

    return msgs

def clean_text(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\[Emosi user.*?\]', '', text, flags=re.DOTALL)
    text = re.sub(r'##\s+', '', text)
    return text.strip()

# ── AI Response ───────────────────────────────────────────────────────────────
def get_ai_response(user_text: str, emotion_id: int, history: list, last_resp: list):
    last_resp = last_resp or []

    if _ANTHROPIC_OK and ANTHROPIC_KEY:
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            msgs   = build_messages(history, emotion_id, last_resp)
            out    = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                temperature=0.85,
                system=SYSTEM_PROMPT,
                messages=msgs,
            )
            text = clean_text(out.content[0].text)
            if text and len(text) > 10:
                if not is_dup(text, last_resp):
                    st.session_state['_provider'] = 'claude'
                    st.session_state['_ai_err']   = None
                    return text
                retry = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=400,
                    temperature=0.95,
                    system=SYSTEM_PROMPT,
                    messages=msgs + [
                        {'role': 'assistant', 'content': text},
                        {'role': 'user', 'content': 'Responmu terlalu mirip dengan sebelumnya. Gunakan pembuka yang berbeda sama sekali dan pertanyaan penutup yang beda topik.'},
                    ],
                )
                text2 = clean_text(retry.content[0].text)
                if text2 and len(text2) > 10:
                    st.session_state['_provider'] = 'claude'
                    st.session_state['_ai_err']   = None
                    return text2
        except Exception as e:
            st.session_state['_ai_err'] = f"Claude: {str(e)[:120]}"

    if _GENAI_OK and GEMINI_KEY:
        emo_name = EMO_LABEL.get(emotion_id, "Netral")
        recent   = history[-10:-1] if len(history) > 1 else []
        ctx      = "\n".join(("User" if m['role']=='user' else "PersonaTalk")+": "+m['content'] for m in recent)
        no_rep   = (" | JANGAN ulangi: " + " | ".join(r[:60] for r in last_resp[-2:])) if last_resp else ""
        prompt   = f"Emosi user: {emo_name}{no_rep}\n\nRiwayat:\n{ctx}\n\nPesan user: \"{user_text}\"\n\nBalas sebagai PersonaTalk. 3-4 kalimat, natural, pembuka bervariasi."
        try:
            mdl = genai.GenerativeModel("gemini-2.0-flash")
            r   = mdl.generate_content(
                content=SYSTEM_PROMPT+"\n\n"+prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.85, top_p=0.95, max_output_tokens=400, top_k=40),
                safety_settings=[
                    {"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"},
                    {"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"},
                    {"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"BLOCK_MEDIUM_AND_ABOVE"},
                    {"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_MEDIUM_AND_ABOVE"},
                ],
            )
            text = clean_text(r.text)
            if text and len(text) > 10 and not is_dup(text, last_resp):
                st.session_state['_provider'] = 'gemini'
                st.session_state['_ai_err']   = None
                return text
        except Exception as e:
            prev = st.session_state.get('_ai_err','')
            st.session_state['_ai_err'] = (prev+" | " if prev else "") + f"Gemini: {str(e)[:100]}"

    return None

# ── Smart Fallback ────────────────────────────────────────────────────────────
def fallback_response(text: str, emotion: int, history: list) -> str:
    t  = text.lower()
    fc = ' '.join(m['content'].lower() for m in (history or [])[-5:] if m['role']=='user')

    kw_pools = {
        ('putus','diputus','ditinggal'): [
            "Duh, berasa ada yang ilang tiba-tiba ya. Kamu lagi banyak sendiri atau ada yang temani?",
            "Aduh, ini pasti menyakitkan. Udah cerita ke orang terdekat belum?",
            "Ya Allah, breakup tuh berat. Ini baru terjadi atau udah beberapa hari?",
        ],
        ('selingkuh','diselingkuhin','dikhianatin'): [
            "Oof, diselingkuhin itu rasa sakitnya berlapis. Dia udah tahu ketahuan?",
            "Ya Allah, dikhianatin sama orang yang dipercaya itu beda levelnya. Kamu lagi gimana?",
        ],
        ('bingung','blank','lost','nggak tahu'): [
            "Hmm, blank kayak gini biasanya karena terlalu banyak yang dipikirin sekaligus. Paling bikin stuck soal apa?",
            "Ngerasa lost itu berat. Kamu butuh didengar atau butuh arah konkret?",
        ],
        ('rindu','kangen','missing'): [
            "Kangen yang dalam kayak gini nyesek banget. Kamu kangen orangnya, momennya, atau keduanya?",
            "Ooh, rindu kayak gini biasanya tanda ada hal penting yang kamu miss. Udah berapa lama?",
        ],
        ('capek','lelah','exhausted','burnout'): [
            "Capek yang kayak gini beda — bukan cuma fisik. Ini dari kerjaan, hubungan, atau semua sekaligus?",
            "Aduh, burnout kayak gini nyata dan berat. Kamu kapan terakhir beneran istirahat?",
        ],
        ('cemas','khawatir','overthinking','gelisah','stress'): [
            "Gelisah kayak gini nggak enak. Ini overthinking atau ada hal konkret yang bikin khawatir?",
            "Cemas yang dalam kayak gini biasanya ada trigger-nya. Soal apa yang paling bikin was-was?",
        ],
        ('crush','naksir','suka sama','gebetan','cantik','ganteng'): [
            "Ooh, ada yang spesial nih! Dia udah tau kamu naksir?",
            "Wah, ada yang bikin deg-degan! Gimana interaksi kalian sejauh ini?",
            "Duh, ada crush! Dia orangnya gimana?",
        ],
    }
    for kws, opts in kw_pools.items():
        if any(k in fc or k in t for k in kws):
            return random.choice(opts)

    emo_defaults = {
        1: ["Wah, kedengarannya ada yang bagus nih! Apaan sih yang terjadi?",
            "Ooh, excited banget dengernya! Cerita dong dari awal.",
            "Duh, seneng banget dengernya. Gimana perasaan kamu sekarang?"],
        2: ["Ooh, ada yang spesial nih! Gimana dia?",
            "Wah, ada yang bikin deg-degan! Cerita lebih dong."],
        3: ["Kemarahan kayak gini valid banget. Ini marah sama orangnya atau situasinya?",
            "Oof, emosi banget nih. Apaan yang paling bikin gemas?"],
        4: ["Gelisah kayak gini nggak enak. Ini soal apa yang paling bikin khawatir?",
            "Cemas yang dalam kayak gini berat. Udah berapa lama ngerasa gini?"],
        0: ["Duh, kedengarannya berat. Mau cerita lebih? Aku dengerin.",
            "Aduh, ada yang lagi berat dipikul. Dari mana mau mulai ceritanya?"],
        5: ["Serius?! Apaan yang bikin kaget banget?",
            "Astaga, unexpected banget! Gimana ceritanya?"],
    }
    return random.choice(emo_defaults.get(emotion, [
        "Hmm, ada apa yang lagi kamu pikirin? Cerita yuk.",
        "Duh, kedengarannya ada sesuatu. Aku di sini kok.",
    ]))

# ── Mood Donut SVG ────────────────────────────────────────────────────────────
def mood_donut_svg(emo_counts: dict) -> str:
    total  = sum(emo_counts.values()) or 1
    colors = [EMO_COLOR[i] for i in range(6)]
    labels = [EMO_LABEL[i] for i in range(6)]
    pcts   = [emo_counts.get(i, 0) / total for i in range(6)]

    r, cx, cy = 42, 50, 50
    paths, offset = "", 0.0

    for i, pct in enumerate(pcts):
        if pct < 0.005:
            offset += pct
            continue
        a1    = offset * 360 - 90
        a2    = (offset + pct) * 360 - 90
        large = 1 if pct > 0.5 else 0

        def pt(angle, _r=r, _cx=cx, _cy=cy):
            rad = math.radians(angle)
            return _cx + _r * math.cos(rad), _cy + _r * math.sin(rad)

        x1, y1 = pt(a1)
        x2, y2 = pt(a2)
        paths += (f'<path d="M {cx} {cy} L {x1:.2f} {y1:.2f} '
                  f'A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f} Z" '
                  f'fill="{colors[i]}"/>')
        offset += pct

    has_data   = any(v > 0 for v in emo_counts.values())
    dominant   = max(emo_counts, key=emo_counts.get) if has_data else 1
    dom_pct    = int(pcts[dominant] * 100)
    dom_label  = EMO_LABEL[dominant]
    dom_face   = EMO_FACE[dominant]

    # Legend rows — only show emotions with data
    legend_rows = ""
    for i in range(6):
        if pcts[i] < 0.01: continue
        pct_val = int(pcts[i] * 100)
        legend_rows += f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="font-size:14px;">{EMO_FACE[i]}</span>
            <span style="font-size:13px;color:#334155;font-weight:500;flex:1;">{labels[i]}</span>
            <span style="font-size:13px;color:#64748b;font-weight:600;">{pct_val}%</span>
        </div>"""

    if not legend_rows:
        legend_rows = """
        <div style="font-size:13px;color:#94a3b8;padding-top:4px;">
            Mulai chat untuk analisis mood
        </div>"""

    # SVG donut — white center with emoji + percentage
    svg = f"""
    <svg viewBox="0 0 100 100" width="110" height="110" style="flex-shrink:0;filter:drop-shadow(0 4px 12px rgba(0,0,0,0.15));">
        <circle cx="50" cy="50" r="42" fill="#dbeafe" opacity="0.5"/>
        {paths if has_data else f'<circle cx="50" cy="50" r="42" fill="{EMO_COLOR[dominant]}"/>'}
        <circle cx="50" cy="50" r="28" fill="white"/>
        <text x="50" y="46" text-anchor="middle" fill="#1e40af"
              font-size="14" font-family="serif">{dom_face}</text>
        <text x="50" y="60" text-anchor="middle" fill="#1e40af"
              font-size="9" font-weight="bold">{dom_pct}%</text>
    </svg>"""

    return f"""
    <div style="display:flex;align-items:center;gap:20px;">
        {svg}
        <div style="flex:1;">
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#64748b;margin-bottom:10px;">
                Mood bagus!
            </div>
            {legend_rows}
        </div>
    </div>"""

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

/* ── Outer app background ── */
.stApp {
    background: #f0f4ff !important;
}
.main > div {
    padding: 0 !important;
    background: transparent !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Sidebar — clean white ── */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04) !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important;
}
section[data-testid="stSidebar"] * {
    color: #334155 !important;
}

/* ── Form / Input ── */
.stTextInput > div > div > input {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 28px !important;
    color: #1e293b !important;
    font-size: 14px !important;
    padding: 13px 22px !important;
    caret-color: #3b82f6;
    box-shadow: none !important;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
    background: white !important;
}
.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;
}
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ── Send button ── */
.stButton > button {
    background: #3b82f6 !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 46px !important;
    height: 46px !important;
    padding: 0 !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    transition: background 0.2s !important;
    min-width: 46px !important;
}
.stButton > button:hover {
    background: #2563eb !important;
}

/* ── Radio ── */
div[role="radiogroup"] label {
    color: #475569 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
    border-radius: 99px !important;
}
.stProgress > div > div > div {
    background: #e2e8f0 !important;
    border-radius: 99px !important;
}

/* ── Chat bubbles ── */
.msg-row {
    display: flex;
    align-items: flex-end;
    margin-bottom: 12px;
    animation: fadeUp 0.2s ease;
    gap: 8px;
}
.msg-row.user-row { flex-direction: row-reverse; }

.bbl {
    max-width: 66%;
    padding: 12px 16px;
    font-size: 14px;
    line-height: 1.65;
    word-wrap: break-word;
    white-space: pre-wrap;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.bbl.bot-bbl {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 4px 18px 18px 18px;
    color: #1e293b;
}
.bbl.user-bbl {
    background: linear-gradient(135deg, #7eb3f8, #a78bfa);
    border-radius: 18px 4px 18px 18px;
    color: white;
}
.av-icon {
    font-size: 1.3rem;
    flex-shrink: 0;
    margin-bottom: 2px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar renderer ──────────────────────────────────────────────────────────
def render_sidebar(emotion_id, confidence, mbti, ai_status, mode):
    emo_emoji = EMO_EMOJI.get(emotion_id, '😐')
    emo_label = EMO_LABEL.get(emotion_id, 'Netral')
    emo_color = EMO_COLOR.get(emotion_id, '#3b82f6')

    status_map = {
        'claude': '🟢 Claude aktif',
        'gemini': '🟡 Gemini aktif',
        'error':  '🔴 Error',
        'none':   '⚪ Siap',
    }
    status_txt = status_map.get(ai_status, '⚪ Siap')

    mbti_block = (
        f'<div style="background:linear-gradient(135deg,#3b82f6,#6366f1);border-radius:10px;'
        f'padding:10px;text-align:center;font-weight:700;font-size:1.1rem;color:white;">{mbti}</div>'
        if mbti else
        '<div style="font-size:11px;color:#94a3b8;">Terdeteksi otomatis dari chat</div>'
    )

    # Nav items
    nav_items = [
        ("🏠", "Dashboard",     False),
        ("🕐", "Riwayat Chat",  False),
        ("ℹ️",  "Tentang",      False),
    ]
    nav_html = '<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin:16px 0 8px;">Panel Analisis</div>'
    panel_items = [
        ("💬", "Curhat",       mode == 'curhat'),
        ("🔮", "Analisis MBTI", mode == 'mbti'),
    ]

    def nav_item(icon, label, active=False):
        bg  = "background:#eff6ff;border-radius:10px;" if active else ""
        clr = "color:#2563eb;font-weight:600;" if active else "color:#64748b;"
        return f'<div style="display:flex;align-items:center;gap:10px;padding:9px 12px;margin-bottom:2px;cursor:pointer;{bg}"><span style="font-size:14px;">{icon}</span><span style="font-size:13px;{clr}">{label}</span></div>'

    top_nav   = "".join(nav_item(i, l, a) for i, l, a in nav_items)
    panel_nav = "".join(nav_item(i, l, a) for i, l, a in panel_items)

    st.markdown(f"""
<div style="padding:20px 16px;height:100%;display:flex;flex-direction:column;">

    <!-- Logo -->
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#3b82f6,#6366f1);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;
                    font-size:18px;color:white;">🐼</div>
        <div>
            <div style="font-weight:700;font-size:15px;color:#0f172a;line-height:1.2;">persona<br>talk</div>
        </div>
        <div style="margin-left:auto;font-size:18px;color:#94a3b8;cursor:pointer;">☰</div>
    </div>

    <!-- Search bar -->
    <div style="background:#f1f5f9;border-radius:10px;padding:9px 14px;
                display:flex;align-items:center;gap:8px;margin-bottom:20px;">
        <span style="color:#94a3b8;font-size:14px;">🔍</span>
        <span style="font-size:13px;color:#94a3b8;">Search...</span>
    </div>

    <!-- Top nav -->
    {top_nav}

    <!-- Panel analisis -->
    {nav_html}
    {panel_nav}

    <div style="flex:1;"></div>

    <!-- AI status -->
    <div style="font-size:11px;color:#94a3b8;margin-bottom:8px;">{status_txt}</div>

    <!-- Settings -->
    <div style="display:flex;align-items:center;gap:8px;padding:9px 12px;cursor:pointer;">
        <span style="font-size:14px;">⚙️</span>
        <span style="font-size:13px;color:#64748b;">Setelan</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="PersonaTalk",
        page_icon="🐼",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    with st.spinner("Memuat PersonaTalk..."):
        emo_model, emo_vec, mbti_model, mbti_vec = load_models()

    # Session defaults
    defaults = {
        'messages':    [{'role':'bot','content':'Halo! 👋 Aku PersonaTalk, teman curhat kamu.\n\nMau cerita apa hari ini?'}],
        'emotion':     1,
        'confidence':  0.5,
        'mbti':        None,
        'mbti_texts':  [],
        'mode':        'curhat',
        'q_idx':       -1,
        'q_resp':      [],
        'last_bot':    [],
        'emo_counts':  {i: 0 for i in range(6)},
        '_provider':   None,
        '_ai_err':     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        ai_status = 'error' if st.session_state['_ai_err'] else (st.session_state['_provider'] or 'none')
        render_sidebar(
            st.session_state.emotion,
            st.session_state.confidence,
            st.session_state.mbti,
            ai_status,
            st.session_state.mode,
        )
        st.markdown("---")
        mode_sel = st.radio("Mode", ["💬 Curhat", "🔮 Analisis MBTI"], label_visibility="collapsed")
        new_mode = 'curhat' if '💬' in mode_sel else 'mbti'
        if new_mode != st.session_state.mode:
            st.session_state.mode   = new_mode
            st.session_state.q_idx  = -1
            st.session_state.q_resp = []
        st.markdown("---")
        if st.button("🔄 Reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        if st.session_state['_ai_err']:
            st.markdown(
                f'<div style="font-size:10px;color:#ef4444;margin-top:8px;word-break:break-all;">'
                f'⚠️ {st.session_state["_ai_err"][:150]}</div>',
                unsafe_allow_html=True,
            )

    # ── Main content wrapper — blue gradient card ─────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(160deg,#1d4ed8 0%,#3b82f6 35%,#60a5fa 65%,#93c5fd 100%);
            border-radius:24px;padding:28px 32px 24px;margin-bottom:0;
            min-height:calc(100vh - 60px);display:flex;flex-direction:column;gap:0;">
""", unsafe_allow_html=True)

    # ── Header row: greeting + donut ─────────────────────────────────────────
    emo_id   = st.session_state.emotion
    dom_emo  = max(st.session_state.emo_counts, key=st.session_state.emo_counts.get) \
               if any(v > 0 for v in st.session_state.emo_counts.values()) else 1

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown(f"""
<div style="padding:8px 0 20px;">
    <div style="font-size:2rem;font-weight:800;color:white;margin-bottom:6px;line-height:1.2;">
        Halo, Kamu! 👋
    </div>
    <div style="font-size:15px;color:rgba(255,255,255,0.8);line-height:1.5;">
        Bagaimana kabar mu hari ini?
    </div>
</div>
""", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
<div style="background:rgba(255,255,255,0.15);backdrop-filter:blur(12px);
            border:1px solid rgba(255,255,255,0.25);border-radius:18px;
            padding:16px 20px;">
    <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                letter-spacing:0.08em;color:rgba(255,255,255,0.7);margin-bottom:12px;">
        Detektor Mood Hari Ini
    </div>
    {mood_donut_svg(st.session_state.emo_counts)}
</div>
""", unsafe_allow_html=True)

    # ── Chat card ─────────────────────────────────────────────────────────────
    mode_label = "Curhat" if st.session_state.mode == 'curhat' else "Analisis MBTI"
    st.markdown(f"""
<div style="background:white;border-radius:18px;padding:20px 24px 16px;
            margin-top:20px;flex:1;
            box-shadow:0 8px 32px rgba(0,0,0,0.12);">
    <div style="font-size:14px;font-weight:600;color:#64748b;margin-bottom:16px;
                border-bottom:1px solid #f1f5f9;padding-bottom:12px;">
        {mode_label}
    </div>
""", unsafe_allow_html=True)

    # MBTI progress
    if st.session_state.mode == 'mbti' and st.session_state.q_idx >= 0:
        total = len(MBTI_Q)
        st.progress(min((st.session_state.q_idx + 1) / total, 1.0))
        st.caption(f"Pertanyaan {st.session_state.q_idx + 1} dari {total}")

    # ── Chat messages ─────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        is_user = msg['role'] == 'user'
        av      = '👤' if is_user else EMO_ICON.get(st.session_state.emotion, '🐼')
        content = msg['content'].replace('<', '&lt;').replace('>', '&gt;')
        row_cls = "user-row" if is_user else ""
        bbl_cls = "user-bbl" if is_user else "bot-bbl"
        if is_user:
            st.markdown(
                f'<div class="msg-row {row_cls}">'
                f'<div class="bbl {bbl_cls}">{content}</div>'
                f'<div class="av-icon">{av}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="msg-row {row_cls}">'
                f'<div class="av-icon">{av}</div>'
                f'<div class="bbl {bbl_cls}">{content}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Close chat card div
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Input bar ─────────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:white;border-radius:18px;padding:12px 16px;margin-top:12px;
            box-shadow:0 4px 16px rgba(0,0,0,0.1);display:flex;align-items:center;gap:10px;">
    <div style="width:32px;height:32px;background:#f1f5f9;border-radius:50%;
                display:flex;align-items:center;justify-content:center;
                font-size:16px;cursor:pointer;flex-shrink:0;">＋</div>
""", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([8, 1])
        with c1:
            ph = "Ketik pesan..." if st.session_state.mode == 'curhat' else "Jawab A atau B..."
            user_input = st.text_input("", placeholder=ph, label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button("➤", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Close outer gradient div
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Handle submit ─────────────────────────────────────────────────────────
    if submitted and user_input.strip():
        user_text = user_input.strip()
        st.session_state.messages.append({'role': 'user', 'content': user_text})

        # ── Curhat ────────────────────────────────────────────────────────────
        if st.session_state.mode == 'curhat':
            emo_id, conf = predict_emotion(user_text, emo_model, emo_vec)
            st.session_state.emotion    = emo_id
            st.session_state.confidence = conf
            st.session_state.emo_counts[emo_id] = st.session_state.emo_counts.get(emo_id, 0) + 1

            mbti_p, mbti_c = predict_mbti(user_text, mbti_model, mbti_vec, st.session_state.mbti_texts)
            if mbti_p and mbti_c > 0.3:
                st.session_state.mbti = mbti_p

            response = get_ai_response(user_text, emo_id, st.session_state.messages, st.session_state.last_bot)

            if not response:
                response = fallback_response(user_text, emo_id, st.session_state.messages)
                if is_dup(response, st.session_state.last_bot):
                    safety_pool = [
                        "Hmm, gimana kamu ngerasa sekarang?",
                        "Ooh, mau cerita lebih?",
                        "Duh, ada yang bisa aku bantu lebih lanjut?",
                        "Aduh, itu kedengarannya penting. Lanjutin dong.",
                        "Wah, interesting! Gimana kelanjutannya?",
                    ]
                    for opt in random.sample(safety_pool, len(safety_pool)):
                        if not is_dup(opt, st.session_state.last_bot):
                            response = opt
                            break

            st.session_state.last_bot.append(response)
            if len(st.session_state.last_bot) > 10:
                st.session_state.last_bot.pop(0)

        # ── MBTI ──────────────────────────────────────────────────────────────
        else:
            if st.session_state.q_idx == -1:
                st.session_state.q_resp = []
                st.session_state.q_idx  = 0
                q = MBTI_Q[0]
                response = (
                    f"Oke, yuk mulai analisis kepribadianmu! 🎯\n\n"
                    f"Pertanyaan 1 dari {len(MBTI_Q)}:\n\n{q['q']}\n\n"
                    f"A. {q['A']}\nB. {q['B']}\n\nJawab A atau B ya 😊"
                )
            else:
                ans = user_text.strip().upper()
                if ans in ['A', 'B']:
                    cq  = MBTI_Q[st.session_state.q_idx]
                    st.session_state.q_resp.append({"qid": cq["id"], "ans": ans})
                    nxt = st.session_state.q_idx + 1
                    if nxt < len(MBTI_Q):
                        st.session_state.q_idx = nxt
                        q = MBTI_Q[nxt]
                        response = (
                            f"Pertanyaan {q['id']} dari {len(MBTI_Q)}:\n\n{q['q']}\n\n"
                            f"A. {q['A']}\nB. {q['B']}\n\nJawab A atau B ya 😊"
                        )
                    else:
                        mtype = analyze_mbti(st.session_state.q_resp)
                        st.session_state.mbti  = mtype
                        st.session_state.q_idx = -1
                        response = (
                            f"✨ Analisis selesai!\n\n{fmt_mbti(mtype)}\n\n"
                            f"---\nPindah ke mode Curhat kalau mau ngobrol santai 😊"
                        )
                else:
                    response = "Jawabnya A atau B aja ya 😄 Coba lagi!"

        st.session_state.messages.append({'role': 'bot', 'content': response})
        st.rerun()


if __name__ == "__main__":
    main()
