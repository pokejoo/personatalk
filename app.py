"""
🐼 PersonaTalk — Portfolio Style with AI Chat
UI: Streamlit with custom CSS, chat as popup/modal
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
EMO_EMOJI  = {0:'😔', 1:'😊', 2:'❤️',  3:'😠',  4:'😨',   5:'😲'}
EMO_COLOR  = {0:'#5b8dd9', 1:'#f59f00', 2:'#e64980', 3:'#f03e3e', 4:'#7950f2', 5:'#2ecc71'}
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

def predict_mbti(text, mm, mv, hist):
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
    return (("E" if sc["E"]>=sc["I"] else "I") + ("S" if sc["S"]>=sc["N"] else "N") +
            ("T" if sc["T"]>=sc["F"] else "F") + ("J" if sc["J"]>=sc["P"] else "P"))

def fmt_mbti(t):
    name, desc = MBTI_DESC.get(t, ("Unknown",""))
    dims = "\n".join(f"  {d} → {DIM_EXP[d]}" for d in t if d in DIM_EXP)
    return f"**{t} — {name}**\n\n{desc}\n\nDimensi kamu:\n{dims}"

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
- Kalau user bercerita hal POSITIF → MATCH the energy, ikut excited, jangan balas dengan nada negatif
- Kalau topik sudah 4+ exchange → mulai berikan satu insight ringan atau saran praktis sambil tetap empati
- Kalau baru pertama cerita → validasi dulu, gali lebih dalam, jangan langsung kasih solusi

## Panduan per emosi
- Sedih/galau: Validasi rasa sakitnya secara spesifik. Tanya yang konkret dan personal.
- Marah: Validasi bahwa kemarahan itu wajar. Tanya apa yang paling menyakitkan.
- Cemas: Normalkan perasaannya. Bedakan overthinking vs ada trigger nyata.
- Senang/excited: Ikut antusias dengan tulus! Gali ceritanya, tanya detail.
- Cinta/crush: Playful tapi genuine. Tanya gimana dia orangnya atau progress-nya.
- Capek/burnout: Empati dulu, tanya ini lelah fisik, emosional, atau keduanya.

## Pantangan keras
- JANGAN mengulang pertanyaan yang sudah dijawab user
- JANGAN mengulang kalimat pembuka yang sama dari respons sebelumnya
- JANGAN diagnosis medis atau psikologis
- JANGAN lebih dari 4 kalimat
- JANGAN pakai template korporat
- JANGAN balas cerita senang dengan nada sedih"""

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
                model="claude-sonnet-4-6", max_tokens=400,
                temperature=0.85, system=SYSTEM_PROMPT, messages=msgs,
            )
            text = clean_text(out.content[0].text)
            if text and len(text) > 10:
                if not is_dup(text, last_resp):
                    st.session_state['_provider'] = 'claude'
                    st.session_state['_ai_err']   = None
                    return text
                retry = client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=400, temperature=0.95,
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
    for kws, opts in {
        ('putus','diputus','ditinggal'): ["Duh, berasa ada yang ilang tiba-tiba ya. Kamu lagi sendirian atau ada yang temani?","Aduh, ini pasti menyakitkan. Udah cerita ke orang terdekat belum?"],
        ('selingkuh','diselingkuhin','dikhianatin'): ["Oof, diselingkuhin itu rasa sakitnya berlapis. Dia udah tahu ketahuan?","Ya Allah, dikhianatin sama orang yang dipercaya itu beda levelnya. Kamu lagi gimana?"],
        ('bingung','blank','lost'): ["Hmm, blank kayak gini biasanya karena terlalu banyak yang dipikirin. Paling bikin stuck soal apa?","Ngerasa lost itu berat. Kamu butuh didengar atau butuh arah konkret?"],
        ('rindu','kangen'): ["Kangen yang dalam kayak gini nyesek banget. Kamu kangen orangnya, momennya, atau keduanya?","Ooh, rindu kayak gini biasanya tanda ada hal penting yang kamu miss. Udah berapa lama?"],
        ('capek','lelah','burnout'): ["Capek yang kayak gini beda — bukan cuma fisik. Dari kerjaan, hubungan, atau semua sekaligus?","Aduh, burnout kayak gini nyata dan berat. Kamu kapan terakhir beneran istirahat?"],
        ('cemas','khawatir','overthinking','gelisah','stress'): ["Gelisah kayak gini nggak enak. Ini overthinking atau ada hal konkret yang bikin khawatir?","Cemas yang dalam kayak gini biasanya ada trigger-nya. Soal apa yang paling bikin was-was?"],
        ('crush','naksir','gebetan','cantik','ganteng'): ["Ooh, ada yang spesial nih! Dia udah tau kamu naksir?","Wah, ada yang bikin deg-degan! Gimana interaksi kalian sejauh ini?"],
    }.items():
        if any(k in fc or k in t for k in kws):
            return random.choice(opts)
    return random.choice({
        1: ["Wah, kedengarannya ada yang bagus nih! Apaan sih yang terjadi?","Ooh, excited banget dengernya! Cerita dong dari awal."],
        2: ["Ooh, ada yang spesial nih! Gimana dia?","Wah, ada yang bikin deg-degan! Cerita lebih dong."],
        3: ["Kemarahan kayak gini valid banget. Ini marah sama orangnya atau situasinya?","Oof, emosi banget nih. Apaan yang paling bikin gemas?"],
        4: ["Gelisah kayak gini nggak enak. Ini soal apa yang paling bikin khawatir?","Cemas yang dalam kayak gini berat. Udah berapa lama ngerasa gini?"],
        0: ["Duh, kedengarannya berat. Mau cerita lebih? Aku dengerin.","Aduh, ada yang lagi berat dipikul. Dari mana mau mulai ceritanya?"],
        5: ["Serius?! Apaan yang bikin kaget banget?","Astaga, unexpected banget! Gimana ceritanya?"],
    }.get(emotion, ["Hmm, ada apa yang lagi kamu pikirin? Cerita yuk.","Duh, kedengarannya ada sesuatu. Aku di sini kok."]))

# ── Mood Donut ────────────────────────────────────────────────────────────────
def mood_donut_html(emo_counts: dict) -> str:
    total    = sum(emo_counts.values()) or 1
    pcts     = [emo_counts.get(i, 0) / total for i in range(6)]
    has_data = any(v > 0 for v in emo_counts.values())
    dominant = max(emo_counts, key=emo_counts.get) if has_data else 1
    dom_pct  = int(pcts[dominant] * 100)
    colors   = [EMO_COLOR[i] for i in range(6)]
    emojis   = ['😢','😊','😍','😤','😰','😲']

    r, cx, cy  = 38, 50, 50
    paths, off = "", 0.0
    for i, pct in enumerate(pcts):
        if pct < 0.005:
            off += pct
            continue
        a1    = off * 360 - 90
        a2    = (off + pct) * 360 - 90
        large = 1 if pct > 0.5 else 0
        def pt(a, _r=r, _cx=cx, _cy=cy):
            rad = math.radians(a)
            return _cx + _r*math.cos(rad), _cy + _r*math.sin(rad)
        x1, y1 = pt(a1); x2, y2 = pt(a2)
        paths += f'<path d="M{cx},{cy}L{x1:.1f},{y1:.1f}A{r},{r},0,{large},1,{x2:.1f},{y2:.1f}Z" fill="{colors[i]}"/>'
        off += pct

    if not has_data:
        paths = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#bfdbfe"/>'

    svg = (
        f'<svg viewBox="0 0 100 100" width="90" height="90">'
        f'{paths}'
        f'<circle cx="{cx}" cy="{cy}" r="24" fill="white"/>'
        f'<text x="{cx}" y="{cy-4}" text-anchor="middle" font-size="13">{emojis[dominant]}</text>'
        f'<text x="{cx}" y="{cy+11}" text-anchor="middle" fill="#1d4ed8" font-size="9" font-weight="bold">{dom_pct}%</text>'
        f'</svg>'
    )

    rows = ""
    for i in range(6):
        if pcts[i] < 0.01: continue
        rows += (
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:{colors[i]};flex-shrink:0;"></div>'
            f'<span style="font-size:12px;color:#1e3a5f;flex:1;">{EMO_LABEL[i]}</span>'
            f'<span style="font-size:12px;color:#3b82f6;">{int(pcts[i]*100)}%</span>'
            f'</div>'
        )
    if not rows:
        rows = '<span style="font-size:12px;color:#3b82f6;">Mulai chat untuk analisis</span>'

    return (
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'{svg}'
        f'<div style="flex:1;">{rows}</div>'
        f'</div>'
    )

# ── CSS — Blue Theme, High Contrast ──────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Background biru muda bersih ── */
.stApp {
    background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 50%, #e0f2fe 100%) !important;
}

/* ── Semua teks default gelap ── */
.stApp, .stApp p, .stApp div, .stApp span, .stApp label {
    color: #1e3a5f !important;
}

/* Main container */
.main > div {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem 2rem;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Hero title ── */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #1d4ed8 !important;
    -webkit-text-fill-color: #1d4ed8 !important;
    margin-bottom: 0.25rem;
    line-height: 1.15;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #2563eb !important;
    margin-bottom: 1.5rem;
    font-weight: 500;
}

/* ── Section headings (h3 dari st.markdown) ── */
h1, h2, h3, h4 {
    color: #1d4ed8 !important;
    font-weight: 700 !important;
}

/* ── About text ── */
.about-text {
    color: #1e3a5f !important;
    font-size: 0.97rem;
    line-height: 1.65;
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border: 1px solid #bfdbfe;
}

/* ── Skill cards ── */
.skill-card {
    background: white !important;
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    border: 1.5px solid #bfdbfe;
    box-shadow: 0 2px 10px rgba(59,130,246,0.08);
    transition: all 0.2s ease;
    min-height: 110px;
}
.skill-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(59,130,246,0.15);
    border-color: #3b82f6;
}
.skill-card .icon {
    font-size: 28px;
    margin-bottom: 8px;
}
.skill-card .title {
    font-weight: 700;
    font-size: 0.95rem;
    color: #1d4ed8 !important;
    margin-bottom: 4px;
}
.skill-card .desc {
    font-size: 0.82rem;
    color: #2563eb !important;
}

/* ── Quick stats cards ── */
.stat-card {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    border: 1.5px solid #bfdbfe;
    box-shadow: 0 2px 8px rgba(59,130,246,0.07);
    margin-bottom: 12px;
    text-align: center;
}
.stat-label {
    font-size: 0.78rem;
    color: #2563eb !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #1d4ed8 !important;
}

/* ── Streamlit metric override ── */
[data-testid="stMetricValue"] {
    color: #1d4ed8 !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #2563eb !important;
    font-weight: 600 !important;
}

/* ── st.info box ── */
[data-testid="stAlert"] {
    background: #eff6ff !important;
    border: 1.5px solid #93c5fd !important;
    color: #1e3a5f !important;
    border-radius: 12px !important;
}
[data-testid="stAlert"] p {
    color: #1e3a5f !important;
}

/* ── Chat bubbles ── */
.chat-bubble-user {
    background: #dbeafe;
    border: 1px solid #93c5fd;
    padding: 10px 14px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0;
    max-width: 88%;
    margin-left: auto;
    color: #1e3a5f !important;
    font-size: 13px;
    line-height: 1.5;
}
.chat-bubble-bot {
    background: white;
    border: 1.5px solid #bfdbfe;
    padding: 10px 14px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 0;
    max-width: 88%;
    color: #1e3a5f !important;
    font-size: 13px;
    line-height: 1.5;
    box-shadow: 0 1px 4px rgba(59,130,246,0.07);
}

/* ── Recent chat preview bubbles ── */
.preview-bubble-user {
    background: #dbeafe;
    padding: 8px 14px;
    border-radius: 12px;
    margin: 5px 0;
    color: #1e3a5f !important;
    font-size: 13px;
    border-left: 3px solid #3b82f6;
}
.preview-bubble-bot {
    background: white;
    padding: 8px 14px;
    border-radius: 12px;
    margin: 5px 0;
    color: #1e3a5f !important;
    font-size: 13px;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #93c5fd;
}
.bubble-label {
    font-weight: 700;
    font-size: 11px;
    color: #1d4ed8 !important;
    margin-bottom: 2px;
}

/* ── Open Full Chat button ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 14px rgba(29,78,216,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.4) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: white !important;
    color: #1e3a5f !important;
    border: 1.5px solid #93c5fd !important;
    border-radius: 24px !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #60a5fa !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: white !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 16px !important;
    padding: 4px 8px !important;
    margin: 4px 0 !important;
}
[data-testid="stChatMessage"] p {
    color: #1e3a5f !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%) !important;
    border-right: 1.5px solid #bfdbfe !important;
}
section[data-testid="stSidebar"] * {
    color: #1e3a5f !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1d4ed8 !important;
}
section[data-testid="stSidebar"] .stCaption {
    color: #3b82f6 !important;
}

/* ── Divider ── */
hr {
    border-color: #bfdbfe !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #2563eb !important;
}
</style>
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
    for k, v in {
        'messages':   [{'role':'bot','content':'Halo! 👋 Aku PersonaTalk, teman curhat kamu.\n\nCerita apa aja, aku dengerin.'}],
        'emotion':    1, 'confidence': 0.5, 'mbti': None, 'mbti_texts': [],
        'mode':       'curhat', 'q_idx': -1, 'q_resp': [],
        'last_bot':   [], 'emo_counts': {i:0 for i in range(6)},
        '_provider':  None, '_ai_err': None, 'chat_open': True,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    emo_id = st.session_state.emotion

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:20px 0 10px;">'
            '<div style="font-size:52px;margin-bottom:6px;">🐼</div>'
            '<div style="font-weight:800;font-size:20px;color:#1d4ed8;">PersonaTalk</div>'
            '<div style="font-size:12px;color:#2563eb;font-weight:500;margin-top:4px;">AI Mental Wellness Companion</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("### 📊 Mood Analytics")
        st.markdown(mood_donut_html(st.session_state.emo_counts), unsafe_allow_html=True)

        st.markdown("---")
        if st.session_state.mbti:
            name, _ = MBTI_DESC.get(st.session_state.mbti, ("",""))
            st.markdown(
                f'<div style="background:white;border:1.5px solid #bfdbfe;border-radius:10px;padding:10px 14px;">'
                f'<div style="font-size:11px;color:#2563eb;font-weight:700;margin-bottom:4px;">🧬 MBTI TYPE</div>'
                f'<div style="font-size:22px;font-weight:800;color:#1d4ed8;">{st.session_state.mbti}</div>'
                f'<div style="font-size:11px;color:#3b82f6;">{name}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:white;border:1.5px solid #bfdbfe;border-radius:10px;padding:10px 14px;">'
                '<div style="font-size:11px;color:#2563eb;font-weight:700;margin-bottom:4px;">🧬 MBTI TYPE</div>'
                '<div style="font-size:13px;color:#60a5fa;">Analyzing dari chat kamu...</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        provider = st.session_state.get('_provider')
        status_color = "#16a34a" if provider else "#3b82f6"
        status_text  = f"🟢 {provider.capitalize()}" if provider else "⚪ Ready"
        st.markdown(
            f'<div style="font-size:12px;color:{status_color};font-weight:600;">AI Status: {status_text}</div>',
            unsafe_allow_html=True,
        )

    # ── Main Content ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        # Hero
        st.markdown('<div class="hero-title">Hi, I\'m PersonaTalk 🐼</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Your AI friend who listens without judgment. 💙</div>', unsafe_allow_html=True)

        # About
        st.markdown("### About Me")
        st.markdown(
            '<div class="about-text">'
            'PersonaTalk adalah teman curhat AI yang genuine — bukan terapis, bukan bot kaku, '
            'tapi teman yang selalu siap dengerin. Apapun yang kamu rasain, senang, sedih, '
            'bingung, atau cuma butuh teman ngobrol, aku di sini.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Feature cards
        st.markdown("### What I Can Do")
        c1, c2, c3 = st.columns(3)
        features = [
            (c1, "🧠", "Emotion Detection", "Mendeteksi emosi dari teks kamu secara real-time"),
            (c2, "📊", "MBTI Analysis",     "Menganalisis tipe kepribadian dari cara kamu ngobrol"),
            (c3, "💬", "Natural Chat",      "Ngobrol santai kayak sama teman beneran"),
        ]
        for col, icon, title, desc in features:
            with col:
                st.markdown(
                    f'<div class="skill-card">'
                    f'<div class="icon">{icon}</div>'
                    f'<div class="title">{title}</div>'
                    f'<div class="desc">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Current mood
        st.markdown("### Current Mood")
        emo_label = EMO_LABEL.get(emo_id, 'Netral')
        emo_emoji = EMO_EMOJI.get(emo_id, '😊')
        conf_pct  = int(st.session_state.confidence * 100)
        st.markdown(
            f'<div style="background:#dbeafe;border:1.5px solid #93c5fd;border-radius:12px;'
            f'padding:12px 16px;display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:28px;">{emo_emoji}</span>'
            f'<div>'
            f'<div style="font-weight:700;font-size:1rem;color:#1d4ed8;">{emo_label} Detected</div>'
            f'<div style="font-size:0.82rem;color:#2563eb;">Confidence: {conf_pct}%</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### Quick Stats")
        total_user = len([m for m in st.session_state.messages if m['role'] == 'user'])
        mbti_val   = st.session_state.mbti if st.session_state.mbti else "Analyzing..."
        prov_val   = st.session_state['_provider'].capitalize() if st.session_state['_provider'] else "Ready"

        for label, value in [("💬 Total Chats", total_user), ("🧬 MBTI Status", mbti_val), ("⚡ AI Provider", prov_val)]:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-label">{label}</div>'
                f'<div class="stat-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Divider
    st.markdown("---")

    # Recent Chat Preview
    st.markdown("### 💬 Recent Chat")
    last_msgs = st.session_state.messages[-4:]
    for msg in last_msgs[-3:]:
        preview = msg["content"][:120] + ("..." if len(msg["content"]) > 120 else "")
        if msg['role'] == 'user':
            st.markdown(
                f'<div class="preview-bubble-user">'
                f'<div class="bubble-label">👤 Kamu</div>'
                f'{preview}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="preview-bubble-bot">'
                f'<div class="bubble-label">🐼 PersonaTalk</div>'
                f'{preview}</div>',
                unsafe_allow_html=True,
            )

    # Toggle button
    btn_label = "✕ Tutup Chat" if st.session_state.chat_open else "💬 Buka Full Chat"
    if st.button(btn_label, use_container_width=True):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

    # ── Full Chat ─────────────────────────────────────────────────────────────
    if st.session_state.chat_open:
        st.markdown("---")
        st.markdown("### 💬 Chat with PersonaTalk")

        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                with st.chat_message("user", avatar="👤"):
                    st.write(msg['content'])
            else:
                with st.chat_message("assistant", avatar=EMO_ICON.get(emo_id, '🐼')):
                    st.write(msg['content'])

        user_text = st.chat_input("Ketik pesan kamu di sini...")

        if user_text and user_text.strip():
            user_text = user_text.strip()
            st.session_state.messages.append({'role': 'user', 'content': user_text})

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

            st.session_state.last_bot.append(response)
            st.session_state.messages.append({'role': 'bot', 'content': response})
            st.rerun()


if __name__ == "__main__":
    main()
