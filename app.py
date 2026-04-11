"""
🐼 PersonaTalk — Streamlit App
Claude as PRIMARY AI, Gemini as FALLBACK
"""

import streamlit as st
import numpy as np
import re
import os
import random
import joblib
from datetime import datetime

# NLTK
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',  quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Anthropic
try:
    import anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

# Gemini
try:
    import google.generativeai as genai
    _GENAI_OK = True
except ImportError:
    _GENAI_OK = False

# Hugging Face Hub
from huggingface_hub import hf_hub_download

# ============================================================================
# KONFIGURASI
# ============================================================================

HF_REPO_ID     = "Jooou139/personatalk"
ANTHROPIC_KEY  = st.secrets.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
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
        emo_model       = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="emo_model.pkl",       token=token))
        emo_vectorizer  = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="emo_vectorizer.pkl",  token=token))
        mbti_model      = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="mbti_model.pkl",      token=token))
        mbti_vectorizer = joblib.load(hf_hub_download(repo_id=HF_REPO_ID, filename="mbti_vectorizer.pkl", token=token))
        return emo_model, emo_vectorizer, mbti_model, mbti_vectorizer
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        st.stop()

# ============================================================================
# EMOTION DETECTION
# ============================================================================

EMOTION_NAMES_ID = {0:'Sedih', 1:'Bahagia', 2:'Cinta', 3:'Marah', 4:'Cemas', 5:'Terkejut'}
EMOTION_ICONS    = {0:'🦊', 1:'🐱', 2:'🐰', 3:'🐯', 4:'🐭', 5:'🐨'}
EMOTION_EMOJI    = {0:'😔', 1:'😊', 2:'❤️', 3:'😠', 4:'😨', 5:'😲'}

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
    t = text.lower()
    scores = {}
    for word, emo in EMOTION_LEXICON.items():
        if word in t:
            scores[emo] = scores.get(emo, 0) + 1
    if scores:
        return max(scores, key=scores.get), 0.85
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
# SYSTEM PROMPT — RINGKAS & EFEKTIF
# ============================================================================

SYSTEM_PROMPT = """Kamu adalah PersonaTalk — teman curhat digital yang beneran peduli, bukan therapist atau bot.

KARAKTER:
Teman dekat yang dengerin dulu sebelum ngasih saran. Natural, hangat, kadang playful, tidak pernah judgmental. Bukan AI, bukan konselor — teman.

WAJIB BACA KONTEKS:
- Selalu baca riwayat percakapan sebelum balas
- Kalau user sudah jawab pertanyaanmu, JANGAN tanya hal yang sama lagi
- Lanjutkan dari informasi yang sudah mereka kasih — acknowledge, lalu gali lebih dalam atau kasih respons yang relevan
- Contoh SALAH: user bilang "ketemu di cafe" tapi bot masih nanya "kamu tau dari mana?"
- Contoh BENAR: user bilang "ketemu di cafe" → bot respond soal perasaan waktu ketemu langsung di sana

CARA BALAS:
- Bahasa Indonesia sehari-hari, santai, boleh campur english kalau natural
- 2-4 kalimat maksimal — pendek, padat, ngena
- SELALU validasi perasaan dulu sebelum kasih insight atau solusi
- Akhiri dengan 1 pertanyaan terbuka yang BERBEDA dari pertanyaan sebelumnya
- Jangan bullet point, jangan formal, jangan template korporat
- Vary pembuka setiap response: "Duh", "Ooh", "Ya Allah", "Aduh", "Hmm", "Serius?", "Oof", "Wah", "Astaga"
- JANGAN ulangi kalimat atau pertanyaan yang sudah pernah kamu tulis sebelumnya
- Jangan pernah sebut diri sebagai AI, bot, atau assistant

PANDUAN EMOSI:
- Sedih/galau → akui rasa sakitnya secara spesifik, baru tanya yang personal
- Marah → validasi bahwa kemarahannya masuk akal, tanya sumber utamanya
- Cemas/overthinking → normalize, bedain antara overthinking vs ada trigger nyata
- Senang/excited → ikut excited dengan tulus, gali kenapa happy
- Cinta/crush → playful tapi genuine, tanya progress atau perasaannya lebih dalam
- Capek/burnout → empati dulu, tanya ini capek fisik atau emosional

SOLUSI:
Kalau user udah cerita cukup banyak (3+ exchange), boleh kasih 1-2 saran konkret yang praktis sambil tetap tanya apa yang mereka butuhkan.

JANGAN:
- Diagnosis medis atau psikologis
- Tanya pertanyaan yang sudah dijawab user
- Ulangi kalimat yang sama persis dari response sebelumnya
- Template corporate ("Saya mengerti perasaan Anda")
- Lebih dari 4 kalimat
- Bullet point atau list"""

# ============================================================================
# DUPLICATE DETECTION
# ============================================================================

def is_duplicate_response(new_response: str, last_bot_responses: list, threshold: float = 0.55) -> bool:
    if not last_bot_responses or not new_response:
        return False
    new_clean = new_response.strip().lower()
    for old_response in last_bot_responses[-3:]:
        old_clean = old_response.strip().lower()
        if new_clean == old_clean:
            return True
        new_words = set(new_clean.split())
        old_words = set(old_clean.split())
        jaccard = len(new_words & old_words) / max(len(new_words | old_words), 1)
        if jaccard > threshold:
            return True
        new_first = new_clean.split('.')[0].strip()
        old_first = old_clean.split('.')[0].strip()
        if new_first and old_first and len(new_first) > 10 and new_first == old_first:
            return True
    return False

# ============================================================================
# AI RESPONSE GENERATOR — CLAUDE PRIMARY (MULTI-TURN), GEMINI FALLBACK
# ============================================================================

def build_claude_messages(history: list, emotion_label, last_bot_responses: list) -> list:
    """
    Build proper multi-turn messages for Claude API.
    Claude requires alternating user/assistant roles — we map:
        bot   → assistant
        user  → user
    We also inject a system-level note about emotion + no-repeat into
    the FIRST user turn so it doesn't break the alternating pattern.
    """
    emotion_name = EMOTION_NAMES_ID.get(emotion_label, "Netral") if isinstance(emotion_label, int) else str(emotion_label)

    # Build recent history (exclude the very last user message — that's sent separately)
    # history[-1] is the current user message already appended before calling this
    recent = history[-13:-1] if len(history) > 1 else []

    # Collapse consecutive same-role messages (safety: Claude API rejects them)
    collapsed = []
    for msg in recent:
        api_role = "assistant" if msg['role'] == 'bot' else "user"
        if collapsed and collapsed[-1]['role'] == api_role:
            collapsed[-1]['content'] += "\n" + msg['content']
        else:
            collapsed.append({'role': api_role, 'content': msg['content']})

    # Build no-repeat hint
    no_repeat = ""
    if last_bot_responses:
        no_repeat = "\n\n[INTERNAL NOTE — jangan ulangi pola ini:\n"
        for r in last_bot_responses[-2:]:
            no_repeat += f"- {r[:80]}...\n"
        no_repeat += "]"

    # Inject emotion note into first user turn (or create one if history is empty)
    emotion_note = f"[Emosi user saat ini terdeteksi: {emotion_name}]{no_repeat}"

    if collapsed and collapsed[0]['role'] == 'user':
        collapsed[0]['content'] = emotion_note + "\n\n" + collapsed[0]['content']
    else:
        # Insert a synthetic user turn at the beginning if first msg is assistant
        collapsed.insert(0, {'role': 'user', 'content': emotion_note})
        # Then we need an assistant ack to keep alternating — skip if collapsed[1] is already assistant
        if len(collapsed) > 1 and collapsed[1]['role'] != 'assistant':
            collapsed.insert(1, {'role': 'assistant', 'content': 'Oke, aku dengerin.'})

    # Ensure we end with a user turn (the current message)
    current_user_msg = history[-1]['content'] if history and history[-1]['role'] == 'user' else ""
    if current_user_msg:
        if collapsed and collapsed[-1]['role'] == 'user':
            collapsed[-1]['content'] += "\n" + current_user_msg
        else:
            collapsed.append({'role': 'user', 'content': current_user_msg})

    # Final safety: must start with 'user'
    if collapsed and collapsed[0]['role'] != 'user':
        collapsed.insert(0, {'role': 'user', 'content': emotion_note})

    return collapsed


def clean_response(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'##\s+', '', text)
    text = re.sub(r'\[INTERNAL NOTE.*?\]', '', text, flags=re.DOTALL)
    return text.strip()


def generate_ai_response(user_text: str, emotion_label, history: list, last_bot_responses: list = None) -> str:
    last_bot_responses = last_bot_responses or []

    # ── CLAUDE PRIMARY — proper multi-turn messages ──────────────────────────
    if _ANTHROPIC_OK and ANTHROPIC_KEY:
        try:
            claude_messages = build_claude_messages(history, emotion_label, last_bot_responses)
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=350,
                temperature=0.85,
                system=SYSTEM_PROMPT,
                messages=claude_messages,
            )
            result = clean_response(msg.content[0].text)
            if result and len(result) > 10:
                if not is_duplicate_response(result, last_bot_responses):
                    st.session_state['_last_ai_error'] = None
                    st.session_state['_ai_provider']   = 'claude'
                    return result
                # If duplicate, ask Claude to retry with explicit instruction
                retry_messages = claude_messages + [
                    {'role': 'assistant', 'content': result},
                    {'role': 'user', 'content': 'Response itu terlalu mirip dengan yang sebelumnya. Coba dengan pembuka dan pertanyaan yang berbeda sama sekali.'}
                ]
                retry_msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=350,
                    temperature=0.95,
                    system=SYSTEM_PROMPT,
                    messages=retry_messages,
                )
                retry_result = clean_response(retry_msg.content[0].text)
                if retry_result and len(retry_result) > 10:
                    st.session_state['_last_ai_error'] = None
                    st.session_state['_ai_provider']   = 'claude'
                    return retry_result
        except Exception as e:
            st.session_state['_last_ai_error'] = f"Claude error: {str(e)[:120]}"

    # ── GEMINI FALLBACK ──────────────────────────────────────────────────────
    if _GENAI_OK and GEMINI_API_KEY:
        # Build context string for Gemini (it doesn't support multi-turn the same way)
        emotion_name = EMOTION_NAMES_ID.get(emotion_label, "Netral") if isinstance(emotion_label, int) else str(emotion_label)
        recent = history[-10:-1] if len(history) > 1 else []
        ctx_lines = [("User" if m['role']=='user' else "PersonaTalk") + ": " + m['content'] for m in recent]
        no_repeat_hint = ""
        if last_bot_responses:
            no_repeat_hint = "\nJANGAN ulangi: " + " | ".join(r[:60] for r in last_bot_responses[-2:])

        gemini_prompt = (
            f"Emosi user: {emotion_name}\n\n"
            f"Riwayat:\n" + "\n".join(ctx_lines) +
            f"\n\nPesan terakhir user: \"{user_text}\"\n{no_repeat_hint}\n\n"
            f"Balas sebagai PersonaTalk. 2-4 kalimat, natural, variasikan pembuka."
        )
        try:
            model    = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                content=f"{SYSTEM_PROMPT}\n\n{gemini_prompt}",
                generation_config=genai.types.GenerationConfig(
                    temperature=0.85,
                    top_p=0.95,
                    max_output_tokens=350,
                    top_k=40,
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ],
            )
            result = clean_response(response.text)
            if result and len(result) > 10:
                if not is_duplicate_response(result, last_bot_responses):
                    st.session_state['_last_ai_error'] = None
                    st.session_state['_ai_provider']   = 'gemini'
                    return result
        except Exception as e:
            prev = st.session_state.get('_last_ai_error', '')
            st.session_state['_last_ai_error'] = (prev + " | " if prev else "") + f"Gemini: {str(e)[:100]}"

    return None

# ============================================================================
# SMART FALLBACK (NO API)
# ============================================================================

def smart_fallback_response(text: str, emotion: int, history: list = None) -> str:
    t  = text.lower().strip()
    fc = t
    if history:
        user_msgs = [m['content'].lower() for m in history[-5:] if m['role'] == 'user']
        fc = ' '.join(user_msgs)

    # Putus / selingkuh
    if any(w in fc for w in ['putus','selingkuh','ditinggal','diputus','diselingkuhin','dikhianatin']):
        if any(w in fc for w in ['2 tahun','bertahun','setahun','lama']):
            return random.choice([
                "Duh, sekian tahun tuh banyak banget yang diinvest bareng seseorang. Sekarang rasanya campur aduk antara marah sama sedih ya? Kamu lagi ada support atau sendirian nih?",
                "Ya ampun, waktu selama itu bareng seseorang terus tiba-tiba putus — itu berat banget. Kamu masih sering kepikiran momen-momen spesifiknya nggak?",
                "Ouch, investasi waktu selama itu terus berakhir kayak gini. Gimana kondisi kamu sekarang — masih kontak sama dia atau udah cut off?",
            ])
        elif 'selingkuh' in fc or 'diselingkuhin' in fc:
            return random.choice([
                "Ya Allah, diselingkuhin itu rasa sakitnya double — patah hati sekaligus merasa dikhianatin. Kamu tau dari mana sampai ketahuan?",
                "Oof, dikhianatin sama orang yang kamu percaya itu beda levelnya. Dia masih denial atau udah ngaku?",
                "Aduh, ini menyakitkan banget. Kapan kamu tau dan gimana perasaanmu waktu pertama kali ketahuan?",
            ])
        else:
            return random.choice([
                "Duh, putus tuh emang berasa kayak ada yang hilang tiba-tiba. Ini baru terjadi atau udah beberapa hari lalu?",
                "Aduh, breakup tuh berat. Sekarang kamu gimana — ada teman yang tahu situasinya atau masih nahan sendiri?",
                "Ya Allah, ini pasti nggak gampang. Siapa sih yang biasanya kamu ceritain kalau lagi susah kayak gini?",
            ])

    # Bingung / lost
    if any(w in fc for w in ['bingung','harus apa','mau ngapain','ga tahu','nggak tahu','blank','lost']):
        return random.choice([
            "Hmm, ngerasa blank kayak gini biasanya karena terlalu banyak yang dipikirin sekaligus. Ini bingung soal satu hal spesifik atau kayak semua hal sekaligus?",
            "Ngerasa lost tuh beda sama nggak tahu jawabannya — lebih ke nggak tahu harus mulai dari mana. Kamu lagi overwhelmed sama apa?",
            "Duh, blank kayak gini emang bikin capek sendiri. Coba cerita — hal apa yang paling bikin kamu stuck sekarang?",
        ])

    # Rindu / kangen
    if any(w in fc for w in ['rindu','kangen','missing','kehilangan']):
        return random.choice([
            "Rindu yang kayak gini biasanya tanda ada hal penting yang kamu rasa hilang. Kamu kangen orangnya, momennya, atau keduanya?",
            "Ooh, kangen yang dalam kayak gini nggak enak banget. Udah berapa lama nggak ketemu atau nggak ada kabar?",
            "Aduh, kangen orang itu bikin dada rasanya berat ya. Masih ada komunikasi atau udah benar-benar jauh?",
        ])

    # Happy / excited
    if emotion == 1 or any(w in fc for w in ['happy','senang','excited','yay','bagus banget','berhasil','lulus']):
        return random.choice([
            "Wah, ada yang bagus nih! Ini senang karena apa — cerita dong, penasaran!",
            "Ooh, kedengarannya ada yang bikin kamu glow up hari ini! Apaan sih yang terjadi?",
            "Duh, seneng banget dengernya! Gimana awalnya bisa terjadi?",
            "Ya ampun, ada yang happy banget nih! Jabarkan dong — dari awal gimana ceritanya?",
        ])

    # Marah
    if emotion == 3 or any(w in fc for w in ['marah','kesal','benci','emosi','jengkel','ngeselin']):
        if any(w in fc for w in ['dibohongin','ditipu','bohong']):
            return random.choice([
                "Oof, dibohongin itu bikin marah plus kecewa sekaligus — dua rasa yang berat banget. Udah lama dia gini atau baru ketahuan?",
                "Duh, ditipu sama orang yang harusnya bisa dipercaya itu nyesek. Ini pertama kali atau udah pola dari dulu?",
            ])
        return random.choice([
            "Kemarahan kayak gini valid banget. Ini marah sama orangnya langsung atau lebih ke situasinya yang frustrasi?",
            "Oof, emosi banget nih. Apaan sih yang paling bikin gemas dari situasi ini?",
            "Duh, rasanya pengen ngeluarin semua ya. Ini udah numpuk lama atau baru meledak hari ini?",
        ])

    # Cemas / anxiety
    if emotion == 4 or any(w in fc for w in ['cemas','khawatir','panik','gelisah','overthinking','was-was','deg degan','stress']):
        return random.choice([
            "Ngerti banget, kekhawatiran kayak gini bikin kepala penuh dan susah fokus. Ini overthinking atau ada hal konkret yang bikin kamu takut?",
            "Gelisah yang kayak gini emang nggak enak. Udah berapa lama ngerasa kayak gini dan ini soal apa?",
            "Cemas yang dalam kayak gini biasanya ada trigger-nya. Kamu lagi khawatir soal orang lain, situasi, atau sesuatu soal dirimu sendiri?",
            "Hmm, stress kayak gini bikin badan ikutan tegang. Ada yang bisa kamu kontrol dari situasi ini atau rasanya semuanya di luar kendali?",
        ])

    # Kaget / shock
    if emotion == 5 or any(w in fc for w in ['kaget','shock','terkejut','astaga','nggak nyangka']):
        return random.choice([
            "Serius?? Ini unexpected banget! Cerita dong lebih — apaan yang bikin kamu syok?",
            "Astaga, nggak nyangka ya! Gimana ceritanya sampai bisa kayak gitu?",
            "Ya ampun, kaget juga dengernya! Ini kejadiannya tiba-tiba atau ada tanda-tanda sebelumnya?",
        ])

    # Crush / naksir
    if any(w in fc for w in ['suka','naksir','gebetan','pdkt','jatuh cinta','crush','deg degan sama']):
        return random.choice([
            "Ooh, ada yang spesial nih kayaknya! Dia udah tahu perasaan kamu atau masih phase investigate?",
            "Wah, ada yang bikin deg-degan nih! Udah berapa lama naksir dan gimana interaksi kalian sejauh ini?",
            "Duh, ada crush! Ini teman lama atau orang baru yang baru kenal?",
        ])

    # Capek / burnout
    if any(w in fc for w in ['capek','lelah','exhausted','burnout','tired','tepar','ngos-ngosan']):
        return random.choice([
            "Capek yang kayak gini beda ya, bukan cuman fisik aja. Ini lebih ke exhausted secara emosional atau emang kelelahan dari semua hal sekaligus?",
            "Aduh, lelah yang dalam kayak gini biasanya tanda udah terlalu lama nahan banyak hal. Kamu kapan terakhir beneran istirahat?",
            "Burnout itu nyata dan berat. Ini capek dari kerjaan, hubungan, atau kehidupan secara general?",
        ])

    # Default by emotion
    defaults = {
        0: ["Duh, kedengarannya berat banget. Mau cerita lebih? Aku dengerin serius nih.",
            "Aduh, ada yang lagi berat dipikul ya. Dari mana mau mulai ceritanya?",
            "Ya Allah, gimana perasaan kamu sekarang — lagi sendirian atau ada yang temani?"],
        1: ["Wah, ada yang bikin hari kamu bagus! Cerita dong.",
            "Ooh, kedengarannya ada yang positif! Apaan nih?"],
        2: ["Ooh, ada yang bikin kamu gini! Cerita soal dia dong.",
            "Duh, kayaknya ada yang spesial. Gimana ceritanya?"],
        3: ["Oof, kemarahan itu valid. Apaan yang paling bikin kesel?",
            "Duh, kesel banget nih. Cerita dari awal gimana?"],
        4: ["Hmm, cemas kayak gini nggak enak. Ada yang spesifik bikin khawatir?",
            "Gelisah yang kayak gini berat. Ini udah berapa lama?"],
        5: ["Wah, kaget banget! Cerita dong apaan yang terjadi.",
            "Astaga, unexpected banget! Gimana ceritanya?"],
    }
    opts = defaults.get(emotion, [
        "Hmm, ada apa nih yang lagi kamu pikirin? Cerita yuk.",
        "Duh, kedengarannya ada yang mau diceritain. Aku di sini kok.",
        "Ada apa nih? Cerita aja, nggak ada yang dihakimi di sini.",
    ])
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
    chars  = {0:'🦊', 1:'🐱', 2:'🐰', 3:'🐯', 4:'🐭', 5:'🐨'}
    labels = {0:'Sedih', 1:'Happy', 2:'Love', 3:'Marah', 4:'Cemas', 5:'Kaget'}
    char   = chars.get(emotion, '🐼')
    label  = labels.get(emotion, 'Normal')
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

    # Load models
    with st.spinner("⚡ Memuat PersonaTalk..."):
        emo_model, emo_vec, mbti_model, mbti_vec = load_models()

    # Header
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

    # Session state defaults
    defaults = {
        'messages':           [{'role':'bot','content':'Halo! Aku PersonaTalk 🐼\n\nAku siap dengerin cerita kamu. Mau curhat soal apa hari ini? 😊'}],
        'current_emotion':    0,
        'current_mbti':       None,
        'last_confidence':    0.5,
        'mbti_texts':         [],
        'mode':               '💬 Curhat',
        'q_index':            -1,
        'q_responses':        [],
        'last_bot_responses': [],
        '_ai_provider':       None,
        '_last_ai_error':     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Sidebar
    with st.sidebar:
        st.markdown("## 🧠 Panel Analisis")
        st.markdown("---")

        mode = st.radio("Mode Interaksi", ["💬 Curhat", "🧬 Analisis MBTI"], horizontal=True)
        if mode != st.session_state.mode:
            st.session_state.mode     = mode
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

        # AI status
        st.markdown("---")
        ai_err      = st.session_state.get('_last_ai_error')
        ai_provider = st.session_state.get('_ai_provider')
        if ai_err:
            st.markdown(f'<span style="color:#e74c3c;font-size:0.75rem;">⚠️ {ai_err}</span>', unsafe_allow_html=True)
        elif ai_provider == 'claude':
            st.markdown('<span style="color:#00ffc8;font-size:0.75rem;">🟢 Claude AI aktif (primary)</span>', unsafe_allow_html=True)
        elif ai_provider == 'gemini':
            st.markdown('<span style="color:#f39c12;font-size:0.75rem;">🟡 Gemini AI aktif (fallback)</span>', unsafe_allow_html=True)
        elif ANTHROPIC_KEY:
            st.markdown('<span style="color:#00ffc8;font-size:0.75rem;">🟢 Claude AI siap</span>', unsafe_allow_html=True)
        elif GEMINI_API_KEY:
            st.markdown('<span style="color:#f39c12;font-size:0.75rem;">🟡 Gemini AI siap (fallback only)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#e74c3c;font-size:0.75rem;">🔴 Tidak ada API key — cek Secrets</span>', unsafe_allow_html=True)

    # Chat display
    for msg in st.session_state.messages:
        is_user = msg['role'] == 'user'
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

    # Input form
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

            mbti_pred, mbti_conf = predict_mbti(user_text, mbti_model, mbti_vec, st.session_state.mbti_texts)
            if mbti_pred and mbti_conf > 0.3:
                st.session_state.current_mbti = mbti_pred

            # Try AI — regenerate up to 2x if duplicate
            response = None
            for attempt in range(2):
                candidate = generate_ai_response(
                    user_text,
                    emotion,
                    st.session_state.messages,
                    st.session_state.last_bot_responses
                )
                if candidate and not is_duplicate_response(candidate, st.session_state.last_bot_responses):
                    response = candidate
                    break

            # Hard fallback if AI fails or keeps duplicating
            if not response:
                response = smart_fallback_response(user_text, emotion, st.session_state.messages)
                # Make sure fallback itself is not a duplicate
                if is_duplicate_response(response, st.session_state.last_bot_responses):
                    # Force a completely different response from the other pool
                    fallback_pool = [
                        "Hmm, gimana kamu ngerasa sekarang setelah cerita ini?",
                        "Aduh, berat banget ya. Ada hal lain yang bikin situasi ini makin susah?",
                        "Ooh, aku dengerin. Kamu mau cerita lebih soal itu?",
                        "Duh, pasti butuh waktu buat proses ini semua. Kamu lagi butuh didengar atau mau cari solusi?",
                        "Ya ampun, itu nggak gampang sama sekali. Sekarang kamu lagi ada support dari siapa?",
                    ]
                    # Pick one that's not duplicate
                    for opt in random.sample(fallback_pool, len(fallback_pool)):
                        if not is_duplicate_response(opt, st.session_state.last_bot_responses):
                            response = opt
                            break
                    else:
                        response = fallback_pool[0]  # worst case, just use it

            st.session_state.last_bot_responses.append(response)
            if len(st.session_state.last_bot_responses) > 10:
                st.session_state.last_bot_responses.pop(0)

        # ── MODE ANALISIS MBTI ────────────────────────────────────────────────
        else:
            if st.session_state.q_index == -1:
                st.session_state.q_responses = []
                st.session_state.q_index     = 0
                q = MBTI_QUESTIONS[0]
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
                        mbti_type = analyze_mbti(st.session_state.q_responses)
                        st.session_state.current_mbti = mbti_type
                        st.session_state.q_index      = -1
                        result   = format_mbti_result(mbti_type)
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
