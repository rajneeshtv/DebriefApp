import streamlit as st
import sqlite3
from datetime import datetime
import uuid
import os
from typing import List, Dict
# Optional ML imports (lazy loaded)
try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

# --- Database utilities ---
DB_PATH = "debrief_app.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    # Tasks, Steps, Participants, Assignments, Debriefs, StepSummaries
    c.executescript('''
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY, title TEXT, description TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS steps (
        id TEXT PRIMARY KEY, task_id TEXT, title TEXT, description TEXT, start_date TEXT, end_date TEXT
    );
    CREATE TABLE IF NOT EXISTS participants (
        id TEXT PRIMARY KEY, name TEXT, email TEXT
    );
    CREATE TABLE IF NOT EXISTS assignments (
        id TEXT PRIMARY KEY, step_id TEXT, participant_id TEXT
    );
    CREATE TABLE IF NOT EXISTS debriefs (
        id TEXT PRIMARY KEY, step_id TEXT, participant_id TEXT, text_response TEXT, audio_path TEXT, sentiment TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS step_summaries (
        step_id TEXT PRIMARY KEY, summary TEXT, compiled_lessons TEXT, negative_actions TEXT, updated_at TEXT
    );
    ''')
    conn.commit()
    return conn

conn = init_db()

# --- Model wrappers (lazy load + fallback) ---

summarizer = None
sentiment_analyzer = None
embedder = None

def load_models():
    global summarizer, sentiment_analyzer, embedder
    if pipeline:
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            summarizer = None
        try:
            sentiment_analyzer = pipeline("sentiment-analysis")
        except Exception:
            sentiment_analyzer = None
    if SentenceTransformer:
        try:
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            embedder = None

# Call model loading in background when app starts (non-blocking in real app)
load_models()

# Transcription placeholder
from pydub import AudioSegment
import speech_recognition as sr

def transcribe_audio(audio_file):
    try:
        sound = AudioSegment.from_file(audio_file)
        wav_path = "temp.wav"
        sound.export(wav_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception:
        return ""
    
#def transcribe_audio(file_path: str) -> str:
#    Try whisper local transcription if available, otherwise return placeholder.
#    try:
#        import whisper
#        model = whisper.load_model("base")
#        res = model.transcribe(file_path)
#        return res.get("text", "")
#    except Exception:
#       # fallback — in prototype return empty string
#        return ""

# Summarize helper
def summarize_text(text: str) -> str:
    if summarizer:
        try:
            out = summarizer(text, max_length=120, min_length=30)
            return out[0]['summary_text']
        except Exception:
            pass
    # fallback simple heuristic
    return text[:400] + ("..." if len(text) > 400 else "")

# Sentiment helper
def get_sentiment(text: str) -> str:
    if sentiment_analyzer:
        try:
            out = sentiment_analyzer(text[:512])
            return out[0]['label']
        except Exception:
            pass
    # simple fallback heuristics
    low = ["not", "no", "never", "unable", "fail"]
    if any(w in text.lower() for w in low):
        return "NEGATIVE"
    return "NEUTRAL"

# Semantic similarity helper
def semantic_similarity(a: str, b: str) -> float:
    if embedder and util:
        try:
            v1 = embedder.encode(a, convert_to_tensor=True)
            v2 = embedder.encode(b, convert_to_tensor=True)
            score = util.pytorch_cos_sim(v1, v2).item()
            return float(score)
        except Exception:
            pass
    # fallback naive ratio of common words
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

# --- Application logic: CRUD operations ---

def create_task(title: str, description: str) -> str:
    id_ = str(uuid.uuid4())
    conn.execute("INSERT INTO tasks (id, title, description, created_at) VALUES (?, ?, ?, ?)",
                 (id_, title, description, datetime.utcnow().isoformat()))
    conn.commit()
    return id_

def create_step(task_id: str, title: str, description: str, start: str, end: str) -> str:
    id_ = str(uuid.uuid4())
    conn.execute("INSERT INTO steps (id, task_id, title, description, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?)",
                 (id_, task_id, title, description, start, end))
    conn.commit()
    return id_

def create_participant(name: str, email: str) -> str:
    id_ = str(uuid.uuid4())
    conn.execute("INSERT INTO participants (id, name, email) VALUES (?, ?, ?)", (id_, name, email))
    conn.commit()
    return id_

def assign_participant(step_id: str, participant_id: str) -> str:
    id_ = str(uuid.uuid4())
    conn.execute("INSERT INTO assignments (id, step_id, participant_id) VALUES (?, ?, ?)", (id_, step_id, participant_id))
    conn.commit()
    return id_

# Save debrief
def save_debrief(step_id: str, participant_id: str, text_response: str, audio_path: str, sentiment: str):
    id_ = str(uuid.uuid4())
    conn.execute("INSERT INTO debriefs (id, step_id, participant_id, text_response, audio_path, sentiment, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (id_, step_id, participant_id, text_response, audio_path, sentiment, datetime.utcnow().isoformat()))
    conn.commit()
    return id_

# Aggregate step
def aggregate_step(step_id: str) -> Dict:
    c = conn.cursor()
    rows = c.execute("SELECT text_response, sentiment FROM debriefs WHERE step_id = ?", (step_id,)).fetchall()
    texts = [r['text_response'] or '' for r in rows]
    sentiments = [r['sentiment'] for r in rows]
    aggregated = "\n\n".join(texts)
    summary = summarize_text(aggregated) if aggregated else ""
    # extract lessons = for prototype, we take top sentences containing 'learn' or 'issue'
    lessons = []
    for t in texts:
        sentences = t.split('.')
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if 'learn' in s.lower() or 'issue' in s.lower() or 'should' in s.lower() or 'recommend' in s.lower():
                lessons.append(s)
    negative_actions = []
    # proposed actions: for each NEGATIVE sentiment, propose simple template action
    for idx, t in enumerate(texts):
        if sentiments and sentiments[idx] == 'NEGATIVE':
            negative_actions.append(f"Investigate response from participant #{idx+1}; consider coaching and follow-up.")
    # save step summary
    conn.execute("REPLACE INTO step_summaries (step_id, summary, compiled_lessons, negative_actions, updated_at) VALUES (?, ?, ?, ?, ?)",
                 (step_id, summary, '\n'.join(lessons), '\n'.join(negative_actions), datetime.utcnow().isoformat()))
    conn.commit()
    return {"summary": summary, "lessons": lessons, "negative_actions": negative_actions}

# --- Streamlit UI ---

st.set_page_config(page_title="Debrief & Lessons Prototype", layout="wide")
st.title("Debrief & Lessons-Learned Prototype")

mode = st.sidebar.selectbox("Mode", ["Admin", "Participant"])

if mode == 'Admin':
    st.header("Admin Panel")
    tab = st.tabs(["Create Task/Step", "Assignments", "Step Summaries"])

    with tab[0]:
        st.subheader("Create Task")
        t_title = st.text_input("Task title")
        t_desc = st.text_area("Task description")
        if st.button("Create Task"):
            tid = create_task(t_title, t_desc)
            st.success(f"Task created: {tid}")

        st.markdown("---")
        st.subheader("Create Step for Task")
        tasks = conn.execute("SELECT id, title FROM tasks").fetchall()
        task_map = {t['title']: t['id'] for t in tasks}
        if tasks:
            sel = st.selectbox("Select task", list(task_map.keys()))
            s_title = st.text_input("Step title")
            s_desc = st.text_area("Step description")
            s_start = st.date_input("Start date")
            s_end = st.date_input("End date")
            if st.button("Create Step"):
                sid = create_step(task_map[sel], s_title, s_desc, s_start.isoformat(), s_end.isoformat())
                st.success("Step created")
        else:
            st.info("Create a task first")

    with tab[1]:
        st.subheader("Participants & Assignments")
        name = st.text_input("Participant name")
        email = st.text_input("Participant email")
        if st.button("Create Participant"):
            pid = create_participant(name, email)
            st.success("Participant created")
        st.markdown("---")
        steps = conn.execute("SELECT id, title FROM steps").fetchall()
        participants = conn.execute("SELECT id, name FROM participants").fetchall()
        if steps and participants:
            step_map = {s['title']: s['id'] for s in steps}
            part_map = {p['name']: p['id'] for p in participants}
            sel_step = st.selectbox("Select Step", list(step_map.keys()))
            sel_part = st.selectbox("Select Participant", list(part_map.keys()))
            if st.button("Assign Participant"):
                assign_participant(step_map[sel_step], part_map[sel_part])
                st.success("Assigned")
                # generate simple link token
                link_token = str(uuid.uuid4())
                st.write("Participant link (prototype token):")
                st.code(f"/participant_submit?step={step_map[sel_step]}&participant={part_map[sel_part]}&token={link_token}")
        else:
            st.info("Create steps and participants to assign")

    with tab[2]:
        st.subheader("Step Summaries")
        steps = conn.execute("SELECT id, title FROM steps").fetchall()
        if steps:
            sel = st.selectbox("Choose Step", [s['title'] for s in steps])
            sid = [s['id'] for s in steps if s['title'] == sel][0]
            if st.button("Aggregate Step Now"):
                res = aggregate_step(sid)
                st.success("Aggregated")
                st.subheader("AI Summary")
                st.write(res['summary'])
                st.subheader("Lessons Learnt")
                for l in res['lessons']:
                    st.write(f"- {l}")
                st.subheader("Proposed Actions for Negative Lessons")
                for a in res['negative_actions']:
                    st.write(f"- {a}")
        else:
            st.info("No steps available")

else:
    st.header("Participant Submission")
    st.write("If you have a direct link, open it with the query params: step, participant (prototype). Otherwise choose below.")
    steps = conn.execute("SELECT id, title FROM steps").fetchall()
    participants = conn.execute("SELECT id, name FROM participants").fetchall()
    step_map = {s['title']: s['id'] for s in steps}
    part_map = {p['name']: p['id'] for p in participants}
    if not steps or not participants:
        st.info("No tasks/participants exist yet. Ask admin to create and assign.")
    else:
        sel_step = st.selectbox("Step", list(step_map.keys()))
        sel_part = st.selectbox("Participant", list(part_map.keys()))
        st.text_area("Describe the activity you performed (debrief)", key="debrief_text")
        audio = st.file_uploader("Upload audio (wav, mp3)", type=["wav","mp3","m4a"])
        if st.button("Submit Debrief"):
            text_resp = st.session_state.get('debrief_text','')
            audio_path = ''
            if audio is not None:
                save_dir = 'uploads'
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f"{uuid.uuid4()}_{audio.name}")
                with open(fname, 'wb') as f:
                    f.write(audio.getbuffer())
                audio_path = fname
                # try transcription
                #trans = transcribe_audio(r"uploads\4dff9a69-5f74-480e-a418-28d1ebe561bc_Sejal_2_wav.wav")
                trans = transcribe_audio(audio_path)
                if trans:
                    text_resp = (text_resp + '\n' + trans).strip()
            sentiment = get_sentiment(text_resp)
            save_debrief(step_map[sel_step], part_map[sel_part], text_resp, audio_path, sentiment)
            st.success("Debrief submitted")

st.markdown("---")
st.caption("Prototype app — replace local models with cloud services as needed for production.")
