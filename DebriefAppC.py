import streamlit as st
import os
import sqlite3
from datetime import datetime
from typing import Optional

# Optional ML imports (lazy loaded)
try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Optional audio libs
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# DB paths
DB_PATH_V1 = "debrief_app1.db"   # keep original table intact
DB_PATH = "debrief_app_full.db"  # new DB for enhanced records

# --- Initialize Database (new enhanced table) ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS debriefs2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            summary TEXT,
            sentiment TEXT,
            lessons TEXT,
            audio_path TEXT,
            embedding BLOB,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- Cache Models ---
@st.cache_resource
def load_models():
    summarizer = None
    sentiment_analyzer = None
    embedder = None
    # Load only if transformers available
    if pipeline is not None:
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception:
            summarizer = None
        try:
            sentiment_analyzer = pipeline("sentiment-analysis")
        except Exception:
            sentiment_analyzer = None
    if SentenceTransformer is not None:
        try:
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            embedder = None
    return summarizer, sentiment_analyzer, embedder

summarizer, sentiment_analyzer, embedder = load_models()

# --- Helpers ---

def save_debrief_full(text: str, summary: str, sentiment: str, lessons: str,
                      audio_path: Optional[str], embedding) -> None:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb_blob = None
    try:
        emb_blob = embedding.tobytes() if embedding is not None else None
    except Exception:
        emb_blob = None
    c.execute(
        "INSERT INTO debriefs2 (text, summary, sentiment, lessons, audio_path, embedding, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (text, summary, sentiment, lessons, audio_path, emb_blob, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def fetch_debriefs_full():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, text, summary, sentiment, lessons, audio_path, timestamp FROM debriefs2 ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def transcribe_file(file_path: str) -> str:
    """
    Try best-effort transcription:
    - If pydub present, convert to wav; then use speech_recognition (Google) if available.
    - Otherwise return empty string.
    """
    if AudioSegment is None or sr is None:
        return ""
    try:
        # Convert to wav (temp)
        wav_path = file_path
        # if not wav, convert
        if not file_path.lower().endswith(".wav"):
            wav_path = file_path + ".wav"
            audio = AudioSegment.from_file(file_path)
            audio.export(wav_path, format="wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        # Use Google's free API (best-effort)
        text = recognizer.recognize_google(audio_data)
        return text
    except Exception:
        return ""

def extract_lesson_sentences(text: str):
    """
    Lightweight sentence extraction: detect sentences containing lesson-like keywords.
    Returns list of candidate sentences.
    """
    if not text:
        return []
    keywords = ["learn", "lesson", "issue", "problem", "recommend", "should", "need to", "improve", "suggest"]
    # naive sentence split
    sentences = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]
    candidates = []
    for s in sentences:
        low = s.lower()
        if any(k in low for k in keywords):
            candidates.append(s)
    return candidates

def summarize_lessons(candidates: list) -> str:
    """
    Turn candidate sentences into a compact lessons summary.
    If summarizer available, run summarizer; otherwise join sentences.
    """
    if not candidates:
        return ""
    joined = ". ".join(candidates)
    # If summarizer available and input fairly long, ask it to summarize as bullets
    if summarizer is not None:
        try:
            # Ensure input length reasonable for summarizer
            input_text = joined
            # call model with conservative lengths
            out = summarizer(input_text, max_length=80, min_length=10)
            return out[0].get("summary_text", "").strip()
        except Exception:
            pass
    # fallback: return joined candidates
    return joined

# --- Streamlit UI ---
st.set_page_config(page_title="Debrief App (Lessons Extraction)", layout="wide")
st.title("📋 Debrief — Lessons Learnt Extraction")

# reset/new debrief button (clears session state)
def reset_form():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

st.sidebar.markdown("## Controls")
if st.sidebar.button("New Debrief (Reset)"):
    reset_form()

menu = ["Add Debrief", "View Debriefs"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Add Debrief":
    st.header("Add a New Debrief")
    # Task/step/participant inputs can be added later; keep simple for now
    user_text = st.text_area("Enter debrief text (or leave empty and upload audio):", height=200)

    audio_file = st.file_uploader("Optionally upload audio (mp3/wav/m4a)", type=["mp3","wav","m4a"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Extract & Save"):
            # If no text but audio present, attempt transcription
            if not user_text.strip() and audio_file is None:
                st.warning("Provide text or upload an audio file.")
            else:
                # Save uploaded audio (if any) to uploads folder
                audio_path = None
                if audio_file is not None:
                    save_path = os.path.join("uploads", f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{audio_file.name}")
                    with open(save_path, "wb") as f:
                        f.write(audio_file.getbuffer())
                    audio_path = save_path
                    # if no text provided try to transcribe
                    if not user_text.strip():
                        st.info("No text provided — attempting transcription (best-effort).")
                        trans = transcribe_file(audio_path)
                        if trans:
                            user_text = trans
                            st.success("Transcription succeeded (best-effort).")
                        else:
                            st.warning("Transcription failed or not available on this environment.")

                if user_text.strip():
                    with st.spinner("Analyzing..."):
                        # Summary
                        summary = ""
                        if summarizer is not None:
                            try:
                                # guard max_length to avoid model warnings: set relative to input
                                summary = summarizer(user_text, max_length= min(120, max(16, len(user_text)//2)), min_length=10, do_sample=False)[0].get("summary_text","").strip()
                            except Exception:
                                summary = user_text[:400] + ("..." if len(user_text) > 400 else "")
                        else:
                            summary = user_text[:400] + ("..." if len(user_text) > 400 else "")

                        # Sentiment
                        sentiment = "NEUTRAL"
                        if sentiment_analyzer is not None:
                            try:
                                s_out = sentiment_analyzer(user_text[:512])
                                # normalize
                                sentiment_label = s_out[0].get("label","")
                                if "NEG" in sentiment_label.upper():
                                    sentiment = "NEGATIVE"
                                elif "POS" in sentiment_label.upper():
                                    sentiment = "POSITIVE"
                                else:
                                    sentiment = sentiment_label
                            except Exception:
                                sentiment = "NEUTRAL"
                        else:
                            # simple heuristic
                            if any(w in user_text.lower() for w in ["not", "no", "never", "unable", "fail", "problem", "issue"]):
                                sentiment = "NEGATIVE"
                            else:
                                sentiment = "NEUTRAL"

                        # Lessons extraction
                        candidates = extract_lesson_sentences(user_text)
                        lessons_summary = summarize_lessons(candidates) if candidates else ""
                        # fallback: if no explicit candidates, try to extract via summary text heuristics
                        if not lessons_summary and len(user_text.split()) > 30:
                            # try to extract 1-2 lesson-like sentences from summary text itself
                            candidates2 = extract_lesson_sentences(summary)
                            lessons_summary = summarize_lessons(candidates2) if candidates2 else ""

                        # Embedding (optional)
                        emb = None
                        if embedder is not None:
                            try:
                                emb = embedder.encode([user_text])[0]
                            except Exception:
                                emb = None

                        # Save to DB
                        save_debrief_full(user_text, summary, sentiment, lessons_summary, audio_path, emb)

                    st.success("Debrief analyzed and saved.")
                    st.write("**Summary:**", summary)
                    st.write("**Sentiment:**", sentiment)
                    if lessons_summary:
                        st.write("**Lessons Learnt (extracted):**")
                        st.info(lessons_summary)
                    else:
                        st.info("No explicit lessons detected automatically.")

    with col2:
        st.markdown("### Preview / Quick checks")
        st.write("Models loaded:" )
        st.write(f"- summarizer: {'yes' if summarizer is not None else 'no'}")
        st.write(f"- sentiment analyzer: {'yes' if sentiment_analyzer is not None else 'no'}")
        st.write(f"- embedder: {'yes' if embedder is not None else 'no'}")
        st.write("")
        st.write("Notes:")
        st.write("- If summarizer is missing, the app falls back to returning the first 400 chars.")
        st.write("- Audio transcription uses `pydub` + `speech_recognition` if available; otherwise transcription is skipped.")

elif choice == "View Debriefs":
    st.header("Stored Debriefs (enhanced table)")
    rows = fetch_debriefs_full()
    if rows:
        for row in rows:
            st.markdown(f"**ID:** {row[0]}  |  **Time:** {row[6]}")
            st.write(f"**Text:** {row[1]}")
            st.write(f"**Summary:** {row[2]}")
            st.write(f"**Sentiment:** {row[3]}")
            if row[4]:
                st.write("**Lessons (extracted):**")
                st.info(row[4])
            if row[5]:
                st.write(f"**Audio saved at:** `{row[5]}`")
                try:
                    st.audio(row[5])
                except Exception:
                    pass
            st.markdown("---")
    else:
        st.info("No debriefs found.")
