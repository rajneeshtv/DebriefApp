import streamlit as st
import os
import sqlite3
from datetime import datetime
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# Database path (warning: Streamlit Cloud resets local DB when app restarts)
DB_PATH = "debrief_app.db"

# --- Initialize Database ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS debriefs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    summary TEXT,
                    sentiment TEXT,
                    embedding BLOB,
                    timestamp TEXT
                )''')
    conn.commit()
    conn.close()

# Initialize DB on first run
init_db()

# --- Cache Models ---
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    sentiment_analyzer = pipeline("sentiment-analysis")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return summarizer, sentiment_analyzer, embedder

summarizer, sentiment_analyzer, embedder = load_models()

# --- Save Debrief ---
def save_debrief(text, summary, sentiment, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO debriefs (text, summary, sentiment, embedding, timestamp) VALUES (?, ?, ?, ?, ?)",
              (text, summary, sentiment, embedding.tobytes(), datetime.now().isoformat()))
    conn.commit()
    conn.close()

# --- Fetch Debriefs ---
def fetch_debriefs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, text, summary, sentiment, timestamp FROM debriefs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# --- Streamlit App ---
st.title("📋 Debrief Application")

menu = ["Home", "View Debriefs"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Add a New Debrief")
    user_text = st.text_area("Enter debrief text:")

    if st.button("Process & Save"):
        if user_text.strip():
            with st.spinner("Processing..."):
                summary = summarizer(user_text, max_length=60, min_length=10, do_sample=False)[0]['summary_text']
                sentiment = sentiment_analyzer(user_text)[0]['label']
                embedding = embedder.encode([user_text])[0]

                save_debrief(user_text, summary, sentiment, embedding)

            st.success("Debrief saved successfully!")
            st.write("**Summary:**", summary)
            st.write("**Sentiment:**", sentiment)
        else:
            st.warning("Please enter some text before processing.")

elif choice == "View Debriefs":
    st.subheader("Stored Debriefs")
    rows = fetch_debriefs()
    if rows:
        for row in rows:
            st.markdown(f"**ID:** {row[0]} | **Time:** {row[4]}")
            st.write(f"**Text:** {row[1]}")
            st.write(f"**Summary:** {row[2]}")
            st.write(f"**Sentiment:** {row[3]}")
            st.markdown("---")
    else:
        st.info("No debriefs found.")

