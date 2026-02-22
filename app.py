# app.py
import streamlit as st
import whisper
import tempfile
import os
import numpy as np
import pandas as pd
import torch
import subprocess
import shutil
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from keybert import KeyBERT

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="AI Podcast Intelligence Platform",
                   layout="wide",
                   page_icon="🎙️")

st.title("🎙️ AI Podcast Topic Intelligence Platform")
st.caption("Semantic Segmentation • NLP Analytics • Interactive Navigation")

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
@st.cache_resource
def load_models():
    # Whisper ASR
    whisper_model = whisper.load_model("base")

    # device for transformers pipelines: 0 if GPU else -1 for CPU
    device = 0 if torch.cuda.is_available() else -1

    # Embeddings for semantic segmentation
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Summarizer (BART)
    summarizer = pipeline("summarization",
                          model="facebook/bart-large-cnn",
                          device=device)

    # Keyword extractor
    kw_model = KeyBERT("all-MiniLM-L6-v2")

    # Sentiment pipeline (uses a small finetuned model)
    sentiment_pipeline = pipeline("sentiment-analysis",
                                  model="distilbert-base-uncased-finetuned-sst-2-english",
                                  device=device)

    return whisper_model, embedder, summarizer, kw_model, sentiment_pipeline

whisper_model, embedder, summarizer, kw_model, sentiment_pipeline = load_models()

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def sec_to_hms(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def extract_keywords(text, top_n=5):
    try:
        kws = kw_model.extract_keywords(text,
                                        keyphrase_ngram_range=(1,2),
                                        stop_words="english",
                                        top_n=top_n)
        return [k[0] for k in kws]
    except Exception:
        return []

def safe_summarize(text, max_len=200):
    if len(text.split()) < 80:
        return text
    try:
        summary = summarizer(text[:1500],
                             max_length=max_len,
                             min_length=80,
                             do_sample=False)
        return summary[0]["summary_text"]
    except Exception:
        return text[:400] + "..."

def semantic_segmentation(segments, threshold=0.30, min_duration=60):
    # segments: whisper raw segments list of dicts with 'start','end','text'
    if not segments:
        return []

    texts = [s["text"] for s in segments]
    times = [(s["start"], s["end"]) for s in segments]
    embeddings = embedder.encode(texts)

    final = []
    start, end = times[0]
    current_text = texts[0]
    current_emb = embeddings[0]

    for i in range(1, len(texts)):
        try:
            sim = cosine_similarity(current_emb.reshape(1,-1),
                                    embeddings[i].reshape(1,-1))[0][0]
        except Exception:
            sim = 1.0

        duration = end - start

        if sim < threshold and duration > min_duration:
            final.append({"start": start, "end": end, "text": current_text})
            start, end = times[i]
            current_text = texts[i]
            current_emb = embeddings[i]
        else:
            end = times[i][1]
            current_text += " " + texts[i]
            current_emb = np.mean([current_emb, embeddings[i]], axis=0)

    final.append({"start": start, "end": end, "text": current_text})
    return final

def convert_to_wav(input_path, output_path):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [ffmpeg_path, "-y", "-i", input_path,
           "-ar", "16000", "-ac", "1", "-vn", output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def compute_sentiment_score(text):
    # use transformers sentiment pipeline; map POSITIVE -> +score, NEGATIVE -> -score
    try:
        # pipeline may have max token limits -- take first 512 tokens as safe slice
        out = sentiment_pipeline(text[:512])
        if isinstance(out, list) and len(out) > 0:
            label = out[0]["label"]
            score = float(out[0]["score"])
            return score if label == "POSITIVE" else -score
    except Exception:
        pass
    return 0.0

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "segments" not in st.session_state:
    st.session_state.segments = None
if "index" not in st.session_state:
    st.session_state.index = 0
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

# ---------------------------------------------------
# FILE UPLOAD UI
# ---------------------------------------------------
uploaded = st.file_uploader("Upload Podcast audio (mp3/wav/m4a)", type=["mp3","wav","m4a"])

if uploaded:
    st.audio(uploaded)

if uploaded and st.button("🚀 Analyze Podcast"):
    progress = st.progress(0)

    # save uploaded to temp file
    with tempfile.NamedTemporaryFile(delete=False,
                                     suffix=os.path.splitext(uploaded.name)[1]) as tmp_input:
        tmp_input.write(uploaded.read())
        input_path = tmp_input.name

    progress.progress(15)

    # convert to wav 16k mono
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav_path = tmp_wav.name

    try:
        convert_to_wav(input_path, wav_path)
    except Exception as e:
        st.error(f"ffmpeg conversion failed: {e}")
        raise

    progress.progress(40)

    # transcribe with whisper
    try:
        result = whisper_model.transcribe(wav_path, fp16=False)
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        raise

    progress.progress(65)

    # semantic segmentation of whisper segments
    raw_segments = result.get("segments", [])
    topic_segments = semantic_segmentation(raw_segments, threshold=0.30, min_duration=60)

    processed = []
    full_text = ""
    for seg in topic_segments:
        text = seg["text"].strip()
        full_text += text + "\n\n"
        processed.append({
            "start": seg["start"],
            "end": seg["end"],
            "duration": seg["end"] - seg["start"],
            "summary": safe_summarize(text, max_len=200),
            "keywords": extract_keywords(text),
            "word_count": len(text.split()),
            "sentiment": compute_sentiment_score(text)
        })

    st.session_state.segments = processed
    st.session_state.full_text = full_text
    st.session_state.index = 0

    # cleanup
    try:
        os.remove(input_path)
        os.remove(wav_path)
    except Exception:
        pass

    progress.progress(100)
    progress.empty()
    st.success("Analysis complete ✅")

# ---------------------------------------------------
# SHOW RESULTS
# ---------------------------------------------------
if st.session_state.segments:
    segments = st.session_state.segments
    idx = st.session_state.index

    total_words = sum([s["word_count"] for s in segments])
    total_duration = segments[-1]["end"] if len(segments) else 0
    wpm = (total_words / (total_duration/60)) if total_duration > 0 else 0.0

    # top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments", len(segments))
    c2.metric("Total Duration", sec_to_hms(int(total_duration)))
    c3.metric("Total Words", int(total_words))
    c4.metric("Words/Minute", f"{wpm:.1f}")

    st.divider()

    # navigation
    left, center, right = st.columns([1,2,1])
    with left:
        if st.button("⬅ Previous") and idx > 0:
            st.session_state.index -= 1
    with center:
        st.markdown(f"## Segment {idx+1} of {len(segments)}")
    with right:
        if st.button("Next ➡") and idx < len(segments)-1:
            st.session_state.index += 1

    seg = segments[st.session_state.index]
    st.markdown(f"### ⏱ {sec_to_hms(int(seg['start']))} - {sec_to_hms(int(seg['end']))}")
    st.write("**Keywords:**", ", ".join(seg["keywords"]))
    st.write("**Word Count:**", seg["word_count"])
    st.write("**Sentiment score:**", round(seg["sentiment"], 3))
    st.write(seg["summary"])

    st.divider()

    # visualization buttons
    col_wc, col_sent, col_tl = st.columns(3)
    show_wordcloud = col_wc.button("☁ Show Word Cloud")
    show_sentiment = col_sent.button("📈 Show Sentiment")
    show_timeline = col_tl.button("📊 Show Timeline")

    if show_wordcloud:
        wc = WordCloud(width=900, height=300, background_color="white").generate(st.session_state.full_text)
        fig, ax = plt.subplots(figsize=(10,3))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    if show_sentiment:
        sentiments = [s["sentiment"] for s in segments]
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(range(1, len(sentiments)+1), sentiments, marker="o")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Sentiment Score")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    if show_timeline:
        starts = [s["start"] for s in segments]
        durations = [s["duration"] for s in segments]
        labels = [f"Segment {i+1}" for i in range(len(segments))]
        fig, ax = plt.subplots(figsize=(10,3))
        ax.barh(labels, durations, left=starts)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("")
        st.pyplot(fig)

    st.divider()

    st.header("📌 Overall Podcast Summary")
    st.write(safe_summarize(st.session_state.full_text, max_len=300))

    # downloads
    st.download_button("Download Full Transcript (txt)",
                       st.session_state.full_text,
                       file_name="podcast_transcript.txt")

    df = pd.DataFrame(segments)
    st.download_button("Download Segments (CSV)",
                       df.to_csv(index=False),
                       file_name="podcast_segments.csv")
import sys
import os
import json
import tempfile
import base64
import streamlit as st

# --------------------------------------------------
# Ensure project root is in Python path
# --------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------
# Import project modules
# --------------------------------------------------
from stt.whisper_stt import transcribe
from topic_segmentation.segmenter import segment_topics

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Podcast Navigation System",
    layout="wide"
)

st.title("🎧 Podcast Navigation System")
st.markdown("Navigate podcasts using topic-based audio navigation.")

# --------------------------------------------------
# UI & Styling Assets (UNCHANGED)
# --------------------------------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

bg_img_path = os.path.join(os.path.dirname(__file__), "assets/waveform_bg.png")

try:
    bin_str = get_base64_of_bin_file(bg_img_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
            linear-gradient(rgba(10,10,10,0.92), rgba(10,10,10,0.95)),
            url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            font-family: 'Inter', sans-serif;
        }}
        mark {{
            background-color: #facc15;
            color: black;
            padding: 2px 4px;
            border-radius: 4px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
except:
    pass

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "seek_time" not in st.session_state:
    st.session_state.seek_time = 0

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None

if "processed" not in st.session_state:
    st.session_state.processed = False

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def highlight(text, keyword):
    if not keyword:
        return text
    return text.replace(
        keyword,
        f"<mark>{keyword}</mark>"
    )

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload podcast audio",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file:

    # Detect new audio
    if st.session_state.last_uploaded_name != uploaded_file.name:
        if os.path.exists("data/transcript/transcript.json"):
            os.remove("data/transcript/transcript.json")
        if os.path.exists("data/segments/segments.json"):
            os.remove("data/segments/segments.json")

        st.session_state.seek_time = 0
        st.session_state.selected_topic = None
        st.session_state.processed = False
        st.session_state.last_uploaded_name = uploaded_file.name

    # Save audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    # Heavy processing (once)
    if not st.session_state.processed:
        with st.spinner("Processing podcast audio..."):
            transcribe(audio_path)
            segment_topics()

        st.session_state.processed = True
        st.success("Podcast processed successfully ✅")

    # Load outputs
    with open("data/segments/segments.json") as f:
        segments = json.load(f)

    with open("data/transcript/transcript.json") as f:
        transcript_json = json.load(f)

    # Audio player
    st.audio(audio_path, start_time=int(st.session_state.seek_time))

    # Layout
    col1, col2 = st.columns([1.3, 2])

    # ---------------- LEFT: Chapters ----------------
    with col1:
        st.subheader("🧭 Chapters")

        search_query = st.text_input(
            "🔍 Search keyword",
            placeholder="e.g. Malta, learning, podcast"
        )

        filtered_segments = segments
        if search_query:
            q = search_query.lower()
            filtered_segments = [
                seg for seg in segments
                if q in seg["title"].lower()
                or q in seg.get("summary", "").lower()
                or q in seg["text"].lower()
            ]

        if search_query and not filtered_segments:
            st.warning("No matching segments found.")

        for i, seg in enumerate(filtered_segments, start=1):
            title_html = highlight(seg["title"], search_query)

            st.markdown(
                f"""
                <div style="
                    padding:16px;
                    margin-bottom:10px;
                    border-radius:12px;
                    background:rgba(255,255,255,0.04);
                    border:1px solid rgba(255,255,255,0.08);
                ">
                    <div style="font-size:15px;color:#f1f5f9;">
                        {title_html}
                    </div>
                    <div style="font-size:13px;color:#94a3b8;">
                        ⏱ {format_time(seg['start'])} – {format_time(seg['end'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(f"▶ Play", key=f"play_{i}"):
                st.session_state.seek_time = int(seg["start"])
                st.session_state.selected_topic = seg
                st.rerun()

    # ---------------- RIGHT: Topic Details ----------------
    with col2:
        if st.session_state.selected_topic:
            seg = st.session_state.selected_topic

            st.subheader("📌 Topic Details")
            st.markdown(f"**Title:** {seg['title']}")
            st.markdown(
                f"**Time:** {format_time(seg['start'])} – {format_time(seg['end'])}"
            )

            # Summary (highlighted)
            st.markdown("### ✨ Summary")
            summary_html = highlight(seg.get("summary", ""), search_query)

            st.markdown(
                f"""
                <div style="
                    padding:14px;
                    border-radius:10px;
                    background:rgba(59,130,246,0.12);
                    border:1px solid rgba(59,130,246,0.35);
                    color:#e5e7eb;
                    line-height:1.6;
                    margin-bottom:18px;
                ">
                    {summary_html}
                </div>
                """,
                unsafe_allow_html=True
            )

            # Transcript (highlighted)
            st.markdown("### 📄 Transcript")
            st.text_area(
                "",
                highlight(seg["text"], search_query),
                height=320
            )
        else:
            # Display the high-impact title
            display_title = "English Learning For Curious Minds: The Evolution"

            st.markdown(
                f"""
                <div style="
                    text-align: center;
                    padding: 40px;
                    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
                    border-radius: 16px;
                    border: 1px solid rgba(255,255,255,0.1);
                    margin-top: 20px;
                ">
                    <h1 style="
                        font-size: 2.5rem;
                        font-weight: 700;
                        color: #f8fafc;
                        background: linear-gradient(90deg, #60a5fa, #c084fc);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        margin-bottom: 16px;
                    ">
                        {display_title}
                    </h1>
                    <p style="
                        font-size: 1.1rem;
                        color: #94a3b8;
                        line-height: 1.6;
                    ">
                        Select a topic from the sidebar to explore the transcript segments.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Downloads
    st.divider()

    st.download_button(
        "⬇️ Download Full Transcript (JSON)",
        json.dumps(transcript_json, indent=4),
        "transcript.json",
        "application/json"
    )

    st.download_button(
        "⬇️ Download Topic Segments (JSON)",
        json.dumps(segments, indent=4),
        "segments.json",
        "application/json"
    )

    # Cleanup
    try:
        os.remove(audio_path)
    except:
        pass

else:
    st.info("Upload a podcast audio file to begin.")
from flask import Flask, render_template, request, jsonify
import whisper
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = "audio"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = whisper.load_model("tiny")

def generate_title(text):
    words = text.split()
    return " ".join(words[:4]).title()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_audio():
    file = request.files["audio"]
    audio_path = os.path.join(UPLOAD_FOLDER, "uploaded.wav")
    file.save(audio_path)

    result = model.transcribe(audio_path, word_timestamps=True)

    segments = []
    buffer_text = ""
    start_time = result["segments"][0]["start"]

    for i, seg in enumerate(result["segments"]):
        buffer_text += " " + seg["text"]

        if (i + 1) % 3 == 0 or i == len(result["segments"]) - 1:
            end_time = seg["end"]
            segments.append({
                "title": generate_title(buffer_text),
                "start": round(start_time),
                "end": round(end_time),
                "summary": buffer_text.strip(),
                "text": buffer_text.strip()
            })
            buffer_text = ""
            if i + 1 < len(result["segments"]):
                start_time = result["segments"][i + 1]["start"]

    output_path = os.path.join(OUTPUT_FOLDER, "podcast.json")
    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)

    return jsonify(segments)

if __name__ == "__main__":
    app.run(debug=True)
