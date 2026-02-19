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
