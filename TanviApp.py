import streamlit as st
import whisper
import torch
import numpy as np
import nltk
import json
from nltk.tokenize import sent_tokenize
from transformers import DistilBertTokenizer, DistilBertModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Podcast Transcription & Segmentation", layout="wide")
st.title("🎧 Podcast Transcription & Topic Segmentation")

nltk.download("punkt")

SIMILARITY_THRESHOLD = 0.68
MIN_SENTENCES_PER_SEGMENT = 7

# -----------------------------
# SESSION STATE
# -----------------------------
if "show_transcription" not in st.session_state:
    st.session_state.show_transcription = {}

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=0 if torch.cuda.is_available() else -1
    )

    return whisper_model, tokenizer, bert_model, summarizer


whisper_model, tokenizer, bert_model, summarizer = load_models()

# -----------------------------
# FILE UPLOAD
# -----------------------------
audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_embedding(text, max_len=400):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding=True
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


def extract_keywords(text, top_k=5):
    words = [w.lower() for w in text.split() if len(w) > 4 and w.isalpha()]
    words = list(set(words))[:25]
    if not words:
        return []

    text_emb = get_embedding(text)
    word_embs = np.vstack([get_embedding(w) for w in words])
    scores = cosine_similarity(text_emb, word_embs)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    return [words[i] for i in top_indices]


def summarize_text(text):
    words = text.split()
    if len(words) < 60:
        return text

    safe_text = " ".join(words[:350])
    max_len = int(len(safe_text.split()) * 0.45)

    return summarizer(
        safe_text,
        max_length=max_len,
        min_length=int(max_len * 0.6),
        do_sample=False
    )[0]["summary_text"]


def generate_title(keywords):
    return " ".join(k.capitalize() for k in keywords[:3]) if keywords else "General Discussion"


def matches_search(segment_data, query):
    query = query.lower()
    return (
        query in segment_data["title"].lower()
        or query in segment_data["summary"].lower()
        or query in segment_data["transcription"].lower()
        or any(query in k.lower() for k in segment_data["keywords"])
    )

# -----------------------------
# MAIN PROCESS
# -----------------------------
if audio_file and st.button("🚀 Process Audio"):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    with st.spinner("🔊 Transcribing audio..."):
        result = whisper_model.transcribe(audio_path)
        segments = result["segments"]

    sentences = []
    sentence_times = []

    for seg in segments:
        seg_sentences = sent_tokenize(seg["text"])
        duration = (seg["end"] - seg["start"]) / max(len(seg_sentences), 1)

        for i, s in enumerate(seg_sentences):
            start = seg["start"] + i * duration
            end = start + duration
            sentences.append(s)
            sentence_times.append((start, end))

    with st.spinner("🧠 Segmenting topics..."):
        embeddings = np.vstack([get_embedding(s) for s in sentences])

        topic_segments = []
        current_segment = [0]
        current_embeddings = [embeddings[0]]

        for i in range(1, len(embeddings)):
            mean_emb = np.mean(current_embeddings, axis=0).reshape(1, -1)
            similarity = cosine_similarity(mean_emb, embeddings[i].reshape(1, -1))[0][0]

            if similarity < SIMILARITY_THRESHOLD and len(current_segment) >= MIN_SENTENCES_PER_SEGMENT:
                topic_segments.append(current_segment)
                current_segment = [i]
                current_embeddings = [embeddings[i]]
            else:
                current_segment.append(i)
                current_embeddings.append(embeddings[i])

        topic_segments.append(current_segment)

    st.subheader("🔊 Podcast Audio")
    st.audio(audio_path)

    st.subheader("📌 Topic-wise Segments")

    search_query = st.text_input("🔍 Search segments by keyword / topic / text")

    output_json = {"segments": []}

    for idx, seg in enumerate(topic_segments):
        seg_text = " ".join([sentences[i] for i in seg]).strip()
        if not seg_text:
            continue

        start_time = round(sentence_times[seg[0]][0], 2)
        end_time = round(sentence_times[seg[-1]][1], 2)

        keywords = extract_keywords(seg_text)
        title = generate_title(keywords)
        summary = summarize_text(seg_text)

        segment_data = {
            "segment_number": idx + 1,
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "keywords": keywords,
            "summary": summary,
            "transcription": seg_text
        }

        if search_query and not matches_search(segment_data, search_query):
            continue

        btn_key = f"read_{idx}"
        if btn_key not in st.session_state.show_transcription:
            st.session_state.show_transcription[btn_key] = False

        with st.expander(f"Segment {idx + 1} | {title} ({start_time}s – {end_time}s)"):
            st.markdown("**Summary**")
            st.write(summary)

            st.markdown("**Keywords**")
            st.write(", ".join(keywords))

            label = "❌ Hide Transcription" if st.session_state.show_transcription[btn_key] else "📖 Read Full Transcription"
            if st.button(label, key=f"{btn_key}_toggle"):
                st.session_state.show_transcription[btn_key] = not st.session_state.show_transcription[btn_key]

            if st.session_state.show_transcription[btn_key]:
                st.markdown("**Segment Transcription**")
                st.write(seg_text)

        output_json["segments"].append(segment_data)

    st.download_button(
        "⬇️ Download JSON Output",
        data=json.dumps(output_json, indent=4),
        file_name="analysis.json",
        mime="application/json"
    )

    os.remove(audio_path)
