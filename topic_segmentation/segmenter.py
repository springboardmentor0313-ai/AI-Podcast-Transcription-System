import json
import os
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


def segment_topics():
    """
    Performs topic segmentation and generates
    YouTube-style chapter titles and summaries.
    """

    transcript_path = os.path.join("data", "transcript", "transcript.json")
    if not os.path.exists(transcript_path):
        raise FileNotFoundError("transcript.json not found")

    with open(transcript_path, "r") as f:
        data = json.load(f)

    segments_data = data["segments"]

    sentences, timestamps = [], []
    for seg in segments_data:
        text = seg["text"].strip()
        if len(text) > 20:
            sentences.append(text)
            timestamps.append((seg["start"], seg["end"]))

    # -------- Models --------
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # -------- Parameters --------
    BLOCK = 8
    THRESHOLD = 0.65
    MIN_BLOCKS = 2

    # -------- Create Blocks --------
    blocks, block_times = [], []
    for i in range(0, len(sentences), BLOCK):
        blocks.append(" ".join(sentences[i:i + BLOCK]))
        block_times.append((
            timestamps[i][0],
            timestamps[min(i + BLOCK - 1, len(timestamps) - 1)][1]
        ))

    embeddings = embedder.encode(blocks)

    # -------- Topic Boundaries --------
    boundaries, last = [], 0
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        if sim < THRESHOLD and (i - last) >= MIN_BLOCKS:
            boundaries.append(i)
            last = i

    # -------- Build Segments --------
    raw_segments, start = [], 0
    for b in boundaries:
        raw_segments.append({
            "text": " ".join(blocks[start:b + 1]),
            "start": block_times[start][0],
            "end": block_times[b][1]
        })
        start = b + 1

    raw_segments.append({
        "text": " ".join(blocks[start:]),
        "start": block_times[start][0],
        "end": block_times[-1][1]
    })

    # -------- CLEAN TITLE GENERATION (FIXED) --------
        # -------- CLEAN TITLE GENERATION (UPGRADED) --------
        # -------- CLEAN TITLE GENERATION (IMPROVED) --------
    def generate_title(text):
        # Generate concise summary
        summary = summarizer(
            text[:900],
            max_length=22,
            min_length=10,
            do_sample=False
        )[0]["summary_text"]

        # Remove filler phrases
        summary = re.sub(
            r"\b(this episode|in this episode|the speaker|this podcast|we discuss|he talks about|she talks about)\b",
            "",
            summary,
            flags=re.IGNORECASE
        )

        summary = summary.strip()

        # Capitalize properly
        if len(summary) > 1:
            title = summary[0].upper() + summary[1:]
        else:
            title = summary

        # Remove trailing punctuation
        title = re.sub(r'[.!?]+$', '', title)

        # Fallback safety
        if len(title.split()) < 4:
            title = "Key Discussion And Insights"

        return title

    # -------- SUMMARY (UNCHANGED) --------
    def generate_summary(text):
        return summarizer(
            text[:1024],
            max_length=90,
            min_length=45,
            do_sample=False
        )[0]["summary_text"]

    # -------- Final Output --------
    final_segments = []
    for seg in raw_segments:
        final_segments.append({
            "title": generate_title(seg["text"]),
            "summary": generate_summary(seg["text"]),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })

    os.makedirs("data/segments", exist_ok=True)
    with open("data/segments/segments.json", "w") as f:
        json.dump(final_segments, f, indent=2)

    return final_segments
