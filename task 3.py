import whisper
import torch
import numpy as np
import nltk
import json
from nltk.tokenize import sent_tokenize
from transformers import DistilBertTokenizer, DistilBertModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# INITIAL SETUP
# -----------------------------
nltk.download("punkt")

AUDIO_PATH = "common.mp3"

TRANSCRIPTION_FILE = "transcription.txt"
ANALYSIS_TXT_FILE = "analysis.txt"
ANALYSIS_JSON_FILE = "analysis.json"

SIMILARITY_THRESHOLD = 0.68
MIN_SENTENCES_PER_SEGMENT = 7

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading models...")

whisper_model = whisper.load_model("base")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=0 if torch.cuda.is_available() else -1
)

# -----------------------------
# STEP 1: TRANSCRIPTION
# -----------------------------
print("Transcribing audio...")
result = whisper_model.transcribe(AUDIO_PATH)
segments = result["segments"]

with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Transcription saved.")

# -----------------------------
# STEP 2: SENTENCE SPLITTING
# -----------------------------
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

# -----------------------------
# STEP 3: EMBEDDINGS
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

print("Generating embeddings...")
embeddings = np.vstack([get_embedding(s) for s in sentences])

# -----------------------------
# STEP 4: IMPROVED TOPIC SEGMENTATION
# -----------------------------
topic_segments = []
current_segment = [0]
current_embeddings = [embeddings[0]]

for i in range(1, len(embeddings)):
    segment_mean = np.mean(current_embeddings, axis=0).reshape(1, -1)
    similarity = cosine_similarity(
        segment_mean,
        embeddings[i].reshape(1, -1)
    )[0][0]

    if similarity < SIMILARITY_THRESHOLD and len(current_segment) >= MIN_SENTENCES_PER_SEGMENT:
        topic_segments.append(current_segment)
        current_segment = [i]
        current_embeddings = [embeddings[i]]
    else:
        current_segment.append(i)
        current_embeddings.append(embeddings[i])

topic_segments.append(current_segment)

print(f"Total segments formed: {len(topic_segments)}")

# -----------------------------
# STEP 5: KEYWORD EXTRACTION
# -----------------------------
def extract_keywords(text, top_k=5):
    words = [
        w.lower() for w in text.split()
        if len(w) > 4 and w.isalpha()
    ]
    words = list(set(words))[:25]

    if not words:
        return []

    text_emb = get_embedding(text)
    word_embs = np.vstack([get_embedding(w) for w in words])

    scores = cosine_similarity(text_emb, word_embs)[0]
    top_indices = scores.argsort()[-top_k:][::-1]

    return [words[i] for i in top_indices]

# -----------------------------
# STEP 6: BETTER SUMMARIZATION
# -----------------------------
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

# -----------------------------
# STEP 7: TITLE GENERATION
# -----------------------------
def generate_title(keywords):
    if not keywords:
        return "General Discussion"
    return " ".join([k.capitalize() for k in keywords[:3]])

# -----------------------------
# STEP 8: WRITE OUTPUT FILES
# -----------------------------
json_output = {"segments": []}

with open(ANALYSIS_TXT_FILE, "w", encoding="utf-8") as txt:
    for idx, seg in enumerate(topic_segments):
        seg_text = " ".join([sentences[i] for i in seg]).strip()
        if not seg_text:
            continue

        start_time = sentence_times[seg[0]][0]
        end_time = sentence_times[seg[-1]][1]

        keywords = extract_keywords(seg_text)
        title = generate_title(keywords)
        summary = summarize_text(seg_text)

        txt.write(f"Segment {idx + 1}\n")
        txt.write(f"Title: {title}\n")
        txt.write(f"Start Time: {round(start_time,2)}s - {round(end_time,2)}s\n")
        txt.write(f"Keywords: {', '.join(keywords)}\n")
        txt.write(f"Summary: {summary}\n")
        txt.write("-" * 65 + "\n")

        json_output["segments"].append({
            "segment_number": idx + 1,
            "title": title,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "keywords": keywords,
            "summary": summary
        })

with open(ANALYSIS_JSON_FILE, "w", encoding="utf-8") as jf:
    json.dump(json_output, jf, indent=4)

print("All outputs generated successfully.")
