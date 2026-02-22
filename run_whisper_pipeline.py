import os
import sys
import json
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

AUDIO_DIR = "audio"
TRANSCRIPT_DIR = "transcripts"
OUTPUT_DIR = "final_output"

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- ARG CHECK ----------------
if len(sys.argv) < 2:
    print("❌ No audio file provided")
    sys.exit(1)

audio_file = sys.argv[1]
audio_path = os.path.join(AUDIO_DIR, audio_file)

print(f"\nProcessing audio: {audio_file}")

# ---------------- LOAD MODELS ----------------
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading summarization model...")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

print("Models loaded.\n")

# ---------------- TRANSCRIPTION ----------------
result = whisper_model.transcribe(audio_path)
segments = result["segments"]

with open(
    os.path.join(TRANSCRIPT_DIR, audio_file.replace(".wav", ".json")),
    "w",
    encoding="utf-8"
) as f:
    json.dump(result, f, indent=2)

# ---------------- TOPIC SEGMENTATION ----------------
texts = [s["text"] for s in segments]
embeddings = embedder.encode(texts)

THRESHOLD = 0.75
topic_breaks = [0]

for i in range(1, len(embeddings)):
    sim = cosine_similarity(
        [embeddings[i - 1]],
        [embeddings[i]]
    )[0][0]

    if sim < THRESHOLD:
        topic_breaks.append(i)

topic_breaks.append(len(segments))

# ---------------- HELPERS ----------------
def sec_to_time(sec):
    m, s = divmod(int(sec), 60)
    return f"00:{m:02d}:{s:02d}"

# ---------------- BUILD FINAL OUTPUT ----------------
final_segments = []

for i in range(len(topic_breaks) - 1):
    start = topic_breaks[i]
    end = topic_breaks[i + 1]

    chunk = " ".join(s["text"] for s in segments[start:end])
    if len(chunk.split()) < 25:
        continue

    summary = summarizer(
        chunk,
        max_length=130,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]

    final_segments.append({
        "segment": len(final_segments) + 1,
        "start_time": sec_to_time(segments[start]["start"]),
        "end_time": sec_to_time(segments[end - 1]["end"]),
        "summary": summary
    })

# ---------------- SAVE ----------------
output_path = os.path.join(
    OUTPUT_DIR,
    audio_file.replace(".wav", "_segments.json")
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_segments, f, indent=2)

print(f"✅ Saved structured output → {output_path}")
