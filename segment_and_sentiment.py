import os
import whisper
import nltk
from transformers import pipeline

# Download required NLP data
nltk.download('punkt')

# -----------------------------
# CONFIGURATION
# -----------------------------
AUDIO_PATH = "audio/new_podcast.wav"
OUTPUT_DIR = "output"

SEGMENT_WORD_LIMIT = 100  # good for 22-minute audio

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading Whisper model (small)...")
whisper_model = whisper.load_model("small")

print("Loading Sentiment Analysis model...")
sentiment_model = pipeline("sentiment-analysis")

# -----------------------------
# TRANSCRIBE AUDIO
# -----------------------------
print("Transcribing audio...")
result = whisper_model.transcribe(AUDIO_PATH)

full_text = result["text"]

# Save full transcript
with open(f"{OUTPUT_DIR}/full_transcript.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Full transcription saved.")

# -----------------------------
# SEGMENT TEXT
# -----------------------------
sentences = nltk.sent_tokenize(full_text)

segments = []
current_segment = ""

for sentence in sentences:
    current_segment += " " + sentence

    if len(current_segment.split()) >= SEGMENT_WORD_LIMIT:
        segments.append(current_segment.strip())
        current_segment = ""

# Add remaining text
if current_segment.strip():
    segments.append(current_segment.strip())

print(f"Total Segments Created: {len(segments)}")

# -----------------------------
# SENTIMENT ANALYSIS ON SEGMENTS
# -----------------------------
output_lines = []

for idx, segment in enumerate(segments, 1):
    sentiment = sentiment_model(segment[:512])[0]  # limit text for model
    label = sentiment["label"]
    score = round(sentiment["score"], 3)

    output_lines.append(
        f"SEGMENT {idx}\n"
        f"Sentiment: {label} (Confidence: {score})\n"
        f"Text:\n{segment}\n"
        f"{'-'*60}\n"
    )

# Save segmented output
with open(f"{OUTPUT_DIR}/segmented_output.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)

print("Segmentation + Sentiment analysis completed successfully!")
