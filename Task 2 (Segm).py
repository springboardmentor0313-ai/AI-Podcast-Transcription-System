import whisper
import torch
import nltk
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

# Download punkt tokenizer (first time only)
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# -----------------------------
# STEP 1: WHISPER TRANSCRIPTION
# -----------------------------

print("🔊 Loading Whisper model...")
whisper_model = whisper.load_model("base")

audio_path = r"C:\Users\Admin\Audios\common.mp3"

print("📝 Transcribing audio...")
result = whisper_model.transcribe(audio_path)
transcription_text = result["text"]

# Save raw transcription
with open("common_transcription.txt", "w", encoding="utf-8") as f:
    f.write(transcription_text)

print("✅ Transcription saved.")

# -----------------------------
# STEP 2: SENTENCE SPLITTING
# -----------------------------

sentences = sent_tokenize(transcription_text)

print(f"🔹 Total sentences detected: {len(sentences)}")

# -----------------------------
# STEP 3: LOAD DISTILBERT MODEL
# -----------------------------

print("🤖 Loading DistilBERT model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# -----------------------------
# STEP 4: EMBEDDING FUNCTION
# -----------------------------

def get_embedding(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding

# Generate embeddings
embeddings = [get_embedding(sentence) for sentence in sentences]

# -----------------------------
# STEP 5: SEGMENTATION LOGIC
# -----------------------------

SIMILARITY_THRESHOLD = 0.75

segments = []
current_segment = [sentences[0]]

for i in range(1, len(sentences)):
    sim = cosine_similarity(
        embeddings[i - 1],
        embeddings[i]
    )[0][0]

    if sim < SIMILARITY_THRESHOLD:
        segments.append(" ".join(current_segment))
        current_segment = [sentences[i]]
    else:
        current_segment.append(sentences[i])

segments.append(" ".join(current_segment))

# -----------------------------
# STEP 6: SAVE SEGMENTED OUTPUT
# -----------------------------

with open("segmented_transcription.txt", "w", encoding="utf-8") as f:
    for idx, segment in enumerate(segments, 1):
        f.write(f"\n--- Segment {idx} ---\n")
        f.write(segment + "\n")

print("✅ Segmentation completed.")
print(f"📂 Total Segments Created: {len(segments)}")
print("📄 Segmented output saved as segmented_transcription.txt")
