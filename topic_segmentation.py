import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")

# -----------------------------
# Load NLP Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load Transcript
# -----------------------------
with open("transcripts/audio1.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# -----------------------------
# Sentence Tokenization
# -----------------------------
sentences = nltk.sent_tokenize(text)

print(f"Total sentences: {len(sentences)}")

# -----------------------------
# Generate Sentence Embeddings
# -----------------------------
embeddings = model.encode(sentences)

# -----------------------------
# Compute Similarity Between Adjacent Sentences
# -----------------------------
similarities = []
for i in range(len(embeddings) - 1):
    sim = cosine_similarity(
        [embeddings[i]], [embeddings[i + 1]]
    )[0][0]
    similarities.append(sim)

# -----------------------------
# Detect Topic Boundaries
# -----------------------------
THRESHOLD = 0.60  # can tune (0.55–0.65)

boundaries = [0]  # first sentence is always a boundary

for i, sim in enumerate(similarities):
    if sim < THRESHOLD:
        boundaries.append(i + 1)

# -----------------------------
# Segment Topics
# -----------------------------
segments = []

for i in range(len(boundaries)):
    start = boundaries[i]
    end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
    segment_text = " ".join(sentences[start:end])
    segments.append(segment_text)

# -----------------------------
# Display Results
# -----------------------------
print("\n===== TOPIC SEGMENTS =====\n")

for i, seg in enumerate(segments, 1):
    print(f"--- Topic {i} ---")
    print(seg[:500])  # show first 500 chars
    print()
