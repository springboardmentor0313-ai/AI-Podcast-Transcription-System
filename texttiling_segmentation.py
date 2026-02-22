import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🔴 FIX for Python 3.12
nltk.download("punkt")
nltk.download("punkt_tab")

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
# TF-IDF Representation
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf = vectorizer.fit_transform(sentences)

# -----------------------------
# Similarity Between Adjacent Sentences
# -----------------------------
similarities = []
for i in range(len(sentences) - 1):
    sim = cosine_similarity(tfidf[i], tfidf[i + 1])[0][0]
    similarities.append(sim)

# -----------------------------
# Topic Boundary Detection
# -----------------------------
threshold = np.mean(similarities) - np.std(similarities)

boundaries = [0]
for i, sim in enumerate(similarities):
    if sim < threshold:
        boundaries.append(i + 1)

# -----------------------------
# Segment Topics
# -----------------------------
segments = []
for i in range(len(boundaries)):
    start = boundaries[i]
    end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
    segments.append(" ".join(sentences[start:end]))

# -----------------------------
# Output
# -----------------------------
print("\n===== TOPIC SEGMENTS (TextTiling / NLP) =====\n")

for i, seg in enumerate(segments, 1):
    print(f"--- Topic {i} ---")
    print(seg[:400])
    print()
