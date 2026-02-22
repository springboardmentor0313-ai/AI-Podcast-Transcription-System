from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import numpy as np

def process_semantic_segments(segments_text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(model)
    
    embeddings = model.encode(segments_text)
    sims = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings)-1)]
    
    # Identify topic shifts
    boundaries = sorted(np.argsort(sims)[:10] + 1)
    return boundaries, kw_model

def get_unique_title(kw_model, chunk, segment_index):
    """Generates a 2-3 word phrase unique to the segment"""
    
    candidates = kw_model.extract_keywords(
        chunk, 
        keyphrase_ngram_range=(2, 3), 
        stop_words='english', 
        top_n=5
    )
    
    if candidates:
        # Return the top unique phrase, title-cased
        return candidates[0][0].title()
    return f"Discussion Part {segment_index + 1}"

def search_transcript(query, report):
    if not query: return report
    q = query.lower()
    return [seg for seg in report if q in seg['content'].lower() or q in seg['title'].lower()]