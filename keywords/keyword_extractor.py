from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords(text):
    kws = kw_model.extract_keywords(text, top_n=5)
    return [k[0] for k in kws]
