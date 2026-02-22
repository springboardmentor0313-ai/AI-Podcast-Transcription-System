import sys
import os
import json
from transformers import pipeline

# --------------------------------------------------
# Fix Python path to allow sibling-folder imports
# --------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keywords.keyword_extractor import extract_keywords

# --------------------------------------------------
# Paths
# --------------------------------------------------
SEGMENTS_PATH = "../data/segments/segments.json"
OUTPUT_DIR = "../data/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "milestone2_output.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load Segments
# --------------------------------------------------
with open(SEGMENTS_PATH, "r") as f:
    segments = json.load(f)

# --------------------------------------------------
# Load Transformer Summarization Model
# --------------------------------------------------
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# --------------------------------------------------
# Utility: Format seconds to HH:MM:SS
# --------------------------------------------------
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

# --------------------------------------------------
# Generate Structured Output
# --------------------------------------------------
output_lines = []

for i, seg in enumerate(segments, start=1):
    text = seg["text"]

    # --- Generate Summary ---
    summary = summarizer(
        text,
        max_length=90,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]

    # --- Generate Title (short summary) ---
    title = summarizer(
        text,
        max_length=15,
        min_length=5,
        do_sample=False
    )[0]["summary_text"]

    # --- Extract Keywords ---
    keywords = extract_keywords(text)

    # --- Format Output ---
    segment_output = f"""
Segment {i}
Title: {title}
Start Time: {format_time(seg["start"])}
End Time: {format_time(seg["end"])}
Keywords: {", ".join(keywords)}
Summary:
{summary}

--------------------------------------------------
"""
    output_lines.append(segment_output.strip())

# --------------------------------------------------
# Save to File
# --------------------------------------------------
with open(OUTPUT_FILE, "w") as f:
    f.write("\n\n".join(output_lines))

print(f"Milestone-2 output successfully saved to: {OUTPUT_FILE}")
