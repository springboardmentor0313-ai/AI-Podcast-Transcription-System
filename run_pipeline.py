import os
import json
from transformers import pipeline

TRANSCRIPT_DIR = "transcripts"

# Load LLMs
print("Loading LLM models...")
summarizer = pipeline(
    "summarization",
    model="google/flan-t5-small",
    max_new_tokens=120
)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=40
)
print("Models loaded.\n")


def seconds_to_time(sec):
    m, s = divmod(int(sec), 60)
    return f"00:{m:02d}:{s:02d}"


for file in os.listdir(TRANSCRIPT_DIR):
    if not file.endswith(".json"):
        continue

    audio_name = file.replace(".json", "")
    print(f"\n================ AUDIO FILE: {audio_name} =================\n")

    with open(os.path.join(TRANSCRIPT_DIR, file), "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]
    full_text = " ".join(seg["text"] for seg in segments)

    # Generate summary
    summary = summarizer(full_text)[0]["summary_text"]

    # Generate title
    title_prompt = f"Generate a short title for this content:\n{summary}"
    title = generator(title_prompt)[0]["generated_text"]

    start_time = seconds_to_time(segments[0]["start"])
    end_time = seconds_to_time(segments[-1]["end"])

    print("Segment 1")
    print(f"Title: {title}")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    print("Summary:")
    print(summary)
    print("\n" + "-"*60)
