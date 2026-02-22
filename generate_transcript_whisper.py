import os
import json
import whisper

# -------------------------
# Configuration
# -------------------------
AUDIO_DIR = "audio"
OUTPUT_DIR = "transcripts"
MODEL_SIZE = "medium"   # good balance for long podcasts

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Load Whisper model
# -------------------------
print("Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE)
print("Whisper model loaded.\n")

# -------------------------
# Process each audio file
# -------------------------
for audio_file in os.listdir(AUDIO_DIR):
    if not audio_file.lower().endswith((".wav", ".mp3", ".m4a")):
        continue

    audio_path = os.path.join(AUDIO_DIR, audio_file)
    output_path = os.path.join(
        OUTPUT_DIR,
        audio_file.rsplit(".", 1)[0] + ".json"
    )

    print(f"Transcribing {audio_file} using Whisper...")

    result = model.transcribe(
        audio_path,
        fp16=False,        # CPU safe
        verbose=False
    )

    transcript_data = {
        "language": result["language"],
        "text": result["text"],
        "segments": []
    }

    for seg in result["segments"]:
        transcript_data["segments"].append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    print(f"Saved transcript → {output_path}\n")

print("All audio files transcribed successfully.")