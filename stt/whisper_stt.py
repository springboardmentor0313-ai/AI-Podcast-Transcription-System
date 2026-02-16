import whisper
import os
import json

model = whisper.load_model("base")

def transcribe(audio_path):
    result = model.transcribe(audio_path)

    transcript_text = result["text"]
    segments = result["segments"]

    os.makedirs("data/transcript", exist_ok=True)

    transcript_json = {
        "full_transcript": transcript_text,
        "segments": segments
    }

    with open("data/transcript/transcript.json", "w") as f:
        json.dump(transcript_json, f, indent=4)

    return transcript_text
