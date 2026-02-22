import os
import json
import wave
from vosk import Model, KaldiRecognizer

AUDIO_DIR = "audio"
OUT_DIR = "transcripts"
MODEL_PATH = "vosk-model"

os.makedirs(OUT_DIR, exist_ok=True)

model = Model(MODEL_PATH)

for file in os.listdir(AUDIO_DIR):
    if not file.endswith(".wav"):
        continue

    audio_path = os.path.join(AUDIO_DIR, file)
    print(f"Transcribing {file} using Vosk...")

    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    start_time = 0.0

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if "text" in res:
                results.append({
                    "start": start_time,
                    "end": start_time + 5,
                    "text": res["text"]
                })
                start_time += 5

    final_res = json.loads(rec.FinalResult())
    if "text" in final_res:
        results.append({
            "start": start_time,
            "end": start_time + 5,
            "text": final_res["text"]
        })

    whisper_style_json = {
        "language": "en",
        "segments": [
            {
                "id": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            }
            for i, seg in enumerate(results)
        ]
    }

    out_file = file.replace(".wav", ".json")
    out_path = os.path.join(OUT_DIR, out_file)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(whisper_style_json, f, indent=2)

    print(f"Saved transcript → {out_path}\n")
