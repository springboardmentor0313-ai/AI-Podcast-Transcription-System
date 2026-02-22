import os
import json
import wave
import numpy as np
import soundfile as sf

import whisper
from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer
from jiwer import wer

# =========================
# Paths
# =========================
AUDIO_DIR = "audio"
REF_DIR = "reference"
VOSK_MODEL_PATH = "vosk-model"

# =========================
# Load Models
# =========================
print("Loading models...")

whisper_model = whisper.load_model("base")
faster_model = WhisperModel("base", device="cpu", compute_type="int8")
vosk_model = Model(VOSK_MODEL_PATH)

print("Models loaded.\n")

# =========================
# Helper Functions
# =========================
def load_reference(txt_file):
    with open(os.path.join(REF_DIR, txt_file), "r", encoding="utf-8") as f:
        return f.read().lower().strip()

# -------- Whisper (NO FFmpeg) --------
def whisper_stt(path):
    audio, sr = sf.read(path)

    # Convert stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 🔴 CRITICAL FIX: float64 → float32
    audio = audio.astype(np.float32)

    result = whisper_model.transcribe(audio, fp16=False)
    return result["text"].lower().strip()

# -------- Faster-Whisper --------
def faster_whisper_stt(path):
    segments, _ = faster_model.transcribe(path)
    return " ".join(seg.text for seg in segments).lower().strip()

# -------- Vosk --------
def vosk_stt(path):
    wf = wave.open(path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    text = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            text += json.loads(rec.Result()).get("text", " ")

    text += json.loads(rec.FinalResult()).get("text", "")
    return text.lower().strip()

# =========================
# Benchmarking
# =========================
scores = {
    "Whisper": [],
    "Faster-Whisper": [],
    "Vosk": []
}

print("Starting benchmarking...\n")

for audio_file in os.listdir(AUDIO_DIR):
    if not audio_file.endswith(".wav"):
        continue

    print(f"Processing {audio_file}...")

    audio_path = os.path.join(AUDIO_DIR, audio_file)
    ref_text = load_reference(audio_file.replace(".wav", ".txt"))

    whisper_text = whisper_stt(audio_path)
    faster_text = faster_whisper_stt(audio_path)
    vosk_text = vosk_stt(audio_path)

    scores["Whisper"].append(wer(ref_text, whisper_text))
    scores["Faster-Whisper"].append(wer(ref_text, faster_text))
    scores["Vosk"].append(wer(ref_text, vosk_text))

# =========================
# Results
# =========================
print("\n=== AVERAGE WORD ERROR RATE (WER) ===")

avg_results = {}
for model, values in scores.items():
    avg = sum(values) / len(values)
    avg_results[model] = avg
    print(f"{model}: {avg:.4f}")

winner = min(avg_results, key=avg_results.get)
print(f"\n🏆 Best Model Based on WER: {winner}")
