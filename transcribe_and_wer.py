import os
import whisper
import jiwer
import subprocess
import wave
import json
from vosk import Model, KaldiRecognizer

# ---------- PATHS ----------
AUDIO_DIR = "audio"
TRANSCRIPT_DIR = "transcripts"

# ---------- LOAD MODELS ----------
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading Vosk model...")
vosk_model = Model(r"C:\Users\deepi\OneDrive\Desktop\podcast\vosk-model-small-en-us-0.15")

# ---------- VOSK TRANSCRIPTION ----------
def transcribe_vosk(audio_path):
    converted = "temp.wav"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        converted
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    wf = wave.open(converted, "rb")
    rec = KaldiRecognizer(vosk_model, 16000)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        try:
            rec.AcceptWaveform(data)
        except:
            pass

    result = json.loads(rec.FinalResult())
    wf.close()
    os.remove(converted)

    return result.get("text", "").strip()

# ---------- MAIN ----------
def main():
    print("\nStarting transcription...\n")

    for file in os.listdir(AUDIO_DIR):
        if not file.endswith(".wav"):
            continue

        audio_path = os.path.join(AUDIO_DIR, file)
        ref_path = os.path.join(TRANSCRIPT_DIR, file.replace(".wav", ".txt"))

        print("=" * 40)
        print(f"Processing: {file}")

        with open(ref_path, "r", encoding="utf-8") as f:
            reference = f.read().strip()

        whisper_text = whisper_model.transcribe(audio_path)["text"].strip()
        vosk_text = transcribe_vosk(audio_path)

        whisper_wer = jiwer.wer(reference, whisper_text)
        vosk_wer = jiwer.wer(reference, vosk_text)

        print("Whisper Text:", whisper_text[:100], "...")
        print("Vosk Text:", vosk_text[:100], "...")
        print(f"Whisper WER: {whisper_wer:.2f}")
        print(f"Vosk WER: {vosk_wer:.2f}")

    print("\nDONE ✅")

if __name__ == "__main__":
    main()
