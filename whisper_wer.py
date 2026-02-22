import whisper
import jiwer

model = whisper.load_model("base")
result = model.transcribe("audio/sample.wav")

transcription = result["text"].strip()
print(transcription)

with open("whisper_output.txt", "w", encoding="utf-8") as f:f.write(transcription)

reference = "Hello everyone welcome to our podcast artificial intelligence and machine learning"
error = jiwer.wer(reference, transcription)
print(f"Word Error Rate: {error:.2f}")