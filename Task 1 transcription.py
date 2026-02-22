import whisper

# Load Whisper model
model = whisper.load_model("base")

# Audio file path
audio_path = r"C:\Users\Admin\Audios\test.mp3"

# Transcribe audio
result = model.transcribe(audio_path)

# Save transcription to text file
output_file = "test_transcription.txt"

with open(output_file, "w", encoding="utf-8") as file:
    file.write(result["text"])

print(f"✅ Transcription completed. Output saved in {output_file}")

