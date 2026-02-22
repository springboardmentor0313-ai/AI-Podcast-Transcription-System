from vosk import Model, KaldiRecognizer
import wave, json, jiwer

wf = wave.open("audio/sample_16k.wav", "rb")


model = Model("vosk-model")
rec = KaldiRecognizer(model, wf.getframerate())

text = ""
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        text += " " + json.loads(rec.Result())["text"]

text += " " + json.loads(rec.FinalResult())["text"]
text = text.strip()

print(text)

with open("vosk_output.txt", "w", encoding="utf-8") as f:
    f.write(text)

reference = "Hello everyone welcome to our podcast artificial intelligence and machine learning"
wer = jiwer.wer(reference, text)

print(f"Word Error Rate: {wer:.2f}")
