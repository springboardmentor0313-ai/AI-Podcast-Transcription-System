from flask import Flask, render_template, request, jsonify
import whisper
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = "audio"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = whisper.load_model("tiny")

def generate_title(text):
    words = text.split()
    return " ".join(words[:4]).title()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_audio():
    file = request.files["audio"]
    audio_path = os.path.join(UPLOAD_FOLDER, "uploaded.wav")
    file.save(audio_path)

    result = model.transcribe(audio_path, word_timestamps=True)

    segments = []
    buffer_text = ""
    start_time = result["segments"][0]["start"]

    for i, seg in enumerate(result["segments"]):
        buffer_text += " " + seg["text"]

        if (i + 1) % 3 == 0 or i == len(result["segments"]) - 1:
            end_time = seg["end"]
            segments.append({
                "title": generate_title(buffer_text),
                "start": round(start_time),
                "end": round(end_time),
                "summary": buffer_text.strip(),
                "text": buffer_text.strip()
            })
            buffer_text = ""
            if i + 1 < len(result["segments"]):
                start_time = result["segments"][i + 1]["start"]

    output_path = os.path.join(OUTPUT_FOLDER, "podcast.json")
    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)

    return jsonify(segments)

if __name__ == "__main__":
    app.run(debug=True)
