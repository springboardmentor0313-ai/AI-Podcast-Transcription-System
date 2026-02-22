from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token="YOUR_HF_TOKEN"
)

def detect_speakers(audio_path):
    diarization = pipeline(audio_path)

    speakers = set()
    segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })

    return len(speakers), segments
