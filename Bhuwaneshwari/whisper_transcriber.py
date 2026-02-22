import whisper

def run_whisper_task(audio_path, model_type="base"):
    
    model = whisper.load_model(model_type)
    result = model.transcribe(audio_path)
    return result