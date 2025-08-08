import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe(audio_path):
    with open(audio_path, "rb") as f:
        # Using OpenAI Whisper endpoint (if available)
        resp = openai.Audio.transcriptions.create(file=f, model="whisper-1")
        # resp may look like {'text': '...'}
        return resp.get("text", "")
