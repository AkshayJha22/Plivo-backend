# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import tempfile

# load .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set in environment")

# OpenAI client (old/new SDK differences exist; this uses `openai` package)
import openai
openai.api_key = OPENAI_API_KEY

app = FastAPI()

# Allow local dev calls from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/summarize")
async def summarize(text: str = Form(...)):
    """
    Accepts a 'text' form field and returns a concise summary using ChatGPT.
    (Called with a form POST or you can change to JSON body easily)
    """
    try:
        # Keep prompts simple. Use a compact model to save tokens if needed.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Summarize the user's text into 4 bullet points."},
                {"role": "user", "content": text}
            ],
            max_tokens=300,
            temperature=0.2
        )
        summary = response.choices[0].message['content'].strip()
        return {"summary": summary}
    except Exception as e:
        import traceback
        traceback.print_exc()  # print full error in your uvicorn console
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )


@app.post("/transcribe-diarize")
async def transcribe_diarize(file: UploadFile = File(...)):
    """
    Minimal proof-of-concept:
      - Save uploaded file temporarily
      - Send it to OpenAI's speech->text (whisper) endpoint for transcription
      - Return transcript and a placeholder diarization (chunk-level)
    NOTE: This is intentionally simple and robust for quick testing.
    """
    try:
        # save uploaded file to temp
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        # Call OpenAI speech-to-text (Whisper) if available in your SDK
        with open(tmp_path, "rb") as audio_file:
            # The exact method name may vary across openai-py versions.
            # This is the common pattern:
            transcription = openai.Audio.transcriptions.create(
                file=audio_file,
                model="whisper-1"
            )
            transcript_text = transcription.get("text", transcription.get("transcript", "") )

        # For now we return the full transcript and a simple placeholder diarization:
        # (You can replace this with a real diarization step later)
        diarization = [
            {"speaker": "speaker_1", "start": 0.0, "end": 3.0, "text": "<segment placeholder>"},
            {"speaker": "speaker_2", "start": 3.0, "end": 8.0, "text": "<segment placeholder>"}
        ]

        # cleanup temp file
        try:
            os.remove(tmp_path)
        except:
            pass

        return {"transcript": transcript_text, "diarization": diarization}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
