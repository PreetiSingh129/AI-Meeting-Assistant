import os
import requests
import whisper
from utils.ffmpeg import configure_ffmpeg


FFMPEG_EXECUTABLE = configure_ffmpeg()

from pydub import AudioSegment

if FFMPEG_EXECUTABLE:
    AudioSegment.converter = FFMPEG_EXECUTABLE


# Sarvam's sync STT API rejects audio longer than 30s. We split each chunk
# into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
_whisper_model = None

def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading local Whisper model: {WHISPER_MODEL} ...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Use Whisper's translation task so non-English speech is returned in English."""
    result = load_whisper_model().transcribe(chunk_path, task="translate")
    return result["text"].strip()


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam for Hindi/Hinglish transcription."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY is not set in the environment or .env file.")

    headers = {"api-subscription-key": api_key}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {
            "model": SARVAM_MODEL,
            "language_code": "hi-IN",
            "with_diarization": "false",
        }
        response = requests.post(
            SARVAM_STT_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

   



def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk by desired output language: Whisper translates to English;
    Sarvam returns Hindi/Hinglish speech without translating it to English.
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    if not chunks:
        raise ValueError("No audio chunks were supplied for transcription.")

    full_transcript = "" 

    engine = "Sarvam AI" if language.lower() == "hinglish" else "local Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)  

        full_transcript += text + " "  

    print("Transcription complete.")

    return full_transcript.strip()
