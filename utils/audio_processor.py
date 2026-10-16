import os
from pathlib import Path

import yt_dlp

from utils.ffmpeg import configure_ffmpeg

FFMPEG_EXECUTABLE = configure_ffmpeg()

from pydub import AudioSegment

if FFMPEG_EXECUTABLE:
    AudioSegment.converter = FFMPEG_EXECUTABLE

PROJECT_DIR = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = PROJECT_DIR / "downloades"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def download_youtube_audio(url :str) ->str:
    output_path = str(DOWNLOAD_DIR / "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"
    return filename



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero.")

    audio = AudioSegment.from_wav(wav_path)
    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {wav_path}")
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks

def process_input(source: str) -> list:
    source = source.strip()
    if not source:
        raise ValueError("Enter a YouTube URL or a path to an audio/video file.")

    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
        wav_path = convert_to_wav(wav_path)
    else:
        input_path = Path(source).expanduser()
        if not input_path.is_absolute() and not input_path.exists():
            input_path = PROJECT_DIR / input_path
        if not input_path.is_file():
            raise FileNotFoundError(f"Audio/video file was not found: {source}")
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(str(input_path))

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


