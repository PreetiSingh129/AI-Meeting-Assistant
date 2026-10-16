"""Locate FFmpeg and make it available to pydub."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def configure_ffmpeg() -> str | None:
    """Return an FFmpeg executable path, honoring FFMPEG_PATH when provided.

    FFMPEG_PATH may name either the executable itself or its containing
    directory. On Windows, the common Downloads installation layout is also
    searched as a convenience when FFmpeg is not already on PATH.
    """
    configured_path = os.getenv("FFMPEG_PATH")
    if configured_path:
        configured = Path(configured_path).expanduser()
        if configured.is_dir():
            os.environ["PATH"] = str(configured) + os.pathsep + os.environ.get("PATH", "")
            executable = shutil.which("ffmpeg")
            if executable:
                return executable
        elif configured.is_file():
            os.environ["PATH"] = str(configured.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(configured)

    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    if os.name == "nt":
        candidates = Path.home().glob("Downloads/ffmpeg*/**/ffmpeg.exe")
        for candidate in candidates:
            if candidate.is_file():
                os.environ["PATH"] = str(candidate.parent) + os.pathsep + os.environ.get("PATH", "")
                return str(candidate)

    return None
