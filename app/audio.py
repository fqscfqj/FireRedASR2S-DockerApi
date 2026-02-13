from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path


def normalize_audio_to_pcm16k_mono(input_path: Path, work_dir: Path) -> Path:
    """
    Convert arbitrary audio into 16kHz / mono / PCM S16LE wav, which FireRedASR2S expects.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / f"{uuid.uuid4().hex}.wav"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required but not found in PATH.")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise ValueError(f"ffmpeg convert failed: {proc.stderr.strip()}")
    return output_path

