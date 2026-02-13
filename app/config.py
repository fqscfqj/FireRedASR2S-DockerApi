from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", "/models"))
    vram_ttl: int = int(os.getenv("VRAM_TTL", "300"))
    model_download_mode: str = os.getenv("MODEL_DOWNLOAD_MODE", "lazy").strip().lower()
    firered_repo_url: str = os.getenv(
        "FIRERED_REPO_URL", "https://github.com/FireRedTeam/FireRedASR2S"
    )
    firered_repo_dir: Path = Path(os.getenv("FIRERED_REPO_DIR", "/opt/FireRedASR2S"))
    asr_type: str = os.getenv("ASR_TYPE", "aed")
    use_half: bool = _as_bool(os.getenv("USE_HALF"), default=False)
    asr_use_half: bool = _as_bool(os.getenv("ASR_USE_HALF", os.getenv("USE_HALF")), default=False)
    vad_use_half: bool = _as_bool(os.getenv("VAD_USE_HALF", os.getenv("USE_HALF")), default=False)
    lid_use_half: bool = _as_bool(os.getenv("LID_USE_HALF", os.getenv("USE_HALF")), default=False)
    punc_use_half: bool = _as_bool(os.getenv("PUNC_USE_HALF", os.getenv("USE_HALF")), default=False)
    asr_beam_size: int = int(os.getenv("ASR_BEAM_SIZE", "3"))
    asr_batch_size: int = int(os.getenv("ASR_BATCH_SIZE", "1"))
    punc_batch_size: int = int(os.getenv("PUNC_BATCH_SIZE", "1"))
    process_all_filter_script_mismatch: bool = _as_bool(
        os.getenv("PROCESS_ALL_FILTER_SCRIPT_MISMATCH"), default=True
    )
    process_all_filter_min_confidence: float = float(
        os.getenv("PROCESS_ALL_FILTER_MIN_CONFIDENCE", "0.80")
    )
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    auto_clone_firered: bool = _as_bool(os.getenv("AUTO_CLONE_FIRERED"), default=True)
    api_key: str = os.getenv("API_KEY", "")
    api_key_header: str = os.getenv("API_KEY_HEADER", "X-API-Key")

    @property
    def startup_download_enabled(self) -> bool:
        return self.model_download_mode == "startup"

    @property
    def api_key_enabled(self) -> bool:
        return bool(self.api_key)
