from __future__ import annotations

import logging
import secrets
from pathlib import Path

import anyio
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile

from .audio import normalize_audio_to_pcm16k_mono
from .config import Settings
from .model_manager import ModelManager
from .schemas import (
    AsrResponse,
    LidResponse,
    ProcessAllResponse,
    PuncRequest,
    PuncResponse,
    VadResponse,
)
from .service import (
    SpeechService,
    create_temp_upload_path,
    persist_upload_file,
    remove_file_quietly,
)

settings = Settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("firered-api")

app = FastAPI(
    title="FireRedASR2S API",
    version="1.0.0",
    description="Industrial speech API with ASR, VAD, LID and punctuation prediction.",
)

manager = ModelManager(settings)
service = SpeechService(manager)
upload_tmp_dir = Path("/tmp/firered-api-upload")


async def verify_api_key(request: Request) -> None:
    if not settings.api_key_enabled:
        return
    supplied = request.headers.get(settings.api_key_header)
    if not supplied or not secrets.compare_digest(supplied, settings.api_key):
        raise HTTPException(status_code=401, detail="invalid or missing api key")


async def handle_audio_upload(file: UploadFile, callback):
    upload_path = create_temp_upload_path(upload_tmp_dir, file.filename)
    normalized_path: Path | None = None
    try:
        await persist_upload_file(file, upload_path)
        normalized_path = await anyio.to_thread.run_sync(
            normalize_audio_to_pcm16k_mono, upload_path, upload_tmp_dir
        )
        return await callback(normalized_path)
    finally:
        remove_file_quietly(upload_path)
        if normalized_path is not None:
            remove_file_quietly(normalized_path)


@app.on_event("startup")
async def _startup() -> None:
    await manager.start()
    logger.info(
        "Service started. model_path=%s, vram_ttl=%ss, download_mode=%s",
        settings.model_path,
        settings.vram_ttl,
        settings.model_download_mode,
    )
    logger.info(
        "Memory optimization: use_half(global=%s, asr=%s, vad=%s, lid=%s, punc=%s)",
        settings.use_half,
        settings.asr_use_half,
        settings.vad_use_half,
        settings.lid_use_half,
        settings.punc_use_half,
    )
    if settings.api_key_enabled:
        logger.info("API key auth enabled. header=%s", settings.api_key_header)
    else:
        logger.warning("API key auth disabled. set API_KEY to enable authentication.")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await manager.shutdown()


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models/status")
async def model_status(_: None = Depends(verify_api_key)) -> dict[str, object]:
    return {
        "vram_ttl": settings.vram_ttl,
        "download_mode": settings.model_download_mode,
        "models": manager.status(),
    }


@app.post("/v1/asr", response_model=AsrResponse)
async def asr_api(
    file: UploadFile = File(..., description="Audio file"),
    force_refresh: bool = Query(False, description="Force refresh ASR model cache"),
    _: None = Depends(verify_api_key),
) -> AsrResponse:
    try:
        result = await handle_audio_upload(
            file,
            lambda normalized_path: service.asr_only(normalized_path, force_refresh=force_refresh),
        )
        return AsrResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ASR API failed")
        raise HTTPException(status_code=500, detail=f"asr failed: {exc}") from exc


@app.post("/v1/vad", response_model=VadResponse)
async def vad_api(
    file: UploadFile = File(..., description="Audio file"),
    force_refresh: bool = Query(False, description="Force refresh VAD model cache"),
    _: None = Depends(verify_api_key),
) -> VadResponse:
    try:
        result = await handle_audio_upload(
            file,
            lambda normalized_path: service.vad_only(normalized_path, force_refresh=force_refresh),
        )
        return VadResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("VAD API failed")
        raise HTTPException(status_code=500, detail=f"vad failed: {exc}") from exc


@app.post("/v1/lid", response_model=LidResponse)
async def lid_api(
    file: UploadFile = File(..., description="Audio file"),
    force_refresh: bool = Query(False, description="Force refresh LID model cache"),
    _: None = Depends(verify_api_key),
) -> LidResponse:
    try:
        result = await handle_audio_upload(
            file,
            lambda normalized_path: service.lid_only(normalized_path, force_refresh=force_refresh),
        )
        return LidResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("LID API failed")
        raise HTTPException(status_code=500, detail=f"lid failed: {exc}") from exc


@app.post("/v1/punc", response_model=PuncResponse)
async def punc_api(
    request: PuncRequest,
    force_refresh: bool = Query(False, description="Force refresh Punc model cache"),
    _: None = Depends(verify_api_key),
) -> PuncResponse:
    try:
        result = await service.punc_only(request.text, force_refresh=force_refresh)
        return PuncResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Punc API failed")
        raise HTTPException(status_code=500, detail=f"punc failed: {exc}") from exc


@app.post("/v1/process_all", response_model=ProcessAllResponse)
async def process_all_api(
    file: UploadFile = File(..., description="Audio file"),
    force_refresh: bool = Query(False, description="Force refresh all model caches"),
    _: None = Depends(verify_api_key),
) -> ProcessAllResponse:
    try:
        result = await handle_audio_upload(
            file,
            lambda normalized_path: service.process_all(normalized_path, force_refresh=force_refresh),
        )
        return ProcessAllResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("process_all API failed")
        raise HTTPException(status_code=500, detail=f"process_all failed: {exc}") from exc
