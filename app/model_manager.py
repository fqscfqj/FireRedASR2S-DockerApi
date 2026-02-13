from __future__ import annotations

import asyncio
import gc
import inspect
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anyio
import torch

from .config import Settings
from .firered_bootstrap import ensure_firered_source

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMeta:
    name: str
    model_id: str
    local_dir: Path
    required_file: str

    @property
    def ready(self) -> bool:
        return (self.local_dir / self.required_file).exists()


@dataclass
class ModelSlot:
    name: str
    loader: Callable[[], Any]
    instance: Any | None = None
    last_used: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    download_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cuda_available = torch.cuda.is_available()
        self._stop_event = asyncio.Event()
        self._cleanup_task: asyncio.Task[Any] | None = None
        self._cleanup_interval = max(10, min(60, settings.vram_ttl // 3 if settings.vram_ttl > 0 else 10))

        model_root = settings.model_path
        self._meta: dict[str, ModelMeta] = {
            "asr": ModelMeta(
                name="asr",
                model_id="xukaituo/FireRedASR2-AED",
                local_dir=model_root / "FireRedASR2-AED",
                required_file="model.pth.tar",
            ),
            "vad": ModelMeta(
                name="vad",
                model_id="xukaituo/FireRedVAD",
                local_dir=model_root / "FireRedVAD",
                required_file="VAD/model.pth.tar",
            ),
            "lid": ModelMeta(
                name="lid",
                model_id="xukaituo/FireRedLID",
                local_dir=model_root / "FireRedLID",
                required_file="model.pth.tar",
            ),
            "punc": ModelMeta(
                name="punc",
                model_id="xukaituo/FireRedPunc",
                local_dir=model_root / "FireRedPunc",
                required_file="model.pth.tar",
            ),
        }

        self._slots: dict[str, ModelSlot] = {
            "asr": ModelSlot(name="asr", loader=self._load_asr),
            "vad": ModelSlot(name="vad", loader=self._load_vad),
            "lid": ModelSlot(name="lid", loader=self._load_lid),
            "punc": ModelSlot(name="punc", loader=self._load_punc),
        }

    async def start(self) -> None:
        self.settings.model_path.mkdir(parents=True, exist_ok=True)
        if self.settings.startup_download_enabled:
            await self.download_models(["asr", "vad", "lid", "punc"])
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="vram-ttl-cleanup")

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._cleanup_task is not None:
            await self._cleanup_task
        await self.refresh(["asr", "vad", "lid", "punc"])

    async def run_with_model(
        self,
        model_name: str,
        runner: Callable[[Any], Any],
    ) -> Any:
        slot = self._slots[model_name]
        async with slot.lock:
            if slot.instance is None:
                await self.ensure_model_downloaded(model_name)
                logger.info("Loading model %s", model_name)
                slot.instance = await anyio.to_thread.run_sync(slot.loader)
            slot.last_used = time.monotonic()
            result = await anyio.to_thread.run_sync(lambda: runner(slot.instance))
            slot.last_used = time.monotonic()
            return result

    async def refresh(self, model_names: list[str]) -> None:
        for model_name in model_names:
            slot = self._slots[model_name]
            async with slot.lock:
                self._unload_locked(slot)

    async def ensure_model_downloaded(self, model_name: str) -> None:
        meta = self._meta[model_name]
        if meta.ready:
            return
        slot = self._slots[model_name]
        async with slot.download_lock:
            if meta.ready:
                return
            await anyio.to_thread.run_sync(self._download_model_sync, meta)

    async def download_models(self, model_names: list[str]) -> None:
        for model_name in model_names:
            await self.ensure_model_downloaded(model_name)

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        data: dict[str, Any] = {}
        for name, slot in self._slots.items():
            meta = self._meta[name]
            idle = now - slot.last_used
            data[name] = {
                "loaded": slot.instance is not None,
                "downloaded": meta.ready,
                "local_dir": str(meta.local_dir),
                "idle_seconds": round(idle, 2),
            }
        return data

    async def _cleanup_loop(self) -> None:
        if self.settings.vram_ttl <= 0:
            return
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._cleanup_interval)
            except TimeoutError:
                pass

            now = time.monotonic()
            for slot in self._slots.values():
                if slot.instance is None:
                    continue
                if now - slot.last_used < self.settings.vram_ttl:
                    continue
                async with slot.lock:
                    if slot.instance is None:
                        continue
                    if now - slot.last_used < self.settings.vram_ttl:
                        continue
                    logger.info(
                        "TTL reached for model %s (idle %.1fs), unloading from GPU/VRAM",
                        slot.name,
                        now - slot.last_used,
                    )
                    self._unload_locked(slot)

    def _download_model_sync(self, meta: ModelMeta) -> None:
        logger.info("Downloading %s from ModelScope: %s", meta.name, meta.model_id)
        meta.local_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except Exception as exc:
            raise RuntimeError(
                "modelscope is required. Install dependency `modelscope` first."
            ) from exc

        try:
            snapshot_download(model_id=meta.model_id, local_dir=str(meta.local_dir))
        except TypeError:
            snapshot_download(meta.model_id, local_dir=str(meta.local_dir))

        if not meta.ready:
            raise RuntimeError(
                f"Model download seems incomplete for {meta.name}: missing {meta.required_file}"
            )
        logger.info("Model %s downloaded into %s", meta.name, meta.local_dir)

    def _unload_locked(self, slot: ModelSlot) -> None:
        if slot.instance is None:
            return
        self._move_to_cpu(slot.instance)
        slot.instance = None
        gc.collect()
        if self.cuda_available:
            torch.cuda.empty_cache()
        slot.last_used = time.monotonic()

    def _move_to_cpu(self, model_instance: Any) -> None:
        # FireRed modules expose different torch module attributes.
        for attr in ("model", "elm", "vad_model"):
            module = getattr(model_instance, attr, None)
            if module is not None and hasattr(module, "cpu"):
                try:
                    module.cpu()
                except Exception:
                    logger.exception("Failed to move %s to CPU", attr)

    def _prepare_imports(self) -> None:
        ensure_firered_source(
            repo_dir=self.settings.firered_repo_dir,
            repo_url=self.settings.firered_repo_url,
            auto_clone=self.settings.auto_clone_firered,
        )

    @staticmethod
    def _build_model_config(config_cls: type[Any], **kwargs: Any) -> Any:
        try:
            sig = inspect.signature(config_cls)
            allowed = {name for name in sig.parameters.keys() if name != "self"}
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            skipped = sorted(set(kwargs.keys()) - set(filtered.keys()))
            if skipped:
                logger.info(
                    "Config %s does not support options %s, skipped",
                    getattr(config_cls, "__name__", str(config_cls)),
                    skipped,
                )
            return config_cls(**filtered)
        except Exception:
            return config_cls(**kwargs)

    def _load_asr(self) -> Any:
        self._prepare_imports()
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

        config = self._build_model_config(
            FireRedAsr2Config,
            use_gpu=self.cuda_available,
            use_half=self.settings.asr_use_half,
            beam_size=self.settings.asr_beam_size,
            return_timestamp=True,
        )
        meta = self._meta["asr"]
        return FireRedAsr2.from_pretrained(
            asr_type=self.settings.asr_type,
            model_dir=str(meta.local_dir),
            config=config,
        )

    def _load_vad(self) -> Any:
        self._prepare_imports()
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

        config = self._build_model_config(
            FireRedVadConfig,
            use_gpu=self.cuda_available,
            use_half=self.settings.vad_use_half,
        )
        model_dir = self._meta["vad"].local_dir / "VAD"
        return FireRedVad.from_pretrained(str(model_dir), config=config)

    def _load_lid(self) -> Any:
        self._prepare_imports()
        from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig

        config = self._build_model_config(
            FireRedLidConfig,
            use_gpu=self.cuda_available,
            use_half=self.settings.lid_use_half,
        )
        model_dir = self._meta["lid"].local_dir
        return FireRedLid.from_pretrained(str(model_dir), config=config)

    def _load_punc(self) -> Any:
        self._prepare_imports()
        from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig

        config = self._build_model_config(
            FireRedPuncConfig,
            use_gpu=self.cuda_available,
            use_half=self.settings.punc_use_half,
        )
        model_dir = self._meta["punc"].local_dir
        return FireRedPunc.from_pretrained(str(model_dir), config=config)
