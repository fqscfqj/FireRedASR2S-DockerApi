from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any

import anyio
import soundfile as sf

from .model_manager import ModelManager


class SpeechService:
    _HAN_CHAR_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
    _SPACE_PATTERN = re.compile(r"\s+")

    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager

    @classmethod
    def _normalize_text_spaces(cls, text: str) -> str:
        return cls._SPACE_PATTERN.sub(" ", text).strip()

    @classmethod
    def _is_english_lid(cls, lang: str | None) -> bool:
        lang_norm = (lang or "").strip().lower()
        return lang_norm == "en" or lang_norm.startswith("en ")

    def _should_filter_han_for_segment(self, lid_item: dict[str, Any] | None) -> bool:
        if not self.manager.settings.process_all_filter_script_mismatch:
            return False
        if not lid_item:
            return False
        confidence = float(lid_item.get("confidence") or 0.0)
        if confidence < self.manager.settings.process_all_filter_min_confidence:
            return False
        return self._is_english_lid(lid_item.get("lang"))

    @classmethod
    def _remove_han_chars(cls, text: str) -> str:
        return cls._normalize_text_spaces(cls._HAN_CHAR_PATTERN.sub("", text))

    def _sanitize_asr_item_by_lid(
        self, asr_item: dict[str, Any], lid_item: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not self._should_filter_han_for_segment(lid_item):
            return asr_item

        sanitized = dict(asr_item)
        timestamps = sanitized.get("timestamp") or []
        if timestamps:
            filtered_timestamps = []
            for token, start_s, end_s in timestamps:
                if self._HAN_CHAR_PATTERN.search(str(token)):
                    continue
                filtered_timestamps.append((token, start_s, end_s))
            sanitized["timestamp"] = filtered_timestamps
            sanitized["text"] = self._normalize_text_spaces(
                " ".join(str(token) for token, _, _ in filtered_timestamps)
            )
            if sanitized["text"]:
                return sanitized

        sanitized["text"] = self._remove_han_chars(str(sanitized.get("text", "")))
        return sanitized

    def _sanitize_sentence_text_by_lid(self, text: str, lid_item: dict[str, Any] | None) -> str:
        if not self._should_filter_han_for_segment(lid_item):
            return text
        return self._remove_han_chars(text)

    async def asr_only(self, wav_path: Path, force_refresh: bool = False) -> dict[str, Any]:
        if force_refresh:
            await self.manager.refresh(["asr"])

        uttid = wav_path.stem or uuid.uuid4().hex

        raw_results = await self.manager.run_with_model(
            "asr",
            lambda asr: asr.transcribe([uttid], [str(wav_path)]),
        )
        if not raw_results:
            return {
                "uttid": uttid,
                "text": "",
                "confidence": None,
                "dur_s": None,
                "timestamps": [],
            }

        result = raw_results[0]
        timestamps = []
        for item in result.get("timestamp", []):
            token, start_s, end_s = item
            timestamps.append({"token": token, "start_s": start_s, "end_s": end_s})

        return {
            "uttid": result.get("uttid", uttid),
            "text": result.get("text", ""),
            "confidence": result.get("confidence"),
            "dur_s": result.get("dur_s"),
            "timestamps": timestamps,
        }

    async def vad_only(self, wav_path: Path, force_refresh: bool = False) -> dict[str, Any]:
        if force_refresh:
            await self.manager.refresh(["vad"])
        result = await self.manager.run_with_model("vad", lambda vad: vad.detect(str(wav_path))[0])
        if not result:
            return {"dur_s": 0.0, "timestamps": [], "wav_path": str(wav_path)}
        return {
            "dur_s": result.get("dur", 0.0),
            "timestamps": [[float(s), float(e)] for s, e in result.get("timestamps", [])],
            "wav_path": result.get("wav_path"),
        }

    async def lid_only(self, wav_path: Path, force_refresh: bool = False) -> dict[str, Any]:
        if force_refresh:
            await self.manager.refresh(["lid"])
        uttid = wav_path.stem or uuid.uuid4().hex
        results = await self.manager.run_with_model(
            "lid",
            lambda lid: lid.process([uttid], [str(wav_path)]),
        )
        if not results:
            return {"uttid": uttid, "lang": "", "confidence": None, "dur_s": None}
        result = results[0]
        return {
            "uttid": result.get("uttid", uttid),
            "lang": result.get("lang", ""),
            "confidence": result.get("confidence"),
            "dur_s": result.get("dur_s"),
        }

    async def punc_only(self, text: str, force_refresh: bool = False) -> dict[str, Any]:
        if force_refresh:
            await self.manager.refresh(["punc"])
        text = text.strip()
        if not text:
            return {"origin_text": "", "punc_text": ""}
        results = await self.manager.run_with_model(
            "punc",
            lambda punc: punc.process([text]),
        )
        if not results:
            return {"origin_text": text, "punc_text": text}
        result = results[0]
        return {
            "origin_text": result.get("origin_text", text),
            "punc_text": result.get("punc_text", text),
        }

    async def process_all(self, wav_path: Path, force_refresh: bool = False) -> dict[str, Any]:
        if force_refresh:
            await self.manager.refresh(["vad", "lid", "asr", "punc"])

        uttid = wav_path.stem or uuid.uuid4().hex
        wav_np, sample_rate = await anyio.to_thread.run_sync(
            lambda: sf.read(str(wav_path), dtype="int16")
        )
        if sample_rate != 16000:
            raise ValueError(f"expected 16k sample rate, got {sample_rate}")

        dur = float(wav_np.shape[0] / sample_rate)

        vad_result = await self.manager.run_with_model(
            "vad",
            lambda vad: vad.detect(str(wav_path))[0],
        )
        vad_segments = (vad_result or {}).get("timestamps", [])
        if not vad_segments:
            vad_segments = [(0.0, dur)]

        asr_results: list[dict[str, Any]] = []
        lid_results: list[dict[str, Any]] = []
        for start_s, end_s in vad_segments:
            start_idx = max(0, int(start_s * sample_rate))
            end_idx = min(wav_np.shape[0], int(end_s * sample_rate))
            if end_idx <= start_idx:
                continue

            segment = wav_np[start_idx:end_idx]
            seg_uttid = f"{uttid}_s{int(start_s * 1000)}_e{int(end_s * 1000)}"
            batch_uttid = [seg_uttid]
            batch_wav = [(sample_rate, segment)]

            batch_asr = await self.manager.run_with_model(
                "asr",
                lambda asr: asr.transcribe(batch_uttid, batch_wav),
            )
            batch_lid = await self.manager.run_with_model(
                "lid",
                lambda lid: lid.process(batch_uttid, batch_wav),
            )

            for asr_item, lid_item in zip(batch_asr, batch_lid):
                if re.search(r"(<blank>)|(<sil>)", asr_item.get("text", "")):
                    continue
                sanitized_asr_item = self._sanitize_asr_item_by_lid(asr_item, lid_item)
                if not sanitized_asr_item.get("text") and not sanitized_asr_item.get("timestamp"):
                    continue
                asr_results.append(sanitized_asr_item)
                lid_results.append(lid_item)

        punc_results: list[dict[str, Any]] = []
        for asr_item in asr_results:
            if asr_item.get("timestamp"):
                batch_punc = await self.manager.run_with_model(
                    "punc",
                    lambda punc: punc.process_with_timestamp(
                        [asr_item["timestamp"]], [asr_item["uttid"]]
                    ),
                )
            else:
                batch_punc = await self.manager.run_with_model(
                    "punc",
                    lambda punc: punc.process([asr_item.get("text", "")], [asr_item["uttid"]]),
                )
            punc_results.extend(batch_punc)

        sentences: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []

        for asr_item, punc_item, lid_item in zip(asr_results, punc_results, lid_results):
            seg_uttid = asr_item["uttid"]
            start_ms_str, end_ms_str = seg_uttid.split("_")[-2:]
            start_ms = int(start_ms_str.lstrip("s"))
            end_ms = int(end_ms_str.lstrip("e"))

            punc_sentences = punc_item.get("punc_sentences")
            if punc_sentences:
                for i, sent in enumerate(punc_sentences):
                    start = start_ms + int(sent["start_s"] * 1000)
                    end = start_ms + int(sent["end_s"] * 1000)
                    if i == 0:
                        start = start_ms
                    if i == len(punc_sentences) - 1:
                        end = end_ms
                    sentence_text = self._sanitize_sentence_text_by_lid(
                        sent["punc_text"], lid_item
                    )
                    if not sentence_text:
                        continue
                    sentences.append(
                        {
                            "start_ms": start,
                            "end_ms": end,
                            "text": sentence_text,
                            "asr_confidence": asr_item["confidence"],
                            "lang": lid_item.get("lang"),
                            "lang_confidence": lid_item.get("confidence", 0.0),
                        }
                    )
            else:
                sentence_text = self._sanitize_sentence_text_by_lid(
                    punc_item.get("punc_text", asr_item.get("text", "")), lid_item
                )
                if not sentence_text:
                    continue
                sentences.append(
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": sentence_text,
                        "asr_confidence": asr_item["confidence"],
                        "lang": lid_item.get("lang"),
                        "lang_confidence": lid_item.get("confidence", 0.0),
                    }
                )

            for token, ts_start, ts_end in asr_item.get("timestamp", []):
                words.append(
                    {
                        "start_ms": int(ts_start * 1000) + start_ms,
                        "end_ms": int(ts_end * 1000) + start_ms,
                        "text": token,
                    }
                )

        return {
            "uttid": uttid,
            "text": "".join(x["text"] for x in sentences),
            "sentences": sentences,
            "vad_segments_ms": [[int(s * 1000), int(e * 1000)] for s, e in vad_segments],
            "dur_s": round(dur, 3),
            "words": words,
            "wav_path": str(wav_path),
        }


async def persist_upload_file(upload, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    async with await anyio.open_file(target_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            await f.write(chunk)
    await upload.close()


def create_temp_upload_path(work_dir: Path, filename: str | None) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix if filename else ".bin"
    return work_dir / f"{uuid.uuid4().hex}{ext}"


def remove_file_quietly(path: Path) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
