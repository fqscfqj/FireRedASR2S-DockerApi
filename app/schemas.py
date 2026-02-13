from __future__ import annotations

from pydantic import BaseModel, Field


class AsrWordTimestamp(BaseModel):
    token: str = Field(..., description="Recognized token")
    start_s: float = Field(..., description="Start time in seconds")
    end_s: float = Field(..., description="End time in seconds")


class AsrResponse(BaseModel):
    uttid: str
    text: str
    confidence: float | None = None
    dur_s: float | None = None
    timestamps: list[AsrWordTimestamp] = Field(default_factory=list)


class SentenceResult(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    asr_confidence: float
    lang: str | None = None
    lang_confidence: float = 0.0


class WordResult(BaseModel):
    start_ms: int
    end_ms: int
    text: str


class ProcessAllResponse(BaseModel):
    uttid: str
    text: str
    sentences: list[SentenceResult]
    vad_segments_ms: list[list[int]]
    dur_s: float
    words: list[WordResult]
    wav_path: str


class VadResponse(BaseModel):
    dur_s: float
    timestamps: list[list[float]]
    wav_path: str | None = None


class LidResponse(BaseModel):
    uttid: str
    lang: str
    confidence: float | None = None
    dur_s: float | None = None


class PuncRequest(BaseModel):
    text: str = Field(..., description="Input text without punctuation")


class PuncResponse(BaseModel):
    origin_text: str
    punc_text: str
