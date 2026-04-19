"""Pydantic request/response schemas for the API."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[tuple[Literal["user", "assistant"], str]] = Field(default_factory=list)
    source_types: list[str] | None = None
    use_cache: bool = True


class CitationOut(BaseModel):
    marker: int
    title: str
    url: str
    source_type: str
    snippet: str


class ChatResponse(BaseModel):
    trace_id: str
    answer: str
    citations: list[CitationOut]
    cache_hit: bool
    invented_citations: list[int]
    coverage: float
    timings_ms: dict[str, float]


class IngestResponse(BaseModel):
    new: int
    updated: int
    unchanged: int
    deleted: int
    chunks_written: int
    errors: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant: bool
    redis: bool
    providers: list[str]
