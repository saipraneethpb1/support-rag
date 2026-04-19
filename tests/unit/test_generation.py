"""Tests for generation layer: citation audit, prompt building, LLM router fallback."""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import AsyncIterator
from datetime import datetime, timezone

import pytest

from generation.citation import audit_citations
from generation.prompt_builder import Citation, PromptBuilder
from generation.llm_router import (
    LLMRouter, LLMProvider, LLMProviderError, RateLimitError, NoProvidersAvailableError
)
from retrieval.retriever import RetrievedChunk


# ---------- citation.audit_citations ----------

def _cites(n: int) -> list[Citation]:
    return [
        Citation(marker=i, chunk_id=f"c{i}", doc_id=f"d{i}", title=f"T{i}",
                 url=f"https://ex.com/{i}", source_type="markdown_docs", snippet="...")
        for i in range(1, n + 1)
    ]


def test_audit_extracts_valid_markers():
    ans = "You can cancel via Settings [1]. Refunds on annual plans [2]."
    audit = audit_citations(ans, _cites(2))
    assert audit.used_markers == {1, 2}
    assert audit.invented_markers == set()


def test_audit_flags_invented_markers():
    ans = "Cancel via Settings [1]. This feature launched in 2026 [7]."
    audit = audit_citations(ans, _cites(2))
    assert audit.used_markers == {1}
    assert audit.invented_markers == {7}
    # Cleaned answer should remove the invented [7]
    assert "[7]" not in audit.cleaned_answer
    assert "[1]" in audit.cleaned_answer


def test_audit_coverage_is_fraction_of_sentences_with_citation():
    # 3 sentences, 2 with citations
    ans = "First thing [1]. Second thing. Third thing [2]."
    audit = audit_citations(ans, _cites(2))
    assert audit.sentence_coverage == pytest.approx(2 / 3, abs=0.01)


def test_audit_handles_multi_marker_citations():
    ans = "This is supported by several sources [1][2][3]."
    audit = audit_citations(ans, _cites(3))
    assert audit.used_markers == {1, 2, 3}


def test_audit_empty_answer():
    audit = audit_citations("", _cites(2))
    assert audit.used_markers == set()
    assert audit.sentence_coverage == 0.0


# ---------- PromptBuilder ----------

def _chunk(i: int, text: str = "body", title: str = "Doc") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{i}", doc_id=f"d{i}", text=text, title=title,
        url=f"https://ex.com/{i}", source_type="markdown_docs",
        rerank_score=0.9, rrf_score=0.5, retriever_hits={"vec": 1},
    )


def test_prompt_builder_formats_numbered_context():
    pb = PromptBuilder()
    built = pb.build("how do I cancel?", [_chunk(1, "Go to Settings > Billing"), _chunk(2, "Refunds apply within 30 days")])
    assert "[1]" in built.text
    assert "[2]" in built.text
    assert "Settings > Billing" in built.text
    assert len(built.citations) == 2
    assert built.citations[0].marker == 1
    assert built.citations[1].marker == 2


def test_prompt_builder_enforces_token_budget():
    pb = PromptBuilder()
    pb.max_context_tokens = 50  # tiny budget
    big = "word " * 200
    built = pb.build("q?", [_chunk(1, big), _chunk(2, big), _chunk(3, big)])
    # First chunk always included; the rest dropped
    assert built.used_chunks == 1
    assert built.dropped_chunks == 2


def test_prompt_builder_handles_empty_chunks():
    pb = PromptBuilder()
    built = pb.build("anything?", [])
    assert "(no relevant documents found)" in built.text
    assert built.citations == []


# ---------- LLMRouter ----------

class _FakeProvider:
    def __init__(self, name: str, *, fail_times: int = 0, error_cls=LLMProviderError, response: str = "ok"):
        self.name = name
        self._fail_times = fail_times
        self._calls = 0
        self._error_cls = error_cls
        self._response = response

    async def complete(self, prompt, *, max_tokens, temperature):
        self._calls += 1
        if self._calls <= self._fail_times:
            raise self._error_cls(f"{self.name} failure #{self._calls}")
        return f"{self._response} from {self.name}"

    async def stream(self, prompt, *, max_tokens, temperature) -> AsyncIterator[str]:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise self._error_cls(f"{self.name} failure #{self._calls}")
        for tok in ["hello ", "from ", self.name]:
            yield tok


@pytest.mark.asyncio
async def test_router_uses_primary_when_healthy():
    p1 = _FakeProvider("primary")
    p2 = _FakeProvider("secondary")
    router = LLMRouter(providers=[p1, p2], retry_attempts=1)
    out = await router.complete("q")
    assert "primary" in out
    assert p2._calls == 0


@pytest.mark.asyncio
async def test_router_falls_back_when_primary_fails():
    p1 = _FakeProvider("primary", fail_times=5)  # exceed retry attempts
    p2 = _FakeProvider("secondary")
    router = LLMRouter(providers=[p1, p2], retry_attempts=2)
    out = await router.complete("q")
    assert "secondary" in out
    assert p1._calls >= 2  # retried within primary first


@pytest.mark.asyncio
async def test_router_raises_when_all_providers_fail():
    p1 = _FakeProvider("p1", fail_times=10)
    p2 = _FakeProvider("p2", fail_times=10)
    router = LLMRouter(providers=[p1, p2], retry_attempts=1)
    with pytest.raises(NoProvidersAvailableError):
        await router.complete("q")


@pytest.mark.asyncio
async def test_router_circuit_opens_after_threshold():
    p1 = _FakeProvider("p1", fail_times=100)  # always fails
    p2 = _FakeProvider("p2")
    router = LLMRouter(
        providers=[p1, p2],
        retry_attempts=1,
        breaker_threshold=2,
        breaker_cooldown_s=60,
    )
    # First call: p1 fails -> fallback p2. Breaker increments.
    await router.complete("q1")
    # Second call: p1 fails again -> breaker now opens.
    await router.complete("q2")
    calls_before = p1._calls
    # Third call: p1 should be SKIPPED entirely (circuit open)
    await router.complete("q3")
    assert p1._calls == calls_before  # p1 not called again


@pytest.mark.asyncio
async def test_router_streams_from_first_working_provider():
    p1 = _FakeProvider("p1", fail_times=5)
    p2 = _FakeProvider("p2")
    router = LLMRouter(providers=[p1, p2], retry_attempts=1)
    tokens = []
    async for t in router.stream("q"):
        tokens.append(t)
    assert "".join(tokens) == "hello from p2"
