"""Prompt builder.

Responsibilities:
  - Load prompt templates from disk (so they're editable without code changes).
  - Format retrieved chunks into a numbered CONTEXT block, each with a [N]
    marker the LLM cites and a title/URL we can echo in the UI.
  - Enforce a token budget: if the context would exceed the budget, drop
    lowest-ranked chunks first (the reranker already ordered them for us).
  - Return (prompt, citations_map) so the generator can validate the
    answer's citations post-hoc.

Token budgeting: we approximate tokens as words * 1.3, same heuristic used
in the chunker. Close enough for budgeting — the LLM will hard-truncate
at its own limits if we overshoot slightly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from retrieval.retriever import RetrievedChunk


@dataclass
class Citation:
    marker: int                 # 1-indexed [N] number
    chunk_id: str
    doc_id: str
    title: str
    url: str
    source_type: str
    snippet: str                # short preview for UI display


@dataclass
class BuiltPrompt:
    text: str
    citations: list[Citation]
    used_chunks: int
    dropped_chunks: int


@dataclass
class PromptBuilder:
    prompts_dir: Path = field(default_factory=lambda: Path("config/prompts"))
    max_context_tokens: int = 3500  # leaves room for question + history + answer

    def __post_init__(self) -> None:
        self._system = (self.prompts_dir / "system.txt").read_text(encoding="utf-8").strip()
        self._template = (self.prompts_dir / "answer.txt").read_text(encoding="utf-8")

    def build(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[tuple[str, str]] | None = None,
    ) -> BuiltPrompt:
        citations: list[Citation] = []
        context_parts: list[str] = []
        budget = self.max_context_tokens
        used = 0
        dropped = 0

        for idx, ch in enumerate(chunks, start=1):
            block = self._format_chunk_block(idx, ch)
            block_tokens = _approx_tokens(block)
            if block_tokens > budget and used > 0:
                # Over budget, and we already have at least one chunk — drop the rest
                dropped = len(chunks) - (idx - 1)
                break
            context_parts.append(block)
            citations.append(
                Citation(
                    marker=idx,
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    title=ch.title,
                    url=ch.url,
                    source_type=ch.source_type,
                    snippet=_snippet(ch.text, 200),
                )
            )
            budget -= block_tokens
            used += 1

        context_str = "\n\n".join(context_parts) if context_parts else "(no relevant documents found)"
        history_str = _format_history(history or [])

        text = self._template.format(
            system=self._system,
            context=context_str,
            history=history_str or "(start of conversation)",
            question=question.strip(),
        )
        return BuiltPrompt(
            text=text, citations=citations, used_chunks=used, dropped_chunks=dropped
        )

    def _format_chunk_block(self, marker: int, ch: RetrievedChunk) -> str:
        header = f"[{marker}] {ch.title}" + (f" — {ch.source_type}" if ch.source_type else "")
        return f"{header}\n{ch.text.strip()}"


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _snippet(text: str, max_chars: int) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "…"


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for role, msg in history[-6:]:  # last 3 turns
        lines.append(f"{role.upper()}: {msg.strip()}")
    return "\n".join(lines)
