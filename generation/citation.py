"""Citation extraction & validation.

The LLM is instructed to cite claims with [1], [2], etc., using only
markers present in the context. Validation checks:

  - EXTRACT: all [N] markers the LLM produced
  - VALIDATE: every cited N is in 1..len(context_chunks)
  - COVERAGE: fraction of sentences that carry at least one citation
  - INVENTED: markers the LLM produced that weren't in the context
    (strong hallucination signal — log, alert, surface in eval)

We return both the parsed citation set and a cleaned-up answer where
invented markers are stripped. In production you'd also return
"hallucination_suspected" so the UI can flag the answer.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

from generation.prompt_builder import Citation


_CITE_PATTERN = re.compile(r"\[(\d+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")


@dataclass
class CitationAudit:
    used_markers: set[int]
    invented_markers: set[int]
    sentence_coverage: float       # fraction of sentences with >=1 citation
    cleaned_answer: str            # answer with invented markers removed
    used_citations: list[Citation] # Citation objects the answer actually referenced


def audit_citations(answer: str, context_citations: list[Citation]) -> CitationAudit:
    valid_markers = {c.marker for c in context_citations}
    marker_to_cite = {c.marker: c for c in context_citations}

    found = [int(m) for m in _CITE_PATTERN.findall(answer)]
    used = {m for m in found if m in valid_markers}
    invented = {m for m in found if m not in valid_markers}

    cleaned = _CITE_PATTERN.sub(
        lambda m: m.group(0) if int(m.group(1)) in valid_markers else "",
        answer,
    )
    # Tidy up any double spaces left behind
    cleaned = re.sub(r" +", " ", cleaned).strip()

    # Sentence-level coverage
    sentences = [s for s in _SENTENCE_END.split(answer.strip()) if s.strip()]
    if sentences:
        with_cite = sum(1 for s in sentences if _CITE_PATTERN.search(s))
        coverage = with_cite / len(sentences)
    else:
        coverage = 0.0

    used_citations = [marker_to_cite[m] for m in sorted(used)]

    return CitationAudit(
        used_markers=used,
        invented_markers=invented,
        sentence_coverage=coverage,
        cleaned_answer=cleaned,
        used_citations=used_citations,
    )
