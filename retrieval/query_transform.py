"""Query transformation.

Three techniques, all optional and composable:

1. REWRITE — clean up conversational queries into search-friendly form.
   "btw how do i cancel my plan" -> "How to cancel a subscription"

2. MULTI-QUERY EXPANSION — generate N paraphrases of the question and
   search all of them. Fixes the common failure where the user's phrasing
   is far from how docs are written.
   "Why did my auto run twice?" -> also try:
     "Duplicate automation execution"
     "Automation triggered multiple times on one event"

3. HyDE (Hypothetical Document Embeddings) — ask the LLM to HALLUCINATE
   an ideal answer, then embed *that* instead of the query. Works well
   when the query is short/vague and docs are long/dense. We gate it on
   query length so we don't pay the LLM tax on every call.

All three use the same LLM router we'll build in Step 4; for Step 3 we
expose a small adapter interface so retrieval can be tested in isolation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


class LLMAdapter(Protocol):
    """Minimal interface retrieval needs from the LLM layer."""
    async def complete(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.2) -> str: ...


@dataclass
class TransformedQuery:
    original: str
    rewritten: str              # the primary query text for search
    expansions: list[str]       # additional queries to run in parallel
    hyde_doc: str | None        # hallucinated doc for embedding-side search


_REWRITE_PROMPT = """Rewrite the user's support question as a concise search query.
Keep product names, error codes, and technical terms verbatim.
Return ONLY the rewritten query, no explanation.

User question: {q}
Rewritten query:"""


_EXPAND_PROMPT = """Generate {n} alternative phrasings of this support question.
Each phrasing should use different words but preserve the meaning.
Include domain synonyms (e.g. "cancel" <-> "terminate", "SSO" <-> "single sign-on").
Return exactly {n} lines, no numbering, no explanation.

Question: {q}"""


_HYDE_PROMPT = """Write a short, factual paragraph that would appear in product
documentation answering this question. Be specific and use concrete terms
the docs would use. 3-4 sentences maximum.

Question: {q}

Documentation paragraph:"""


class QueryTransformer:
    def __init__(
        self,
        llm: LLMAdapter | None = None,
        *,
        rewrite: bool = True,
        expansions: int = 2,
        use_hyde: bool = False,
        hyde_min_query_words: int = 4,
    ):
        self.llm = llm
        self.rewrite = rewrite
        self.expansions = expansions
        self.use_hyde = use_hyde
        self.hyde_min_query_words = hyde_min_query_words

    async def transform(self, query: str) -> TransformedQuery:
        query = query.strip()
        # No LLM available (dev mode, eval without keys) -> pass-through
        if self.llm is None:
            return TransformedQuery(
                original=query, rewritten=query, expansions=[], hyde_doc=None
            )

        rewritten = query
        if self.rewrite:
            try:
                out = await self.llm.complete(
                    _REWRITE_PROMPT.format(q=query), max_tokens=64, temperature=0.0
                )
                rewritten = _first_line(out) or query
            except Exception:
                rewritten = query  # degrade gracefully

        expansions: list[str] = []
        if self.expansions > 0:
            try:
                out = await self.llm.complete(
                    _EXPAND_PROMPT.format(n=self.expansions, q=query),
                    max_tokens=128,
                    temperature=0.4,
                )
                expansions = [
                    line.strip(" -•\t")
                    for line in out.splitlines()
                    if line.strip()
                ][: self.expansions]
            except Exception:
                expansions = []

        hyde_doc: str | None = None
        if self.use_hyde and len(query.split()) >= self.hyde_min_query_words:
            try:
                hyde_doc = (
                    await self.llm.complete(
                        _HYDE_PROMPT.format(q=query), max_tokens=180, temperature=0.3
                    )
                ).strip() or None
            except Exception:
                hyde_doc = None

        return TransformedQuery(
            original=query, rewritten=rewritten, expansions=expansions, hyde_doc=hyde_doc
        )


def _first_line(s: str) -> str:
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return ""
