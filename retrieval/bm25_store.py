"""BM25 keyword index.

Why we need this: vector search misses exact-token queries (error codes,
API names, product feature names) more often than people expect. BM25
catches them. We fuse the two with RRF at query time.

Implementation: rank_bm25 in-process, persisted to disk as a pickle.
For >1M chunks you'd want OpenSearch or Tantivy; for our scale this is fine.
"""
from __future__ import annotations
import pickle
import re
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi

from observability.logger import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    def __init__(self, persist_path: str | Path = "data/registry/bm25.pkl"):
        self.path = Path(persist_path)
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._payloads: list[dict] = []

    def load(self) -> None:
        if self.path.exists():
            with self.path.open("rb") as f:
                state = pickle.load(f)
            self._bm25 = state["bm25"]
            self._chunk_ids = state["chunk_ids"]
            self._payloads = state["payloads"]
            log.info("bm25_loaded", chunks=len(self._chunk_ids))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            pickle.dump(
                {"bm25": self._bm25, "chunk_ids": self._chunk_ids, "payloads": self._payloads},
                f,
            )

    def rebuild(self, items: Iterable[tuple[str, str, dict]]) -> None:
        """items: iterable of (chunk_id, text, payload)."""
        items = list(items)
        self._chunk_ids = [i[0] for i in items]
        self._payloads = [i[2] for i in items]
        tokenized = [_tokenize(i[1]) for i in items]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        log.info("bm25_rebuilt", chunks=len(self._chunk_ids))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self._bm25:
            return []
        toks = _tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"score": float(scores[i]), **self._payloads[i]}
            for i in ranked
            if scores[i] > 0
        ]
