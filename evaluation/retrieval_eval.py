"""Retrieval quality evaluation.

Metrics (all computed at the doc_id level, not chunk level — we care
whether the right document surfaced, not which chunk of it):

  - Hit@K       : fraction of queries where at least one relevant doc is in top-K
  - MRR@K       : mean of 1/rank-of-first-relevant (0 if none)
  - nDCG@K      : normalized discounted cumulative gain
  - Recall@K    : fraction of relevant docs returned among top-K

We run FOUR configurations in one pass so you can see the lift of each
retrieval knob:

  1. vector_only     : disable BM25, no rerank, no transforms
  2. hybrid_no_rerank: vector + BM25 fused via RRF, no rerank
  3. hybrid_rerank   : vector + BM25 + cross-encoder rerank
  4. full            : adds LLM query expansion (only if LLM configured)

Usage:
    python -m evaluation.retrieval_eval
"""
from __future__ import annotations
import asyncio
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from retrieval.retriever import Retriever
from retrieval.hybrid import HybridSearcher
from retrieval.reranker import Reranker
from retrieval.query_transform import QueryTransformer
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store
from ingestion.embedder import Embedder
from observability.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

GOLDEN_PATH = Path("evaluation/datasets/golden_qa.jsonl")


@dataclass
class EvalCase:
    query: str
    relevant_doc_ids: set[str]


def load_cases(path: Path = GOLDEN_PATH) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cases.append(
                EvalCase(
                    query=obj["query"],
                    relevant_doc_ids=set(obj["relevant_doc_ids"]),
                )
            )
    return cases


# ---------- Metrics ----------

def hit_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(d in relevant for d in retrieved_doc_ids[:k]) else 0.0


def mrr_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(retrieved_doc_ids[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    # Binary relevance -> DCG = sum(1 / log2(i+1)) over hits
    dcg = sum(
        (1.0 / math.log2(i + 1))
        for i, d in enumerate(retrieved_doc_ids[:k], start=1)
        if d in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_doc_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved_doc_ids[:k] if d in relevant)
    return hits / len(relevant)


# ---------- Runner ----------

async def evaluate(retriever: Retriever, cases: list[EvalCase], *, k: int = 5) -> dict[str, float]:
    hits, mrrs, ndcgs, recalls = [], [], [], []

    for case in cases:
        result = await retriever.retrieve(case.query, top_k=k, candidate_k=20)
        # Dedup to doc_id level in rank order
        seen: set[str] = set()
        retrieved_docs: list[str] = []
        for chunk in result.chunks:
            if chunk.doc_id not in seen:
                seen.add(chunk.doc_id)
                retrieved_docs.append(chunk.doc_id)

        hits.append(hit_at_k(retrieved_docs, case.relevant_doc_ids, k))
        mrrs.append(mrr_at_k(retrieved_docs, case.relevant_doc_ids, k))
        ndcgs.append(ndcg_at_k(retrieved_docs, case.relevant_doc_ids, k))
        recalls.append(recall_at_k(retrieved_docs, case.relevant_doc_ids, k))

        if hits[-1] == 0:
            log.warning(
                "miss",
                query=case.query,
                expected=list(case.relevant_doc_ids),
                got=retrieved_docs[:k],
            )

    return {
        f"hit@{k}": statistics.mean(hits),
        f"mrr@{k}": statistics.mean(mrrs),
        f"ndcg@{k}": statistics.mean(ndcgs),
        f"recall@{k}": statistics.mean(recalls),
        "n_queries": len(cases),
    }


def _make_retriever(
    *, hybrid: bool, rerank: bool, llm_for_transforms=None
) -> Retriever:
    """Build retriever variants for A/B comparison."""
    embedder = Embedder()
    vs = VectorStore()
    bm25 = BM25Store() if hybrid else None

    # Monkey-patch a null BM25 when we want vector-only: searcher handles None gracefully
    # by returning empty results from bm25, so RRF naturally degrades to vector ranks.
    class _NullBM25:
        def load(self): pass
        def search(self, *a, **kw): return []

    searcher = HybridSearcher(
        embedder=embedder,
        vector_store=vs,
        bm25_store=bm25 if hybrid else _NullBM25(),  # type: ignore[arg-type]
    )
    return Retriever(
        query_transformer=QueryTransformer(
            llm=llm_for_transforms,
            rewrite=llm_for_transforms is not None,
            expansions=2 if llm_for_transforms else 0,
        ),
        hybrid=searcher,
        reranker=Reranker() if rerank else None,
        enable_rerank=rerank,
    )


async def main() -> None:
    cases = load_cases()
    log.info("eval_start", n_cases=len(cases))

    configs = {
        "vector_only": _make_retriever(hybrid=False, rerank=False),
        "hybrid_no_rerank": _make_retriever(hybrid=True, rerank=False),
        "hybrid_rerank": _make_retriever(hybrid=True, rerank=True),
    }

    results: dict[str, dict] = {}
    for name, retriever in configs.items():
        log.info("eval_config_start", name=name)
        results[name] = await evaluate(retriever, cases, k=5)
        log.info("eval_config_done", name=name, **results[name])

    # Pretty print comparison table
    print("\n" + "=" * 80)
    print(f"{'Config':<22} {'Hit@5':>8} {'MRR@5':>8} {'nDCG@5':>8} {'Recall@5':>10}")
    print("-" * 80)
    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m['hit@5']:>8.3f} {m['mrr@5']:>8.3f} "
            f"{m['ndcg@5']:>8.3f} {m['recall@5']:>10.3f}"
        )
    print("=" * 80 + "\n")

    Path("evaluation/results").mkdir(parents=True, exist_ok=True)
    Path("evaluation/results/retrieval_eval.json").write_text(
        json.dumps(results, indent=2)
    )


if __name__ == "__main__":
    asyncio.run(main())
