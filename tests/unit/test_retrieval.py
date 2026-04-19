"""Tests for retrieval components.

Focus on pure logic (RRF fusion, metrics) that doesn't need Qdrant/Redis.
"""
from __future__ import annotations

from retrieval.hybrid import HybridSearcher, RRF_K, Candidate
from evaluation.retrieval_eval import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k


def test_rrf_combines_retrievers_by_rank_not_score():
    """Doc appearing at rank 1 in BOTH retrievers should beat doc at rank 1 in one."""
    searcher = HybridSearcher.__new__(HybridSearcher)  # skip init (no IO)

    results = {
        "vec": [
            {"chunk_id": "A", "text": "a", "score": 0.95},
            {"chunk_id": "B", "text": "b", "score": 0.90},
            {"chunk_id": "C", "text": "c", "score": 0.85},
        ],
        "bm25": [
            {"chunk_id": "A", "text": "a", "score": 50.0},  # A is top in both
            {"chunk_id": "D", "text": "d", "score": 20.0},
            {"chunk_id": "C", "text": "c", "score": 10.0},
        ],
    }
    fused = searcher._rrf_fuse(results)
    assert fused[0].chunk_id == "A"
    # A appears in both -> expected score = 1/(60+1) + 1/(60+1)
    assert abs(fused[0].rrf_score - (2 / (RRF_K + 1))) < 1e-9
    # B appears only in vec at rank 2
    b = next(c for c in fused if c.chunk_id == "B")
    assert abs(b.rrf_score - (1 / (RRF_K + 2))) < 1e-9


def test_rrf_single_retriever_falls_back_cleanly():
    searcher = HybridSearcher.__new__(HybridSearcher)
    fused = searcher._rrf_fuse({"vec": [
        {"chunk_id": "X", "text": "x"},
        {"chunk_id": "Y", "text": "y"},
    ]})
    assert [c.chunk_id for c in fused] == ["X", "Y"]


def test_rrf_handles_empty_retrievers():
    searcher = HybridSearcher.__new__(HybridSearcher)
    fused = searcher._rrf_fuse({"vec": [], "bm25": []})
    assert fused == []


def test_rrf_carries_per_retriever_ranks_for_observability():
    searcher = HybridSearcher.__new__(HybridSearcher)
    fused = searcher._rrf_fuse({
        "vec":  [{"chunk_id": "A", "text": "a"}, {"chunk_id": "B", "text": "b"}],
        "bm25": [{"chunk_id": "B", "text": "b"}, {"chunk_id": "A", "text": "a"}],
    })
    by_id = {c.chunk_id: c for c in fused}
    assert by_id["A"].ranks == {"vec": 1, "bm25": 2}
    assert by_id["B"].ranks == {"vec": 2, "bm25": 1}


# ---------- Metrics ----------

def test_hit_at_k():
    assert hit_at_k(["a", "b", "c"], {"b"}, 5) == 1.0
    assert hit_at_k(["a", "b", "c"], {"z"}, 5) == 0.0
    assert hit_at_k(["a", "b", "c", "z"], {"z"}, 3) == 0.0  # cut off before hit
    assert hit_at_k(["a", "b", "c", "z"], {"z"}, 4) == 1.0


def test_mrr_at_k():
    assert mrr_at_k(["z", "b"], {"z"}, 5) == 1.0
    assert mrr_at_k(["a", "z"], {"z"}, 5) == 0.5
    assert mrr_at_k(["a", "b"], {"z"}, 5) == 0.0


def test_ndcg_at_k_monotonic_in_rank_position():
    # Same relevant doc, earlier rank -> higher nDCG
    higher = ndcg_at_k(["z", "a", "b"], {"z"}, 5)
    lower = ndcg_at_k(["a", "b", "z"], {"z"}, 5)
    assert higher > lower
    # Perfect ranking yields 1.0
    assert ndcg_at_k(["z"], {"z"}, 5) == 1.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0
    assert recall_at_k(["a"], {"a", "b"}, 5) == 0.5
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert recall_at_k(["a", "b"], set(), 5) == 0.0
