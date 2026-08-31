"""Unit tests for per-visitor upload isolation and retrieval visibility."""
from retrieval.bm25_store import BM25Store
from retrieval.visibility import owner_visible
from ingestion.uploads import save_upload


def test_owner_visible_shared_and_private():
    assert owner_visible({"owner": "shared"}, "alice")
    assert owner_visible({}, "alice")
    assert owner_visible({"owner": "alice"}, "alice")
    assert not owner_visible({"owner": "bob"}, "alice")
    assert owner_visible({"owner": "bob"}, None)


def test_save_upload_namespaces_by_visitor(tmp_path, monkeypatch):
    from ingestion import uploads as uploads_mod

    monkeypatch.setattr(uploads_mod, "UPLOAD_DIR", tmp_path)
    a_path, a_rec = save_upload("exercise.md", b"# A\nAlice notes", visitor_id="alice")
    b_path, b_rec = save_upload("exercise.md", b"# B\nBob notes", visitor_id="bob")
    assert a_path == tmp_path / "alice" / "exercise.md"
    assert b_path == tmp_path / "bob" / "exercise.md"
    assert a_rec.source_id == "alice/exercise.md"
    assert b_rec.source_id == "bob/exercise.md"
    assert a_rec.doc_id != b_rec.doc_id
    assert a_rec.extra_metadata["owner"] == "alice"
    assert b_rec.extra_metadata["owner"] == "bob"


def test_bm25_hides_other_visitors_uploads():
    store = BM25Store.__new__(BM25Store)
    store.rebuild([
        ("c1", "shared refund policy for subscriptions", {
            "chunk_id": "c1", "text": "shared refund policy for subscriptions",
            "source_type": "markdown_docs", "owner": "shared",
        }),
        ("c2", "alice private exercise plan", {
            "chunk_id": "c2", "text": "alice private exercise plan",
            "source_type": "markdown_docs", "owner": "alice",
        }),
        ("c3", "bob private exercise plan", {
            "chunk_id": "c3", "text": "bob private exercise plan",
            "source_type": "markdown_docs", "owner": "bob",
        }),
    ])
    alice = store.search("exercise plan", top_k=10, owner_id="alice")
    assert {h["chunk_id"] for h in alice} == {"c2"}
    bob = store.search("exercise plan", top_k=10, owner_id="bob")
    assert {h["chunk_id"] for h in bob} == {"c3"}
    shared = store.search("refund policy", top_k=10, owner_id="alice")
    assert shared and shared[0]["chunk_id"] == "c1"
