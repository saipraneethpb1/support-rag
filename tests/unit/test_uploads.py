import pytest

from ingestion.uploads import UploadError, save_upload, safe_filename


def test_safe_filename_strips_paths_and_junk():
    assert safe_filename("../../etc/passwd.md") == "passwd.md"
    assert safe_filename("My Doc (1).md") == "My_Doc__1_.md"


def test_save_upload_writes_utf8(tmp_path, monkeypatch):
    from ingestion import uploads as uploads_mod

    monkeypatch.setattr(uploads_mod, "UPLOAD_DIR", tmp_path)
    path, rec = save_upload("guide.md", b"# Onboarding\nHello")
    assert path == tmp_path / "guide.md"
    assert rec.source_type == "markdown_docs"
    assert rec.title == "Onboarding"
    assert rec.source_id == "guide.md"


def test_save_upload_rejects_binary_and_empty(tmp_path, monkeypatch):
    from ingestion import uploads as uploads_mod

    monkeypatch.setattr(uploads_mod, "UPLOAD_DIR", tmp_path)
    with pytest.raises(UploadError):
        save_upload("x.md", b"")
    with pytest.raises(UploadError):
        save_upload("x.md", b"\xff\xfe")
    with pytest.raises(UploadError):
        save_upload("photo.png", b"abc")
    assert list(tmp_path.iterdir()) == []
