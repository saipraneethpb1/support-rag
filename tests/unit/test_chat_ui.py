from pathlib import Path


def test_chat_ui_has_assistant_shell():
    html = Path("api/chat_ui.html").read_text(encoding="utf-8")
    assert "What can I help with?" in html
    assert 'id="new-chat"' in html
    assert 'id="sidebar"' in html
    assert "/chat/stream" in html
    assert "/ingest/upload" in html
    assert "JSON.stringify({question, history})" in html
