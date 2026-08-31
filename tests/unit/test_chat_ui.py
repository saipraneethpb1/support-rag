from pathlib import Path


def test_chat_ui_has_assistant_shell():
    html = Path("api/chat_ui.html").read_text(encoding="utf-8")
    assert "What can I help with?" in html
    assert 'id="new-chat"' in html
    assert 'id="sidebar"' in html
    assert "/chat/stream" in html
    assert "/ingest/upload" in html
    assert "JSON.stringify({question, history, source_types})" in html
    assert "Paste API_KEY" not in html
    assert 'id="key"' not in html
    assert "credentials: 'same-origin'" in html
    assert 'id="chat-list"' in html
    assert "ask-chats-v1" in html
    assert "startNewChat" in html
    assert "/chats/" in html
    assert "AbortController" in html
    assert 'id="file-list"' in html
    assert 'id="health-dot"' in html
    assert ".pdf" in html
