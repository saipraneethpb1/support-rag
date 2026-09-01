from core.chat_store import ChatStore


def test_chat_store_keeps_threads_independent(tmp_path):
    store = ChatStore(tmp_path / "chats.db")
    a = store.create_chat("v1", title="New chat")
    store.replace_messages(
        a.id,
        "v1",
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ],
        title="first question",
    )
    b = store.create_chat("v1", title="New chat")
    found_a, msgs_a = store.get_chat(a.id, "v1")
    found_b, msgs_b = store.get_chat(b.id, "v1")
    assert found_a is not None and found_b is not None
    assert [m.content for m in msgs_a] == ["first question", "first answer"]
    assert msgs_b == []
    store.replace_messages(
        b.id,
        "v1",
        [{"role": "user", "content": "second question"}],
        title="second question",
    )
    _, msgs_a2 = store.get_chat(a.id, "v1")
    _, msgs_b2 = store.get_chat(b.id, "v1")
    assert [m.content for m in msgs_a2] == ["first question", "first answer"]
    assert [m.content for m in msgs_b2] == ["second question"]
    listed = store.list_chats("v1")
    assert [c.id for c in listed] == [b.id, a.id]
