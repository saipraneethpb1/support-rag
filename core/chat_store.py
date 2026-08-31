"""SQLite persistence for per-visitor chat threads."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChatSummary:
    id: str
    visitor_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class ChatMessage:
    role: str
    content: str
    citations: list[dict[str, Any]]
    error: str | None
    created_at: str


class ChatStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    visitor_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chats_visitor ON chats(visitor_id, updated_at);
                """
            )

    def list_chats(self, visitor_id: str) -> list[ChatSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, visitor_id, title, created_at, updated_at
                FROM chats WHERE visitor_id = ?
                ORDER BY updated_at DESC
                """,
                (visitor_id,),
            ).fetchall()
        return [ChatSummary(**dict(r)) for r in rows]

    def create_chat(self, visitor_id: str, title: str = "New chat") -> ChatSummary:
        now = _utc_now()
        chat_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chats (id, visitor_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (chat_id, visitor_id, title, now, now),
            )
        return ChatSummary(id=chat_id, visitor_id=visitor_id, title=title, created_at=now, updated_at=now)

    def get_chat(self, chat_id: str, visitor_id: str) -> tuple[ChatSummary, list[ChatMessage]] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, visitor_id, title, created_at, updated_at FROM chats WHERE id = ? AND visitor_id = ?",
                (chat_id, visitor_id),
            ).fetchone()
            if not row:
                return None
            msgs = conn.execute(
                """
                SELECT role, content, citations, error, created_at
                FROM messages WHERE chat_id = ? ORDER BY id ASC
                """,
                (chat_id,),
            ).fetchall()
        messages = [
            ChatMessage(
                role=m["role"],
                content=m["content"],
                citations=json.loads(m["citations"] or "[]"),
                error=m["error"],
                created_at=m["created_at"],
            )
            for m in msgs
        ]
        return ChatSummary(**dict(row)), messages

    def replace_messages(
        self,
        chat_id: str,
        visitor_id: str,
        messages: list[dict[str, Any]] | None,
        title: str | None = None,
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM chats WHERE id = ? AND visitor_id = ?",
                (chat_id, visitor_id),
            ).fetchone()
            if not row:
                return False
            now = _utc_now()
            if title:
                conn.execute(
                    "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, chat_id),
                )
            else:
                conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
            if messages is None:
                return True
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            for msg in messages:
                conn.execute(
                    """
                    INSERT INTO messages (chat_id, role, content, citations, error, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        msg.get("role", "user"),
                        msg.get("content", ""),
                        json.dumps(msg.get("citations") or []),
                        msg.get("error"),
                        msg.get("created_at") or now,
                    ),
                )
        return True

    def delete_chat(self, chat_id: str, visitor_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM chats WHERE id = ? AND visitor_id = ?",
                (chat_id, visitor_id),
            )
            return cur.rowcount > 0
