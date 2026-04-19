"""Resolved-tickets connector.

Reads JSONL where each line is a ticket:
    {
      "id": "T-12345",
      "subject": "...",
      "status": "resolved",
      "created_at": "...",
      "updated_at": "...",
      "messages": [{"author": "user|agent", "body": "...", "ts": "..."}]
    }

Only resolved tickets are ingested (unresolved ones can mislead the bot).
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from ingestion.connectors.base import BaseConnector, SourceRecord


class TicketsConnector(BaseConnector):
    source_type = "tickets"

    def __init__(self, jsonl_path: str | Path, base_url: str = "https://support.example.com/tickets"):
        self.path = Path(jsonl_path).resolve()
        self.base_url = base_url.rstrip("/")

    async def list_records(self) -> AsyncIterator[SourceRecord]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if t.get("status") != "resolved":
                    continue

                tid = str(t["id"])
                subject = t.get("subject", "").strip() or f"Ticket {tid}"

                # Render conversation as readable text. Keep author tags so the
                # chunker can keep question + resolution together.
                lines = [f"# {subject}", ""]
                for m in t.get("messages", []):
                    author = m.get("author", "unknown")
                    body = (m.get("body") or "").strip()
                    if body:
                        lines.append(f"[{author}] {body}")
                        lines.append("")
                content = "\n".join(lines)

                updated_at_raw = t.get("updated_at") or t.get("created_at")
                updated_at = (
                    datetime.fromisoformat(updated_at_raw.replace("Z", "+00:00"))
                    if updated_at_raw else self.now_utc()
                )

                yield SourceRecord(
                    source_type="tickets",
                    source_id=tid,
                    title=subject,
                    content=content,
                    url=f"{self.base_url}/{tid}",
                    updated_at=updated_at,
                    extra_metadata={
                        "status": t.get("status"),
                        "tags": t.get("tags", []),
                        "message_count": len(t.get("messages", [])),
                    },
                )
