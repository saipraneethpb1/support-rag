"""Markdown documentation connector.

Reads .md files from a directory tree (e.g. a Docusaurus / MkDocs site
checked out locally, or synced from a docs repo). In production you'd
add a Git-pull worker on top to keep it fresh.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from ingestion.connectors.base import BaseConnector, SourceRecord


class MarkdownDocsConnector(BaseConnector):
    source_type = "markdown_docs"

    def __init__(self, root_dir: str | Path, base_url: str = "https://docs.example.com"):
        self.root = Path(root_dir).resolve()
        self.base_url = base_url.rstrip("/")

    async def list_records(self) -> AsyncIterator[SourceRecord]:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*.md")):
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            rel = path.relative_to(self.root).as_posix()
            url_path = rel.removesuffix(".md").removesuffix("/index")
            url = f"{self.base_url}/{url_path}"

            # Title: first H1 if present, else filename
            title = path.stem.replace("-", " ").replace("_", " ").title()
            for line in content.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

            stat = path.stat()
            yield SourceRecord(
                source_type="markdown_docs",
                source_id=rel,
                title=title,
                content=content,
                url=url,
                updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                extra_metadata={"path": str(path), "size_bytes": stat.st_size},
            )
