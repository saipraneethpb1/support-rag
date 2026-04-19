"""Help center HTML connector.

Reads exported HTML articles (e.g. from Zendesk/Intercom export, or a
crawl). Strips chrome (nav, footer, ads) so the cleaner stage gets
mostly-content HTML.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from bs4 import BeautifulSoup

from ingestion.connectors.base import BaseConnector, SourceRecord


class HelpCenterHTMLConnector(BaseConnector):
    source_type = "help_center"

    def __init__(self, root_dir: str | Path, base_url: str = "https://help.example.com"):
        self.root = Path(root_dir).resolve()
        self.base_url = base_url.rstrip("/")

    async def list_records(self) -> AsyncIterator[SourceRecord]:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*.html")):
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            soup = BeautifulSoup(raw, "lxml")

            # Drop obvious non-content
            for tag in soup.select("nav, header, footer, script, style, .sidebar, .ads"):
                tag.decompose()

            # Try common article containers; fall back to body
            article = soup.select_one("article, main, .article, .content") or soup.body
            content_html = str(article) if article else raw

            title_tag = soup.select_one("h1, title")
            title = title_tag.get_text(strip=True) if title_tag else path.stem

            rel = path.relative_to(self.root).as_posix()
            url = f"{self.base_url}/{rel.removesuffix('.html')}"

            stat = path.stat()
            yield SourceRecord(
                source_type="help_center",
                source_id=rel,
                title=title,
                content=content_html,
                url=url,
                updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                extra_metadata={"path": str(path)},
            )
