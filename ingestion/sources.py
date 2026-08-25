"""Shared connector wiring for the files under ./data."""
from __future__ import annotations
from pathlib import Path

from ingestion.connectors.changelog import ChangelogConnector
from ingestion.connectors.help_center_html import HelpCenterHTMLConnector
from ingestion.connectors.markdown_docs import MarkdownDocsConnector
from ingestion.connectors.openapi import OpenAPIConnector
from ingestion.connectors.tickets import TicketsConnector

DATA = Path("data")
UPLOAD_DIR = DATA / "uploads"


def default_connectors(data: Path = DATA) -> list:
    connectors = []
    if (data / "sample_docs").exists():
        connectors.append(MarkdownDocsConnector(data / "sample_docs"))
    if (data / "sample_help_center").exists():
        connectors.append(HelpCenterHTMLConnector(data / "sample_help_center"))
    uploads = data / "uploads"
    if uploads.exists():
        connectors.append(
            MarkdownDocsConnector(
                uploads,
                base_url="uploaded://docs",
                patterns=("*.md", "*.txt"),
            )
        )
        connectors.append(
            HelpCenterHTMLConnector(uploads, base_url="uploaded://html")
        )
    if (data / "sample_tickets" / "tickets.jsonl").exists():
        connectors.append(TicketsConnector(data / "sample_tickets" / "tickets.jsonl"))
    if (data / "CHANGELOG.md").exists():
        connectors.append(ChangelogConnector(data / "CHANGELOG.md"))
    if (data / "openapi.json").exists():
        connectors.append(OpenAPIConnector(data / "openapi.json"))
    return connectors
