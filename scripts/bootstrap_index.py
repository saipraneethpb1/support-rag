"""Bootstrap initial index.

Usage:
    python -m scripts.bootstrap_index

Wires up all connectors against ./data/* and runs one full ingestion
pass. Safe to re-run — the registry will short-circuit unchanged docs.
"""
from __future__ import annotations
import asyncio
from pathlib import Path

from ingestion.connectors.markdown_docs import MarkdownDocsConnector
from ingestion.connectors.help_center_html import HelpCenterHTMLConnector
from ingestion.connectors.tickets import TicketsConnector
from ingestion.connectors.changelog import ChangelogConnector
from ingestion.connectors.openapi import OpenAPIConnector
from ingestion.pipeline import IngestionPipeline
from observability.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

DATA = Path("data")


async def main() -> None:
    connectors = []

    if (DATA / "sample_docs").exists():
        connectors.append(MarkdownDocsConnector(DATA / "sample_docs"))
    if (DATA / "sample_help_center").exists():
        connectors.append(HelpCenterHTMLConnector(DATA / "sample_help_center"))
    if (DATA / "sample_tickets" / "tickets.jsonl").exists():
        connectors.append(TicketsConnector(DATA / "sample_tickets" / "tickets.jsonl"))
    if (DATA / "CHANGELOG.md").exists():
        connectors.append(ChangelogConnector(DATA / "CHANGELOG.md"))
    if (DATA / "openapi.json").exists():
        connectors.append(OpenAPIConnector(DATA / "openapi.json"))

    if not connectors:
        log.error("no_data_found", hint="Run `python -m scripts.seed_demo_data` first")
        return

    pipeline = IngestionPipeline(connectors)
    stats = await pipeline.run()
    log.info("bootstrap_done", **stats.__dict__)


if __name__ == "__main__":
    asyncio.run(main())
