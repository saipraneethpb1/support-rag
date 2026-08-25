"""Bootstrap initial index.

Usage:
    python -m scripts.bootstrap_index

Wires up all connectors against ./data/* and runs one full ingestion
pass. Safe to re-run — the registry will short-circuit unchanged docs.
"""
from __future__ import annotations
import asyncio

from ingestion.pipeline import IngestionPipeline
from ingestion.sources import default_connectors
from observability.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)


async def main() -> None:
    connectors = default_connectors()
    if not connectors:
        log.error("no_data_found", hint="Run `python -m scripts.seed_demo_data` first")
        return

    pipeline = IngestionPipeline(connectors)
    stats = await pipeline.run()
    log.info("bootstrap_done", **stats.__dict__)


if __name__ == "__main__":
    asyncio.run(main())
