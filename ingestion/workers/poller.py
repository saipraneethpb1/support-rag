"""Polling worker.

Runs the ingestion pipeline on a fixed interval. Cheap because the
content-hash check in the registry skips unchanged docs — only changed
docs hit the embedder.

For sources that support webhooks (GitHub, Zendesk, etc.), prefer the
webhook handler instead and use this only as a safety net.
"""
from __future__ import annotations
import asyncio
import signal

from ingestion.pipeline import IngestionPipeline
from observability.logger import get_logger

log = get_logger(__name__)


class Poller:
    def __init__(self, pipeline: IngestionPipeline, interval_seconds: int = 300):
        self.pipeline = pipeline
        self.interval = interval_seconds
        self._stop = asyncio.Event()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                pass  # Windows

    async def run_forever(self) -> None:
        self._install_signal_handlers()
        log.info("poller_started", interval_s=self.interval)
        while not self._stop.is_set():
            try:
                stats = await self.pipeline.run()
                log.info("poll_cycle_done", **stats.__dict__)
            except Exception as e:
                log.exception("poll_cycle_failed", error=str(e))

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass
        log.info("poller_stopped")
