"""Langfuse tracing.

Every /chat request gets a trace with spans for retrieval, prompt build,
LLM call, and citation audit. Scores (coverage, invented citations) are
attached so you can build dashboards later.

Gracefully no-ops if Langfuse keys aren't configured — we don't want a
missing observability dep to block a dev from running the app.
"""
from __future__ import annotations
from typing import Any

from config.settings import get_settings
from observability.logger import get_logger

log = get_logger(__name__)


class _NullTracer:
    """Stand-in when Langfuse isn't configured. Matches the interface subset we use."""
    def trace(self, **kwargs: Any) -> "_NullTrace":
        return _NullTrace()
    def flush(self) -> None:
        pass


class _NullTrace:
    def span(self, **kwargs: Any) -> "_NullSpan":
        return _NullSpan()
    def score(self, **kwargs: Any) -> None:
        pass
    def update(self, **kwargs: Any) -> None:
        pass


class _NullSpan:
    def end(self, **kwargs: Any) -> None:
        pass
    def update(self, **kwargs: Any) -> None:
        pass


def get_tracer():
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return _NullTracer()
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
    except Exception as e:
        log.warning("langfuse_init_failed", error=str(e))
        return _NullTracer()
