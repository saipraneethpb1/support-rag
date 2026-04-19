"""In-process token-bucket rate limiter.

Per-key (API key or IP) bucket with refill rate + capacity. Good enough
for a single-instance deployment; for multi-instance you'd back it with
Redis INCR + TTL. We log when limits are hit so operators can see abuse
patterns before users complain.

Why token bucket over fixed-window: smooths bursty traffic without
allowing the well-known "2x-at-window-boundary" burst exploit.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from fastapi import HTTPException, Request

from observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    def __init__(self, *, capacity: int = 30, refill_per_second: float = 0.5):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._buckets: dict[str, _Bucket] = {}

    def check(self, key: str) -> None:
        now = time.monotonic()
        b = self._buckets.get(key)
        if b is None:
            b = _Bucket(tokens=self.capacity - 1, last_refill=now)
            self._buckets[key] = b
            return
        elapsed = now - b.last_refill
        b.tokens = min(self.capacity, b.tokens + elapsed * self.refill_per_second)
        b.last_refill = now
        if b.tokens < 1:
            log.warning("rate_limit_hit", key=key, tokens=b.tokens)
            retry_after = int((1 - b.tokens) / self.refill_per_second) + 1
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )
        b.tokens -= 1


_limiter = RateLimiter()


async def rate_limit(request: Request) -> None:
    # Prefer API key as the limit dimension; fall back to client IP
    key = request.headers.get("x-api-key") or (request.client.host if request.client else "anon")
    _limiter.check(key)
