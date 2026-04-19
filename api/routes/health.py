"""Health check. Covers the three external deps the app actually relies on."""
from __future__ import annotations
from fastapi import APIRouter, Request

from api.schemas import HealthResponse
from config.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = get_settings()
    status = "ok"

    qdrant_ok = False
    try:
        vs = request.app.state.vector_store
        await vs._client.get_collections()
        qdrant_ok = True
    except Exception:
        status = "degraded"

    redis_ok = False
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        redis_ok = bool(await client.ping())
    except Exception:
        # Not fatal — cache is optional. Don't flip status to degraded on redis alone.
        pass

    providers: list[str] = []
    if settings.groq_api_key:
        providers.append("groq")
    if settings.google_api_key:
        providers.append("gemini")
    if not providers:
        status = "degraded"

    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        qdrant=qdrant_ok,
        redis=redis_ok,
        providers=providers,
    )
