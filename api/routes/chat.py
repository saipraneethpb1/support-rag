"""Chat endpoints.

Two endpoints:
  - POST /chat         : blocking JSON response. Uses the semantic cache.
  - POST /chat/stream  : Server-Sent Events. Tokens stream live; a final
                         'meta' event carries citations + trace info.

Every request creates a Langfuse trace with spans for retrieval, LLM call,
and citation audit. Scores attached: coverage, invented_citations_count.
"""
from __future__ import annotations
import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatResponse, CitationOut
from api.middleware.auth import require_api_key
from api.middleware.rate_limit import rate_limit
from generation.generator import Generator
from observability.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["chat"], dependencies=[Depends(require_api_key), Depends(rate_limit)])


def _get_generator(request: Request) -> Generator:
    return request.app.state.generator


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    gen = _get_generator(request)
    tracer = request.app.state.tracer
    trace = tracer.trace(name="chat", input={"question": req.question})

    try:
        result = await gen.generate(
            req.question,
            history=req.history,
            source_types=req.source_types,
            use_cache=req.use_cache,
        )
    except Exception as e:
        log.exception("chat_failed", error=str(e))
        trace.update(output={"error": str(e)}, level="ERROR")
        raise

    trace.update(output={"answer": result.answer, "cache_hit": result.cache_hit})
    trace.score(name="coverage", value=result.audit.sentence_coverage)
    trace.score(name="invented_citations", value=float(len(result.audit.invented_markers)))

    return ChatResponse(
        trace_id=result.trace_id,
        answer=result.answer,
        citations=[
            CitationOut(
                marker=c.marker, title=c.title, url=c.url,
                source_type=c.source_type, snippet=c.snippet,
            )
            for c in result.audit.used_citations
        ],
        cache_hit=result.cache_hit,
        invented_citations=sorted(result.audit.invented_markers),
        coverage=result.audit.sentence_coverage,
        timings_ms=result.timings_ms,
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    gen = _get_generator(request)
    tracer = request.app.state.tracer
    trace = tracer.trace(name="chat_stream", input={"question": req.question})

    async def event_generator():
        try:
            async for event in gen.stream(
                req.question,
                history=req.history,
                source_types=req.source_types,
            ):
                # SSE format: "data: <json>\n\n"
                payload = json.dumps({"type": event.type, **event.data})
                yield f"event: {event.type}\ndata: {payload}\n\n"
                # Flush periodically
                if event.type == "token":
                    await asyncio.sleep(0)
                if event.type == "meta":
                    trace.update(output={"trace_id": event.data.get("trace_id")})
                    coverage = event.data.get("coverage", 0.0)
                    invented = len(event.data.get("invented_citations", []))
                    trace.score(name="coverage", value=coverage)
                    trace.score(name="invented_citations", value=float(invented))
        except Exception as e:
            log.exception("stream_failed", error=str(e))
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if proxied
        },
    )
