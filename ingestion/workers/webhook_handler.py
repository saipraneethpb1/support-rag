"""Webhook handler — real-time, single-doc ingest on push events.

Mounted as a sub-app on FastAPI at /webhooks. Each connector type has its
own endpoint that translates the source payload into a SourceRecord and
calls pipeline.ingest_single().

Auth uses WEBHOOK_SECRET (falls back to API_KEY). Prefer HMAC verification
per provider in a real product.
"""
from __future__ import annotations
import secrets
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from api.middleware.rate_limit import rate_limit
from config.settings import get_settings
from ingestion.connectors.base import SourceRecord
from ingestion.pipeline import IngestionPipeline

router = APIRouter(
    prefix="/webhooks",
    tags=["webhooks"],
    dependencies=[Depends(rate_limit)],
)


def _get_pipeline(request: Request) -> IngestionPipeline:
    pipeline = getattr(request.app.state, "ingestion_pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "Ingestion pipeline not initialized")
    return pipeline


def _check_secret(x_webhook_secret: str | None) -> None:
    expected = get_settings().resolved_webhook_secret()
    if not x_webhook_secret or not secrets.compare_digest(x_webhook_secret, expected):
        raise HTTPException(401, "Invalid webhook secret")


class TicketMessage(BaseModel):
    author: str = "unknown"
    body: str = ""


class TicketResolvedPayload(BaseModel):
    id: str | int
    status: str
    subject: str = ""
    messages: list[TicketMessage] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class DocsUpdatedPayload(BaseModel):
    reason: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


@router.post("/tickets/resolved")
async def ticket_resolved(
    payload: TicketResolvedPayload,
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
):
    _check_secret(x_webhook_secret)

    if payload.status != "resolved":
        return {"ignored": True, "reason": "not resolved"}

    tid = str(payload.id)
    subject = payload.subject.strip() or f"Ticket {tid}"
    lines = [f"# {subject}", ""]
    for m in payload.messages:
        body = (m.body or "").strip()
        if body:
            lines.append(f"[{m.author}] {body}")
            lines.append("")

    rec = SourceRecord(
        source_type="tickets",
        source_id=tid,
        title=subject,
        content="\n".join(lines),
        url=f"https://example.com/tickets/{tid}",
        updated_at=datetime.now(timezone.utc),
        extra_metadata={"status": "resolved", "tags": payload.tags},
    )

    pipeline = _get_pipeline(request)
    changed = await pipeline.ingest_single(rec)
    return {"ingested": True, "changed": changed, "doc_id": rec.doc_id}


@router.post("/docs/updated")
async def docs_updated(
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
    payload: DocsUpdatedPayload | None = None,
):
    """Triggered by a docs CI build (e.g. GitHub action on docs repo push)."""
    _check_secret(x_webhook_secret)
    pipeline = _get_pipeline(request)
    # For docs repo we re-run the full markdown connector — the registry will
    # ensure only changed files actually do work.
    stats = await pipeline.run()
    return {"ok": True, "stats": stats.__dict__, "reason": payload.reason if payload else None}
