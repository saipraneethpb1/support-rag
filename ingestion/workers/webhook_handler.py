"""Webhook handler — real-time, single-doc ingest on push events.

Mounted as a sub-app on FastAPI at /webhooks. Each connector type has its
own endpoint that translates the source payload into a SourceRecord and
calls pipeline.ingest_single().

In a real product you'd verify HMAC signatures per provider; we stub that
out with a shared-secret header for now.
"""
from __future__ import annotations
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request

from config.settings import get_settings
from ingestion.connectors.base import SourceRecord
from ingestion.pipeline import IngestionPipeline

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Pipeline is constructed in api/main.py and attached to app.state.
def _get_pipeline(request: Request) -> IngestionPipeline:
    pipeline = getattr(request.app.state, "ingestion_pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "Ingestion pipeline not initialized")
    return pipeline


def _check_secret(x_webhook_secret: str | None) -> None:
    expected = get_settings().api_key  # reuse for simplicity in dev
    if not x_webhook_secret or x_webhook_secret != expected:
        raise HTTPException(401, "Invalid webhook secret")


@router.post("/tickets/resolved")
async def ticket_resolved(
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
):
    _check_secret(x_webhook_secret)
    payload = await request.json()

    if payload.get("status") != "resolved":
        return {"ignored": True, "reason": "not resolved"}

    tid = str(payload["id"])
    subject = payload.get("subject", "").strip() or f"Ticket {tid}"
    lines = [f"# {subject}", ""]
    for m in payload.get("messages", []):
        body = (m.get("body") or "").strip()
        if body:
            lines.append(f"[{m.get('author', 'unknown')}] {body}")
            lines.append("")

    rec = SourceRecord(
        source_type="tickets",
        source_id=tid,
        title=subject,
        content="\n".join(lines),
        url=f"https://support.example.com/tickets/{tid}",
        updated_at=datetime.now(timezone.utc),
        extra_metadata={"status": "resolved", "tags": payload.get("tags", [])},
    )

    pipeline = _get_pipeline(request)
    changed = await pipeline.ingest_single(rec)
    return {"ingested": True, "changed": changed, "doc_id": rec.doc_id}


@router.post("/docs/updated")
async def docs_updated(
    request: Request,
    x_webhook_secret: str | None = Header(default=None),
):
    """Triggered by a docs CI build (e.g. GitHub action on docs repo push)."""
    _check_secret(x_webhook_secret)
    pipeline = _get_pipeline(request)
    # For docs repo we re-run the full markdown connector — the registry will
    # ensure only changed files actually do work.
    stats = await pipeline.run()
    return {"ok": True, "stats": stats.__dict__}
