"""Admin endpoints for ingest control.

Protected by API key. Useful for forcing a re-index from ops tooling
without restarting the poller.
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, Request

from api.schemas import IngestResponse
from api.middleware.auth import require_api_key

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_api_key)])


@router.post("/run", response_model=IngestResponse)
async def run_ingest(request: Request) -> IngestResponse:
    pipeline = request.app.state.ingestion_pipeline
    stats = await pipeline.run()
    return IngestResponse(**stats.__dict__)
