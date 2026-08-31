"""Admin endpoints for ingest control.

Protected by API key. Useful for forcing a re-index from ops tooling
without restarting the app (or when the optional poller is disabled).
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from api.schemas import IngestResponse, UploadFileResult, UploadResponse
from api.middleware.auth import require_api_key
from api.middleware.rate_limit import rate_limit
from api.middleware.visitor import visitor_id_from
from ingestion.uploads import (
    UploadError,
    delete_upload_file,
    list_uploads,
    record_from_upload,
    save_upload,
    suffix_ok,
)
from observability.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"], dependencies=[Depends(require_api_key)])


@router.post("/run", response_model=IngestResponse)
async def run_ingest(request: Request) -> IngestResponse:
    pipeline = request.app.state.ingestion_pipeline
    stats = await pipeline.run()
    return IngestResponse(**stats.__dict__)


@router.get("/status")
async def ingest_status(request: Request) -> dict:
    pipeline = request.app.state.ingestion_pipeline
    count = await pipeline.document_count()
    return {"documents": count}


@router.get("/uploads")
async def get_uploads(request: Request) -> dict:
    return {"files": list_uploads(visitor_id_from(request))}


@router.delete("/uploads/{filename}")
async def delete_upload(filename: str, request: Request) -> dict:
    visitor = visitor_id_from(request)
    pipeline = request.app.state.ingestion_pipeline
    if not suffix_ok(filename):
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    path = delete_upload_file(filename, visitor)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found.")
    record = record_from_upload(path.name, "x", visitor)
    await pipeline.delete_doc(record.doc_id)
    return {"ok": True, "filename": path.name}


@router.post("/upload", response_model=UploadResponse, dependencies=[Depends(rate_limit)])
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Upload at most 20 files at a time.")

    pipeline = request.app.state.ingestion_pipeline
    visitor = visitor_id_from(request)
    results: list[UploadFileResult] = []
    ingested = 0

    for upload in files:
        name = upload.filename or "untitled.txt"
        try:
            data = await upload.read()
            path, record = save_upload(name, data, visitor_id=visitor)
            changed = await pipeline.ingest_single(record)
            ingested += 1
            results.append(
                UploadFileResult(
                    filename=path.name,
                    ok=True,
                    changed=changed,
                    doc_id=record.doc_id,
                )
            )
        except UploadError as e:
            results.append(UploadFileResult(filename=name, ok=False, error=str(e)))
        except Exception:
            log.exception("upload_index_failed", filename=name)
            results.append(
                UploadFileResult(filename=name, ok=False, error="Could not index this file.")
            )

    return UploadResponse(ingested=ingested, files=results)
