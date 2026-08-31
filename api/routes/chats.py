"""Per-visitor chat history stored in SQLite."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.middleware.auth import require_api_key
from api.middleware.visitor import visitor_id_from
from core.chat_store import ChatStore

router = APIRouter(prefix="/chats", tags=["chats"], dependencies=[Depends(require_api_key)])


def _store(request: Request) -> ChatStore:
    return request.app.state.chat_store


class ChatCreate(BaseModel):
    title: str = "New chat"


class ChatMessageIn(BaseModel):
    role: str
    content: str = ""
    citations: list[dict] = Field(default_factory=list)
    error: str | None = None
    created_at: str | None = None


class ChatUpdate(BaseModel):
    title: str | None = None
    messages: list[ChatMessageIn] | None = None


@router.get("/")
async def list_chats(request: Request) -> dict:
    visitor = visitor_id_from(request)
    chats = _store(request).list_chats(visitor)
    return {
        "chats": [
            {
                "id": c.id,
                "title": c.title,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
            }
            for c in chats
        ]
    }


@router.post("/")
async def create_chat(req: ChatCreate, request: Request) -> dict:
    visitor = visitor_id_from(request)
    chat = _store(request).create_chat(visitor, title=req.title or "New chat")
    return {
        "id": chat.id,
        "title": chat.title,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
        "messages": [],
    }


@router.get("/{chat_id}")
async def get_chat(chat_id: str, request: Request) -> dict:
    visitor = visitor_id_from(request)
    found = _store(request).get_chat(chat_id, visitor)
    if not found:
        raise HTTPException(status_code=404, detail="Chat not found")
    summary, messages = found
    return {
        "id": summary.id,
        "title": summary.title,
        "created_at": summary.created_at,
        "updated_at": summary.updated_at,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "citations": m.citations,
                "error": m.error,
                "created_at": m.created_at,
            }
            for m in messages
        ],
    }


@router.put("/{chat_id}")
async def update_chat(chat_id: str, req: ChatUpdate, request: Request) -> dict:
    visitor = visitor_id_from(request)
    payload = None if req.messages is None else [m.model_dump() for m in req.messages]
    ok = _store(request).replace_messages(chat_id, visitor, payload, title=req.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"ok": True}


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str, request: Request) -> dict:
    visitor = visitor_id_from(request)
    if not _store(request).delete_chat(chat_id, visitor):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"ok": True}
