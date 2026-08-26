"""API-key auth.

Dev-grade: single shared key from settings, checked on protected routes.
The chat UI does not ask visitors for a key — GET / sets an HttpOnly
cookie so the browser can call chat and upload. curl still uses
x-api-key. For production you'd swap this for JWT or OAuth2.
"""
from __future__ import annotations
import secrets

from fastapi import Cookie, Header, HTTPException
from starlette.responses import Response

from config.settings import get_settings

ACCESS_COOKIE = "ask_access"


def attach_access_cookie(response: Response) -> None:
    """Let the chat page call authenticated routes without a key field."""
    settings = get_settings()
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=settings.api_key,
        httponly=True,
        samesite="lax",
        secure=settings.env == "production",
        path="/",
        max_age=60 * 60 * 24 * 30,
    )


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    ask_access: str | None = Cookie(default=None),
) -> None:
    expected = get_settings().api_key
    provided = x_api_key or ask_access
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
