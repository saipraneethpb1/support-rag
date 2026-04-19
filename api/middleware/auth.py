"""API-key auth.

Dev-grade: single shared key from settings, checked on protected routes.
For production you'd swap this for JWT or OAuth2. The dependency-injection
shape here means the swap is a one-file change.
"""
from __future__ import annotations
from fastapi import Header, HTTPException

from config.settings import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
