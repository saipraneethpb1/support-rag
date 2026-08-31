"""Assign a stable visitor id (cookie), not a login."""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from config.settings import get_settings
from core.session import COOKIE_NAME, get_visitor_id, new_visitor_id


def attach_visitor_cookie(response: Response, visitor_id: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=COOKIE_NAME,
        value=visitor_id,
        httponly=True,
        samesite="lax",
        secure=settings.env == "production",
        path="/",
        max_age=60 * 60 * 24 * 365,
    )


class VisitorMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        existing = request.cookies.get(COOKIE_NAME) or (request.headers.get("X-Visitor-Id") or "").strip()
        minted = False
        if existing:
            request.state.visitor_id = existing
        else:
            request.state.visitor_id = new_visitor_id()
            minted = True
        response = await call_next(request)
        if minted:
            attach_visitor_cookie(response, request.state.visitor_id)
        return response


def visitor_id_from(request: Request) -> str:
    return get_visitor_id(request)
