"""Per-visitor session id (cookie), not full user accounts."""

from __future__ import annotations

import uuid

from fastapi import Request

COOKIE_NAME = "ask_visitor"
ANON_ID = "anon"


def new_visitor_id() -> str:
    return str(uuid.uuid4())


def get_visitor_id(request: Request) -> str:
    vid = getattr(request.state, "visitor_id", None)
    if vid:
        return str(vid)
    cookie = request.cookies.get(COOKIE_NAME)
    if cookie:
        return cookie
    header = request.headers.get("X-Visitor-Id")
    if header:
        return header.strip()
    return ANON_ID
