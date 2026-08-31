"""Who can see a retrieved chunk.

Bootstrap/sample corpus is `owner=shared` (or missing owner on older indexes).
UI uploads set `owner` to the visitor id so two people can upload the same
filename without colliding or leaking into each other's answers.
"""
from __future__ import annotations

SHARED = frozenset({None, "", "shared"})


def owner_visible(payload: dict, owner_id: str | None) -> bool:
    if owner_id is None:
        return True
    owner = payload.get("owner")
    if owner in SHARED:
        return True
    return owner == owner_id
