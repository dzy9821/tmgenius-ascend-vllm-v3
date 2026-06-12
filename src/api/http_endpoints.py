from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1")


@router.get("/connections")
async def connections(request: Request) -> dict:
    sm = getattr(request.app.state, "session_manager", None)
    count = sm.current_count if sm is not None else 0
    return {"active": count}
