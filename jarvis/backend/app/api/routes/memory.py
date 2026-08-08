"""Memory HTTP surface: search, create, delete, and the derived user profile.

Thin validation-at-the-boundary wrappers around `SQLiteMemoryStore`
(CONTRACTS.md section 4) and `UserProfile` (owned by `memory-profile`,
`app/core/memory/profile.py`). Every handler turns a store/profile failure
into a typed HTTP error rather than a 500 traceback leaking internals.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import get_memory_store
from app.models.schemas import MemoryHit, MemoryKind

log = logging.getLogger("jarvis.api.memory")

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=4_000)
    kind: MemoryKind = MemoryKind.FACT
    meta: dict[str, Any] = Field(default_factory=dict)


class MemoryCreateResponse(BaseModel):
    id: str


class DeleteResponse(BaseModel):
    deleted: bool


class ProfileResponse(BaseModel):
    name: str | None = None
    location: str | None = None
    timezone: str | None = None
    occupation: str | None = None
    interests: list[str] = Field(default_factory=list)
    people: dict[str, str] = Field(default_factory=dict)
    devices: list[str] = Field(default_factory=list)
    preferences: dict[str, str] = Field(default_factory=dict)
    communication_style: list[str] = Field(default_factory=list)
    summary: str = ""


@router.get("/search", response_model=list[MemoryHit])
async def search_memory(
    q: str = Query(..., min_length=1),
    k: int = Query(6, ge=1, le=50),
    memory: Any = Depends(get_memory_store),
) -> list[MemoryHit]:
    try:
        return await memory.recall(q, k=k)
    except Exception as exc:
        log.exception("memory search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("", response_model=MemoryCreateResponse)
@router.post("/", response_model=MemoryCreateResponse, include_in_schema=False)
async def create_memory(
    payload: MemoryCreate, memory: Any = Depends(get_memory_store)
) -> MemoryCreateResponse:
    try:
        mem_id = await memory.remember(payload.text, payload.kind.value, payload.meta)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("memory create failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return MemoryCreateResponse(id=mem_id)


@router.delete("/{memory_id}", response_model=DeleteResponse)
async def delete_memory(memory_id: str, memory: Any = Depends(get_memory_store)) -> DeleteResponse:
    forget = getattr(memory, "forget", None)
    if forget is None:
        raise HTTPException(status_code=501, detail="memory store does not support deletion")
    try:
        deleted = await forget(memory_id)
    except Exception as exc:
        log.exception("memory delete failed for %s", memory_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"memory {memory_id!r} not found")
    return DeleteResponse(deleted=True)


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(memory: Any = Depends(get_memory_store)) -> ProfileResponse:
    from app.core.memory.profile import UserProfile

    profile = UserProfile(memory)
    try:
        await profile.load()
    except Exception as exc:
        log.exception("profile load failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ProfileResponse(
        name=profile.name,
        location=profile.location,
        timezone=profile.timezone,
        occupation=profile.occupation,
        interests=profile.interests,
        people=profile.people,
        devices=profile.devices,
        preferences=profile.preferences,
        communication_style=profile.communication_style,
        summary=profile.render_for_prompt(),
    )
