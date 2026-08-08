"""POST /api/chat (non-streaming) and GET /api/history.

Non-streaming wrapper around the same `AgentLoop.handle` async generator the
WebSocket layer streams live -- this endpoint just drains it and returns the
final `done` text plus the full event trail, for callers that don't want a
socket (curl, simple scripts, server-to-server calls).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_agent_loop, get_memory_store
from app.models.schemas import Event, Message, TurnContext

log = logging.getLogger("jarvis.api.chat")

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8_000)
    id: str | None = None


class ChatResponse(BaseModel):
    id: str
    text: str
    events: list[Event] = Field(default_factory=list)


@router.post("/chat", response_model=ChatResponse)
async def post_chat(payload: ChatRequest, agent_loop: Any = Depends(get_agent_loop)) -> ChatResponse:
    turn_id = payload.id or uuid.uuid4().hex
    ctx = TurnContext(turn_id=turn_id)
    events: list[Event] = []
    reply = ""
    try:
        async for event in agent_loop.handle(payload.text, ctx):
            events.append(event)
            if event.type == "done":
                reply = event.text or ""
    except Exception as exc:
        log.exception("chat turn %s failed", turn_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(id=turn_id, text=reply, events=events)


@router.get("/history", response_model=list[Message])
async def get_history(limit: int = 20, memory: Any = Depends(get_memory_store)) -> list[Message]:
    try:
        return await memory.history(limit=limit)
    except Exception as exc:
        log.exception("history fetch failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
