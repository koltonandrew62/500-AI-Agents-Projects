"""System HTTP surface: telemetry snapshot, tool listing, and tool invocation."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_telemetry_monitor, get_tool_registry
from app.models.schemas import Telemetry, ToolResult, ToolSpec

log = logging.getLogger("jarvis.api.system")

router = APIRouter(tags=["system"])


class ToolInvoke(BaseModel):
    args: dict[str, Any] = Field(default_factory=dict)


@router.get("/telemetry", response_model=Telemetry)
async def get_telemetry(monitor: Any = Depends(get_telemetry_monitor)) -> Telemetry:
    if monitor is None:
        raise HTTPException(status_code=503, detail="telemetry subsystem unavailable")
    try:
        return monitor.sample()
    except Exception as exc:
        log.exception("telemetry sample failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/tools", response_model=list[ToolSpec])
async def list_tools(registry: Any = Depends(get_tool_registry)) -> list[ToolSpec]:
    return registry.all()


@router.post("/tools/{name}", response_model=ToolResult)
async def invoke_tool(name: str, payload: ToolInvoke = ToolInvoke()) -> ToolResult:
    from app.core.tools import run_tool

    return await run_tool(name, payload.args)
