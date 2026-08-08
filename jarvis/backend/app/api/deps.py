"""Shared singleton providers for the API layer.

Everything the app builds once at startup (memory store, LLM provider/router,
agent loop, vision pipeline, telemetry monitor, connection manager) lives on
``app.state`` and is handed out through small ``Depends``-friendly functions
here, so route modules never reach into `app.main` directly.

`RoutedLLMProvider` is a thin adapter: `ModelRouter` (owned by `core-llm`)
exposes purpose-specific methods (`stream_chat`, `complete_chat`, `vision`)
rather than the flat `LLMProvider` protocol from CONTRACTS.md section 4 that
`AgentLoop` / `Planner` / `TurnClassifier` are written against. This adapter
bridges the two without either side needing to change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from fastapi import Request

from app.models.schemas import LLMDelta, Message, ToolSpec

if TYPE_CHECKING:
    from app.core.agent.loop import AgentLoop
    from app.core.llm.router import ModelRouter
    from app.core.memory.store import SQLiteMemoryStore
    from app.core.vision.pipeline import VisionPipeline

    from .ws import ConnectionManager

log = logging.getLogger("jarvis.api.deps")

__all__ = [
    "RoutedLLMProvider",
    "get_settings",
    "get_memory_store",
    "get_llm_router",
    "get_llm_provider",
    "get_agent_loop",
    "get_vision_pipeline",
    "get_telemetry_monitor",
    "get_connection_manager",
    "get_tool_registry",
]


class RoutedLLMProvider:
    """Adapts :class:`ModelRouter` to the ``LLMProvider`` protocol.

    Chat turns route through the `chat` chain. Streaming tool-call turns
    still go through `chat` -- planning has its own dedicated call path
    (`Planner` talks to the router directly via `complete_with_model`), so
    this adapter only needs to cover what `AgentLoop`'s tool-calling loop
    and `UserProfile` fact extraction actually use: `stream`, `complete`,
    `vision`.
    """

    def __init__(self, router: "ModelRouter") -> None:
        self._router = router

    async def stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> AsyncIterator[LLMDelta]:
        routed = self._router.stream_chat(messages, tools)
        async for delta in routed:
            yield delta

    async def complete(self, messages: list[Message]) -> str:
        result = await self._router.complete_chat(messages)
        return result.text

    async def vision(self, messages: list[Message], image_b64: str) -> str:
        result = await self._router.vision(messages, image_b64)
        return result.text


# ---------------------------------------------------------------------------
# Dependency providers -- thin reads off app.state, populated in main.py's
# lifespan. Kept as plain functions (not classes) so they compose naturally
# with FastAPI's `Depends`.
# ---------------------------------------------------------------------------


def get_settings(request: Request) -> Any:
    return request.app.state.settings


def get_memory_store(request: Request) -> "SQLiteMemoryStore":
    return request.app.state.memory_store


def get_llm_router(request: Request) -> "ModelRouter":
    return request.app.state.llm_router


def get_llm_provider(request: Request) -> RoutedLLMProvider:
    return request.app.state.llm_provider


def get_agent_loop(request: Request) -> "AgentLoop":
    return request.app.state.agent_loop


def get_vision_pipeline(request: Request) -> "VisionPipeline | None":
    return getattr(request.app.state, "vision_pipeline", None)


def get_telemetry_monitor(request: Request) -> Any:
    return getattr(request.app.state, "telemetry_monitor", None)


def get_connection_manager(request: Request) -> "ConnectionManager":
    return request.app.state.connection_manager


def get_tool_registry() -> Any:
    from app.core.tools import registry

    return registry
