"""FastAPI application factory + lifespan for J.A.R.V.I.S.

Wires every subsystem built by the ten-agent swarm into one process: LLM
routing, memory, tools, vision, telemetry, and the reminder scheduler. Owned
entirely by agent `be-api` (docs/CONTRACTS.md section 2) -- this file and
everything under `app/api/` is the only thing this agent touches.

Boot sequence (see `lifespan` below): init the memory DB, build the agent
loop, start the telemetry broadcast task and the reminder scheduler, then
tear everything down cleanly on shutdown. Every subsystem is soft-imported
where a sibling module might legitimately not be built yet -- this module
must still boot (and answer `/health`) with any of them missing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import ws as ws_module
from app.api.deps import RoutedLLMProvider
from app.api.routes import chat, memory, system, vision
from app.config import settings as app_settings
from app.core.agent.loop import AgentLoop
from app.core.llm.router import ModelRouter
from app.core.memory import SQLiteMemoryStore, init_db
from app.models.schemas import Event

log = logging.getLogger("jarvis.main")

BACKEND_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"

_REMINDER_POLL_S = 5.0


def _configure_logging() -> None:
    level = getattr(logging, str(app_settings.log_level).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _build_vision_pipeline() -> Any:
    """Soft-import: `core.vision` is a sibling module (`be-sense`)."""
    try:
        from app.core.vision import VisionPipeline

        return VisionPipeline()
    except Exception:
        log.warning("vision subsystem unavailable at startup", exc_info=True)
        return None


def _build_telemetry_monitor() -> Any:
    """Soft-import: `core.telemetry` is a sibling module (`be-sense`)."""
    try:
        from app.core.telemetry import TelemetryMonitor

        return TelemetryMonitor()
    except Exception:
        log.warning("telemetry subsystem unavailable at startup", exc_info=True)
        return None


def _load_voice_module() -> Any:
    """Soft-import: `core.voice` is a sibling module (`be-sense`)."""
    try:
        import app.core.voice as voice_module

        return voice_module
    except Exception:
        log.warning("voice subsystem unavailable at startup", exc_info=True)
        return None


async def _telemetry_broadcast_loop(app: FastAPI) -> None:
    """Push `telemetry` to every connected client on a fixed interval.

    Degrades to a no-op tick (never a crash) when the telemetry monitor
    isn't available -- e.g. `core.telemetry` hasn't landed yet.
    """
    manager = app.state.connection_manager
    interval_s = max(0.05, app_settings.telemetry_interval_ms / 1000.0)
    while True:
        monitor = getattr(app.state, "telemetry_monitor", None)
        if monitor is not None:
            try:
                telemetry = monitor.sample()
                await manager.broadcast(Event(type="telemetry", data=telemetry.model_dump()))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("telemetry broadcast tick failed", exc_info=True)
        await asyncio.sleep(interval_s)


async def _reminder_scheduler_loop(app: FastAPI) -> None:
    """Fire due reminders (created via the `create_reminder` tool) to every client.

    `core.tools.scheduler.run_scheduler` owns persistence and due-polling;
    this just supplies the `on_fire` callback that turns a fired reminder
    into a `speak` + `log` broadcast.
    """
    try:
        from app.core.tools.scheduler import run_scheduler
    except Exception:
        log.warning("reminder scheduler unavailable at startup", exc_info=True)
        return

    manager = app.state.connection_manager

    async def _on_fire(reminder_id: str, text: str) -> None:
        log.info("reminder %s fired: %s", reminder_id, text)
        await manager.broadcast(Event(type="speak", text=f"Reminder: {text}", voice="jarvis"))
        await manager.broadcast(Event(type="log", level="info", text=f"Reminder fired: {text}"))

    try:
        await run_scheduler(_on_fire, interval_s=_REMINDER_POLL_S)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("reminder scheduler loop crashed")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _configure_logging()
    app_settings.ensure_dirs()

    log.info("initializing memory DB at %s", app_settings.memory_db_path)
    await init_db(str(app_settings.memory_db_path))
    memory_store = SQLiteMemoryStore(app_settings.memory_db_path)

    llm_router = ModelRouter(
        api_key=app_settings.openrouter_api_key or None,
        chains={
            "planning": tuple(app_settings.planning_models),
            "chat": tuple(app_settings.chat_models),
            "vision": tuple(app_settings.vision_models),
        },
    )
    llm_provider = RoutedLLMProvider(llm_router)

    agent_loop = AgentLoop(
        llm_provider,
        memory=memory_store,
        max_iterations=app_settings.max_tool_iterations,
        emit_speak=app_settings.voice_enabled,
    )

    app.state.settings = app_settings
    app.state.memory_store = memory_store
    app.state.llm_router = llm_router
    app.state.llm_provider = llm_provider
    app.state.agent_loop = agent_loop
    app.state.vision_pipeline = _build_vision_pipeline()
    app.state.telemetry_monitor = _build_telemetry_monitor()
    app.state.voice_module = _load_voice_module()
    app.state.connection_manager = ws_module.ConnectionManager()
    app.state.started_at = time.time()

    background: list[asyncio.Task[Any]] = [
        asyncio.create_task(_telemetry_broadcast_loop(app), name="telemetry-broadcast"),
        asyncio.create_task(_reminder_scheduler_loop(app), name="reminder-scheduler"),
    ]

    log.info("jarvis backend ready (vision=%s, telemetry=%s, voice=%s)",
             app.state.vision_pipeline is not None,
             app.state.telemetry_monitor is not None,
             app.state.voice_module is not None)
    try:
        yield
    finally:
        log.info("jarvis backend shutting down")
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)


def create_app() -> FastAPI:
    app = FastAPI(title="J.A.R.V.I.S.", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_settings.cors_origins) or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ws_module.router)
    app.include_router(chat.router, prefix="/api")
    app.include_router(memory.router, prefix="/api")
    app.include_router(system.router, prefix="/api")
    app.include_router(vision.router, prefix="/api")

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        state = request.app.state
        return {
            "llm": _llm_health(),
            "db": await _db_health(state),
            "vision": _subsystem_health(getattr(state, "vision_pipeline", None), app_settings.vision_enabled),
            "voice": _subsystem_health(getattr(state, "voice_module", None), app_settings.voice_enabled),
        }

    if FRONTEND_DIST.is_dir():
        log.info("serving frontend from %s", FRONTEND_DIST)
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

    return app


def _llm_health() -> dict[str, Any]:
    configured = bool(app_settings.openrouter_api_key)
    return {
        "status": "ok" if configured else "unconfigured",
        "detail": "OPENROUTER_API_KEY not set" if not configured else "",
        "chat_models": app_settings.chat_models,
    }


async def _db_health(state: Any) -> dict[str, Any]:
    store = getattr(state, "memory_store", None)
    if store is None:
        return {"status": "error", "detail": "memory store not initialized"}
    try:
        await store.history(limit=1)
    except Exception as exc:  # noqa: BLE001 - health check must never raise
        return {"status": "error", "detail": str(exc)}
    return {"status": "ok", "detail": ""}


def _subsystem_health(instance: Any, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"status": "disabled", "detail": ""}
    if instance is None:
        return {"status": "unavailable", "detail": "module not loaded"}
    return {"status": "ok", "detail": ""}


app = create_app()
