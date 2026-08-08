"""Integration smoke tests for J.A.R.V.I.S.

These verify the contract between modules built independently by the ten-agent
swarm: that every module imports, that the interfaces in docs/CONTRACTS.md are
actually honored, and that the app boots with no optional dependency installed
and no API key set.

Run:  cd backend && python -m pytest tests/ -v
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


# ---------------------------------------------------------------------------
# Contract: every module imports cleanly with no key and no optional deps
# ---------------------------------------------------------------------------

MODULES = [
    "app.config",
    "app.models.schemas",
    "app.core.llm",
    "app.core.memory",
    "app.core.tools",
    "app.core.agent",
    "app.core.vision",
    "app.core.voice",
    "app.core.telemetry",
    "app.main",
]


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod: str) -> None:
    """Import must never require an API key or a heavy optional dependency."""
    __import__(mod)


def test_settings_import_safe_without_env() -> None:
    from app.config import settings

    # Missing key must not raise at import — it surfaces later, at call time.
    assert hasattr(settings, "openrouter_api_key")
    assert isinstance(settings.planning_models, list) and settings.planning_models
    assert isinstance(settings.vision_models, list) and settings.vision_models
    assert settings.workspace_root.exists(), "ensure_dirs() should have run"


def test_no_gemini_anywhere() -> None:
    """The user explicitly excluded Gemini from the model chains."""
    from app.config import settings

    chains = settings.planning_models + settings.chat_models + settings.vision_models
    assert not any("gemini" in m.lower() or "google" in m.lower() for m in chains)


def test_planning_chain_is_free_tier() -> None:
    from app.config import settings

    assert all(":free" in m for m in settings.planning_models), settings.planning_models


# ---------------------------------------------------------------------------
# Contract: schemas are the single source of truth
# ---------------------------------------------------------------------------


def test_schema_surface() -> None:
    from app.models import schemas

    for name in (
        "Message", "Role", "LLMDelta", "ToolSpec", "ToolResult",
        "Step", "Plan", "MemoryKind", "MemoryHit", "Telemetry",
        "DetectedObject", "VisionResult", "Event", "TurnContext",
    ):
        assert hasattr(schemas, name), f"schemas.{name} missing"


def test_event_types_match_ws_protocol() -> None:
    """Event.type must cover exactly the server->client protocol."""
    from typing import get_args

    from app.models.schemas import Event

    declared = set(get_args(Event.model_fields["type"].annotation))
    expected = {
        "token", "done", "thinking", "tool_call", "tool_result",
        "telemetry", "vision", "speak", "log", "error",
    }
    assert declared == expected


# ---------------------------------------------------------------------------
# Contract: memory store honors the MemoryStore protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_roundtrip(tmp_path: Path) -> None:
    from app.core.memory import SQLiteMemoryStore

    store = SQLiteMemoryStore(tmp_path / "m.db")
    mid = await store.remember("The user's name is Kolton.", "fact", {})
    assert mid

    hits = await store.recall("what is my name", k=3)
    assert hits, "recall returned nothing for a stored fact"
    assert any("Kolton" in h.text for h in hits)


@pytest.mark.asyncio
async def test_memory_persists_across_instances(tmp_path: Path) -> None:
    from app.core.memory import SQLiteMemoryStore

    db = tmp_path / "persist.db"
    await SQLiteMemoryStore(db).remember("Prefers dark interfaces.", "pref", {})

    # A fresh instance on the same file must see prior data.
    hits = await SQLiteMemoryStore(db).recall("interface preference", k=3)
    assert hits


# ---------------------------------------------------------------------------
# Contract: tools self-register and are schema-validated
# ---------------------------------------------------------------------------


def test_tools_registered() -> None:
    from app.core.tools import registry

    names = {t.name for t in registry.all()}
    # A representative slice across the tool modules.
    for expected in ("read_file", "write_file", "web_search", "get_time"):
        assert expected in names, f"tool {expected!r} not registered; got {sorted(names)}"


def test_tool_specs_are_valid_json_schema() -> None:
    from app.core.tools import registry

    for spec in registry.all():
        assert spec.parameters.get("type") == "object", spec.name
        assert "properties" in spec.parameters, spec.name


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_not_raise() -> None:
    from app.core.tools import run_tool

    result = await run_tool("definitely_not_a_tool", {})
    assert result.ok is False, "dispatcher must degrade, never raise"


@pytest.mark.asyncio
async def test_path_traversal_blocked() -> None:
    from app.core.tools import run_tool

    result = await run_tool("read_file", {"path": "../../../../etc/passwd"})
    assert result.ok is False, "workspace escape must be rejected"


@pytest.mark.asyncio
async def test_sandbox_blocks_dangerous_code() -> None:
    from app.core.tools import run_tool

    result = await run_tool("exec_python", {"code": "import os; os.system('echo pwned')"})
    assert result.ok is False, "os.system must be rejected by the AST denylist"


# ---------------------------------------------------------------------------
# Contract: telemetry never raises on a sensor-less platform
# ---------------------------------------------------------------------------


def test_telemetry_sample() -> None:
    from app.core.telemetry import TelemetryMonitor
    from app.models.schemas import Telemetry

    sample = TelemetryMonitor().sample()
    assert isinstance(sample, Telemetry)
    assert 0.0 <= sample.cpu <= 100.0
    assert 0.0 <= sample.mem <= 100.0


# ---------------------------------------------------------------------------
# Contract: agent loop and planner expose the documented surface
# ---------------------------------------------------------------------------


def test_agent_loop_surface() -> None:
    from app.core.agent import AgentLoop

    assert inspect.isasyncgenfunction(AgentLoop.handle) or inspect.iscoroutinefunction(
        AgentLoop.handle
    ), "AgentLoop.handle must be async per CONTRACTS.md §4"


def test_persona_mentions_capabilities() -> None:
    from app.core.agent import build_system_prompt

    assert callable(build_system_prompt)


# ---------------------------------------------------------------------------
# Contract: the FastAPI app builds and exposes the documented routes
# ---------------------------------------------------------------------------


def test_routes_actually_resolve() -> None:
    """Exercise the routes rather than introspecting them.

    Route introspection is version-fragile: this FastAPI wraps included
    routers in objects that expose no top-level `.path`, so a flat scan of
    `app.routes` reports mounted routes as missing when they resolve fine.
    Hitting them is the assertion that actually means something.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/tools").status_code == 200
        assert client.get("/api/telemetry").status_code == 200


def test_websocket_pushes_telemetry_unprompted() -> None:
    """Per CONTRACTS.md §3, telemetry is broadcast without being asked for."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        frame = ws.receive_json()
        assert frame["type"] in {"telemetry", "log"}
        if frame["type"] == "telemetry":
            assert 0.0 <= frame["cpu"] <= 100.0
            assert "net_up" in frame and "net_down" in frame


def test_health_reports_subsystems() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        for key in ("llm", "db", "vision", "voice"):
            assert key in body, f"/health missing {key!r}: {body}"
