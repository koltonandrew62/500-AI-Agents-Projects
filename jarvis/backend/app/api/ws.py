"""The `/ws` endpoint -- docs/CONTRACTS.md section 3, implemented exactly.

One duplex JSON channel per client. `ConnectionManager` tracks every
connected client and supports fan-out broadcast (telemetry). Each connection
runs at most one turn at a time: a new `chat`/`voice` message cooperatively
cancels whatever turn is still in flight before starting the next one, via
`AgentLoop.request_cancel` plus a hard `asyncio.Task.cancel()` fallback.

Isolation: every inbound message is dispatched inside a try/except that turns
any failure into an `error` event for that one client. One client's bug or
disconnect must never affect another client or kill the broadcast loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.schemas import Event, TurnContext

log = logging.getLogger("jarvis.api.ws")

router = APIRouter()


def serialize_event(event: Event) -> dict[str, Any]:
    """Event -> wire JSON, flattening `data` onto the top level.

    `Event` (schemas.py, read-only) models every server->client frame with a
    fixed set of named fields plus a catch-all `data` dict for anything
    protocol-specific (e.g. the `vision` event's `objects`/`faces`/`scene`,
    or `telemetry`'s `cpu`/`mem`/...). CONTRACTS.md section 3 puts those
    fields directly on the frame, so this flattens `data` up before send.
    """
    payload = event.model_dump(exclude_none=True)
    data = payload.pop("data", None)
    if isinstance(data, dict):
        payload.update(data)
    return payload


@dataclass(slots=True)
class _ClientState:
    websocket: WebSocket
    ctx: TurnContext
    task: asyncio.Task[Any] | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ConnectionManager:
    """Tracks connected clients for per-connection turns and broadcast."""

    def __init__(self) -> None:
        self._clients: dict[int, _ClientState] = {}

    async def connect(self, websocket: WebSocket) -> int:
        await websocket.accept()
        cid = id(websocket)
        self._clients[cid] = _ClientState(websocket=websocket, ctx=TurnContext(turn_id=""))
        return cid

    def disconnect(self, cid: int) -> None:
        state = self._clients.pop(cid, None)
        if state is not None and state.task is not None and not state.task.done():
            state.task.cancel()

    def state_for(self, cid: int) -> _ClientState | None:
        return self._clients.get(cid)

    def count(self) -> int:
        return len(self._clients)

    async def broadcast(self, event: Event) -> None:
        """Send `event` to every connected client. Dead sockets are dropped."""
        payload = serialize_event(event)
        dead: list[int] = []
        for cid, state in list(self._clients.items()):
            try:
                await state.websocket.send_json(payload)
            except Exception:
                dead.append(cid)
        for cid in dead:
            self.disconnect(cid)


async def _send(state: _ClientState, event: Event) -> None:
    """Best-effort send -- a client that vanished mid-turn must not raise."""
    try:
        await state.websocket.send_json(serialize_event(event))
    except Exception:
        log.debug("failed to deliver %s event; client likely disconnected", event.type)


async def _cancel_current(state: _ClientState, agent_loop: Any) -> None:
    """Cooperatively then forcibly stop whatever turn is still in flight."""
    task = state.task
    if task is None or task.done():
        return
    if state.ctx.turn_id:
        agent_loop.request_cancel(state.ctx.turn_id)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        log.debug("previous turn raised while being cancelled", exc_info=True)


async def _run_turn(text: str, turn_id: str, state: _ClientState, agent_loop: Any) -> None:
    try:
        async for event in agent_loop.handle(text, state.ctx):
            await _send(state, event)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("agent loop raised for turn %s", turn_id)
        await _send(state, Event(type="error", id=turn_id, text=str(exc)))
        await _send(state, Event(type="done", id=turn_id, text=""))


async def _handle_turn(msg: dict[str, Any], state: _ClientState, agent_loop: Any) -> None:
    text = str(msg.get("text") or "")
    turn_id = str(msg.get("id") or uuid.uuid4().hex)

    async with state.lock:
        await _cancel_current(state, agent_loop)
        state.ctx.turn_id = turn_id
        state.task = asyncio.create_task(_run_turn(text, turn_id, state, agent_loop))


def _handle_cancel(msg: dict[str, Any], state: _ClientState, agent_loop: Any) -> None:
    turn_id = msg.get("id")
    if turn_id:
        agent_loop.request_cancel(str(turn_id))
    elif state.ctx.turn_id:
        agent_loop.request_cancel(state.ctx.turn_id)
    task = state.task
    if task is not None and not task.done():
        task.cancel()


async def _handle_frame(msg: dict[str, Any], state: _ClientState, vision_pipeline: Any) -> None:
    jpeg_b64 = msg.get("jpeg_b64")
    frame_id = msg.get("id")
    if not isinstance(jpeg_b64, str) or not jpeg_b64:
        await _send(state, Event(type="error", id=frame_id, text="frame message missing jpeg_b64"))
        return

    # Stash immediately so the next chat turn sees the frame even if
    # analysis below fails or is disabled.
    state.ctx.last_frame_b64 = jpeg_b64

    if vision_pipeline is None:
        return
    try:
        result = await vision_pipeline.analyze(jpeg_b64)
    except Exception:
        log.warning("vision analyze failed", exc_info=True)
        return

    state.ctx.vision = result
    await _send(
        state,
        Event(
            type="vision",
            id=frame_id,
            data={
                "objects": [obj.model_dump() for obj in result.objects],
                "faces": result.faces,
                "hands": result.hands,
                "text": result.text,
                "scene": result.scene,
            },
        ),
    )


async def _dispatch(
    raw: str, cid: int, manager: ConnectionManager, agent_loop: Any, vision_pipeline: Any
) -> None:
    state = manager.state_for(cid)
    if state is None:
        return

    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await _send(state, Event(type="error", text="invalid JSON"))
        return
    if not isinstance(msg, dict) or "type" not in msg:
        await _send(state, Event(type="error", text="message must be a JSON object with a 'type'"))
        return

    mtype = msg.get("type")
    try:
        if mtype in ("chat", "voice"):
            await _handle_turn(msg, state, agent_loop)
        elif mtype == "frame":
            await _handle_frame(msg, state, vision_pipeline)
        elif mtype == "cancel":
            _handle_cancel(msg, state, agent_loop)
        elif mtype == "ping":
            pass
        else:
            await _send(state, Event(type="error", id=msg.get("id"), text=f"unknown message type: {mtype!r}"))
    except Exception as exc:
        log.exception("error handling %r message from client %s", mtype, cid)
        await _send(state, Event(type="error", id=msg.get("id"), text=str(exc)))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    manager: ConnectionManager = websocket.app.state.connection_manager
    agent_loop = websocket.app.state.agent_loop
    vision_pipeline = getattr(websocket.app.state, "vision_pipeline", None)

    cid = await manager.connect(websocket)
    log.info("client %s connected (%d total)", cid, manager.count())
    try:
        while True:
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except RuntimeError:
                # Socket already closed out from under us.
                break
            await _dispatch(raw, cid, manager, agent_loop, vision_pipeline)
    except Exception:
        log.exception("ws loop crashed unexpectedly for client %s", cid)
    finally:
        manager.disconnect(cid)
        log.info("client %s disconnected (%d total)", cid, manager.count())
