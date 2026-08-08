"""The agent loop — JARVIS's turn-handling brain.

One turn runs: recall memory -> classify simple vs multi-step -> plan when it is
multi-step -> optionally look through the camera -> then a bounded tool-calling
loop that streams tokens out as they arrive. Everything is emitted as
:class:`~app.models.schemas.Event` objects over an async generator, which the
WebSocket layer forwards verbatim.

Failure policy: the loop degrades, it does not hard-fail. Memory, planner, vision,
tools and even the streaming provider may each break independently; the user still
gets a reply and a `done` event.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import (
    Event,
    Message,
    Plan,
    Role,
    ToolResult,
    ToolSpec,
    TurnContext,
    VisionResult,
)

from app.core.agent.classifier import Classification, TurnClassifier, needs_vision
from app.core.agent.persona import build_system_prompt, degraded_reply
from app.core.agent.planner import Planner

__all__ = ["AgentLoop"]

log = logging.getLogger("jarvis.agent.loop")

MAX_ITERATIONS = 8
_TOOL_TIMEOUT_S = 30.0
_RECALL_TIMEOUT_S = 6.0
_VISION_TIMEOUT_S = 45.0
_PERSIST_TIMEOUT_S = 5.0
_REPEAT_LIMIT = 3
_SUMMARY_CHARS = 240
_TOOL_OUTPUT_CHARS = 8_000


@dataclass(slots=True)
class _PendingCall:
    """A tool invocation accumulated from one or more provider deltas."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def signature(self) -> str:
        try:
            return f"{self.name}:{json.dumps(self.args, sort_keys=True, default=str)}"
        except (TypeError, ValueError):
            return f"{self.name}:{self.args!r}"


@dataclass(slots=True)
class _StreamOutcome:
    """Everything one streaming pass produced."""

    text: str = ""
    calls: list[_PendingCall] = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False


class AgentLoop:
    """Handles one user turn end to end, yielding protocol events as it goes.

    Args:
        provider: An ``LLMProvider`` (``stream`` / ``complete`` / ``vision``).
        memory: Optional ``MemoryStore`` used for recall, history and persistence.
        tools: A tool registry, a zero-arg callable returning tools, or a plain
            sequence of tools. Defaults to ``core.tools.base.registry``.
        planner: Override for the :class:`Planner`.
        classifier: Override for the :class:`TurnClassifier`.
        max_iterations: Hard ceiling on tool-calling rounds per turn.
    """

    def __init__(
        self,
        provider: Any,
        *,
        memory: Any = None,
        tools: Any = None,
        planner: Planner | None = None,
        classifier: TurnClassifier | None = None,
        max_iterations: int = MAX_ITERATIONS,
        recall_k: int = 6,
        persist_memory: bool = True,
        emit_speak: bool = False,
        tool_timeout_s: float = _TOOL_TIMEOUT_S,
    ) -> None:
        self._provider = provider
        self._memory = memory
        self._tools_source = tools
        self._planner = planner or Planner(provider)
        self._classifier = classifier or TurnClassifier(provider)
        self._max_iterations = max(1, max_iterations)
        self._recall_k = max(0, recall_k)
        self._persist_memory = persist_memory
        self._emit_speak = emit_speak
        self._tool_timeout_s = tool_timeout_s
        self._cancelled: set[str] = set()

    # -- cancellation ------------------------------------------------------

    def request_cancel(self, turn_id: str) -> None:
        """Ask the turn to stop cooperatively at the next checkpoint."""
        self._cancelled.add(turn_id)

    def cancel_all(self) -> None:
        """Cancel every turn currently in flight."""
        self._cancelled.add("*")

    def is_cancelled(self, turn_id: str) -> bool:
        """True when this turn has been asked to stop."""
        return "*" in self._cancelled or turn_id in self._cancelled

    # -- public entry point ------------------------------------------------

    async def handle(self, user_text: str, ctx: TurnContext) -> AsyncIterator[Event]:
        """Run one turn, yielding :class:`Event` objects until ``done``."""
        try:
            async for event in self._run(user_text, ctx):
                yield event
        except asyncio.CancelledError:
            log.info("turn %s cancelled by the event loop", ctx.turn_id)
            raise
        finally:
            self._cancelled.discard(ctx.turn_id)

    # -- turn body ---------------------------------------------------------

    async def _run(self, user_text: str, ctx: TurnContext) -> AsyncIterator[Event]:
        turn = ctx.turn_id
        text = (user_text or "").strip()
        if not text:
            yield Event(type="done", id=turn, text="")
            return

        # 1. Recall -------------------------------------------------------
        yield Event(type="thinking", id=turn, stage="recall", text="Consulting memory")
        await self._recall(text, ctx)
        if self.is_cancelled(turn):
            yield self._cancel_done(turn)
            return

        tools, specs = await self._resolve_tools()

        # 2. Classify + 3. Plan -------------------------------------------
        verdict = await self._classify(text, ctx)
        plan: Plan | None = None
        if verdict.multi_step:
            yield Event(
                type="thinking", id=turn, stage="planning", text="Working out an approach"
            )
            plan = await self._plan(text, ctx, specs)
            if plan is not None and plan.steps:
                yield Event(
                    type="thinking",
                    id=turn,
                    stage="planning",
                    text=f"{len(plan.steps)} step(s) via {plan.model_used or 'fallback'}",
                    data={"plan": plan.model_dump()},
                )
        if self.is_cancelled(turn):
            yield self._cancel_done(turn)
            return

        # 4. Vision -------------------------------------------------------
        if self._should_look(text, ctx):
            yield Event(type="thinking", id=turn, stage="vision", text="Taking a look")
            described = await self._describe_frame(text, ctx)
            if described:
                yield Event(type="vision", id=turn, text=described, data={"scene": described})

        # 5. Tool-calling / streaming loop --------------------------------
        messages = self._build_messages(text, ctx, specs, plan)
        reply = ""
        seen: dict[str, int] = {}

        for iteration in range(self._max_iterations):
            if self.is_cancelled(turn):
                yield self._cancel_done(turn, reply)
                return

            outcome = _StreamOutcome()
            last = iteration == self._max_iterations - 1
            async for event in self._stream(messages, specs if not last else None, ctx, outcome):
                yield event

            if outcome.text:
                reply = outcome.text
            if outcome.cancelled:
                yield self._cancel_done(turn, reply)
                return
            if outcome.error and not outcome.calls:
                reply = await self._salvage(messages, ctx, outcome.error, reply)
                break
            if not outcome.calls:
                break

            messages.append(
                Message(role=Role.ASSISTANT, content=outcome.text or "", name="jarvis")
            )
            for call in outcome.calls:
                if self.is_cancelled(turn):
                    yield self._cancel_done(turn, reply)
                    return
                yield Event(type="tool_call", id=turn, name=call.name, args=dict(call.args))
                sig = call.signature()
                seen[sig] = seen.get(sig, 0) + 1
                if seen[sig] > _REPEAT_LIMIT:
                    result = ToolResult(
                        ok=False,
                        output=(
                            f"Refused: `{call.name}` has already been called with these "
                            "exact arguments. Use the result you already have, or change "
                            "approach."
                        ),
                        summary="repeated call suppressed",
                    )
                else:
                    result = await self._invoke(tools, call)
                yield Event(
                    type="tool_result",
                    id=turn,
                    name=call.name,
                    ok=result.ok,
                    summary=result.summary or _clip(result.output, _SUMMARY_CHARS),
                )
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=_clip(result.output, _TOOL_OUTPUT_CHARS),
                        name=call.name,
                        tool_call_id=call.id,
                    )
                )
        else:  # pragma: no cover - only when every iteration made tool calls
            log.warning("turn %s hit the %d-iteration ceiling", turn, self._max_iterations)

        if not reply.strip():
            reply = degraded_reply()

        await self._persist(text, reply, ctx)

        if self._emit_speak:
            yield Event(type="speak", id=turn, text=reply, voice="jarvis")
        yield Event(type="done", id=turn, text=reply)

    # -- stage helpers -----------------------------------------------------

    def _cancel_done(self, turn: str, partial: str = "") -> Event:
        """Terminal event for a cooperatively cancelled turn."""
        log.info("turn %s stopped on request", turn)
        return Event(type="done", id=turn, text=partial, summary="cancelled")

    async def _recall(self, text: str, ctx: TurnContext) -> None:
        """Populate ``ctx.recalled`` and, when empty, ``ctx.history``."""
        if self._memory is None or self._recall_k == 0:
            return
        try:
            hits = await asyncio.wait_for(
                self._memory.recall(text, k=self._recall_k), timeout=_RECALL_TIMEOUT_S
            )
            ctx.recalled = list(hits or [])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("memory recall failed: %s", exc)

        if ctx.history:
            return
        history = getattr(self._memory, "history", None)
        if history is None:
            return
        try:
            past = await asyncio.wait_for(history(limit=20), timeout=_RECALL_TIMEOUT_S)
            ctx.history = list(past or [])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("memory history failed: %s", exc)

    async def _classify(self, text: str, ctx: TurnContext) -> Classification:
        """Decide whether the planner is worth engaging."""
        try:
            verdict = await self._classifier.classify(text, ctx)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("classifier failed, assuming simple turn: %s", exc)
            return Classification(False, 0, f"classifier error: {exc}", "default")
        log.debug("turn classified %s (%s)", verdict.multi_step, verdict.reason)
        return verdict

    async def _plan(
        self, text: str, ctx: TurnContext, specs: Sequence[ToolSpec]
    ) -> Plan | None:
        """Run the planner, tolerating any failure."""
        try:
            plan = await self._planner.plan(text, ctx, specs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("planner failed: %s", exc)
            return None
        return plan if plan and plan.steps else None

    def _should_look(self, text: str, ctx: TurnContext) -> bool:
        """True when a fresh camera read would actually add something."""
        if not ctx.last_frame_b64 or self._provider is None:
            return False
        if not hasattr(self._provider, "vision"):
            return False
        if ctx.vision is not None and ctx.vision.scene.strip():
            return False
        return needs_vision(text)

    async def _describe_frame(self, text: str, ctx: TurnContext) -> str:
        """Ask the provider to describe the current frame; returns "" on failure."""
        prompt = [
            Message(
                role=Role.SYSTEM,
                content=(
                    "Describe this webcam frame factually and concisely for another "
                    "assistant to reason over: people, objects, activity, legible text, "
                    "setting. No speculation, no commentary."
                ),
            ),
            Message(role=Role.USER, content=" ".join(text.split())[:500]),
        ]
        try:
            scene = await asyncio.wait_for(
                self._provider.vision(prompt, ctx.last_frame_b64 or ""),
                timeout=_VISION_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("vision pass failed: %s", exc)
            return ""

        scene = (scene or "").strip()
        if not scene:
            return ""
        if ctx.vision is None:
            ctx.vision = VisionResult(scene=scene)
        else:
            ctx.vision.scene = scene
        return scene

    def _build_messages(
        self,
        text: str,
        ctx: TurnContext,
        specs: Sequence[ToolSpec],
        plan: Plan | None,
    ) -> list[Message]:
        """System prompt + prior turns + the current user message."""
        system = build_system_prompt(ctx, tools=specs, plan=plan)
        messages = [Message(role=Role.SYSTEM, content=system)]
        messages.extend(m for m in ctx.history if m.role is not Role.SYSTEM)

        already = (
            messages[-1].role is Role.USER and messages[-1].content.strip() == text
            if len(messages) > 1
            else False
        )
        if not already:
            messages.append(
                Message(
                    role=Role.USER,
                    content=text,
                    images=[ctx.last_frame_b64] if ctx.last_frame_b64 else [],
                )
            )
        return messages

    # -- streaming ---------------------------------------------------------

    async def _stream(
        self,
        messages: Sequence[Message],
        specs: Sequence[ToolSpec] | None,
        ctx: TurnContext,
        outcome: _StreamOutcome,
    ) -> AsyncIterator[Event]:
        """One provider pass: emit `token` events, collect text and tool calls."""
        turn = ctx.turn_id
        pending: _PendingCall | None = None
        chunks: list[str] = []

        try:
            stream = self._provider.stream(list(messages), list(specs) if specs else None)
            if inspect.isawaitable(stream):
                stream = await stream
            async for delta in stream:
                if self.is_cancelled(turn):
                    outcome.cancelled = True
                    break
                if delta.tool_name:
                    args = delta.tool_args or {}
                    if (
                        pending is not None
                        and pending.name == delta.tool_name
                        and not (set(args) & set(pending.args))
                    ):
                        pending.args.update(args)
                    else:
                        if pending is not None:
                            outcome.calls.append(pending)
                        pending = _PendingCall(name=delta.tool_name, args=dict(args))
                elif delta.tool_args and pending is not None:
                    pending.args.update(delta.tool_args)
                if delta.text:
                    chunks.append(delta.text)
                    yield Event(type="token", id=turn, text=delta.text)
                if delta.finished:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            outcome.error = str(exc) or exc.__class__.__name__
            log.warning("provider stream failed: %s", exc)
            yield Event(
                type="log",
                id=turn,
                level="warn",
                text=f"stream error: {outcome.error}",
            )

        if pending is not None:
            outcome.calls.append(pending)
        outcome.text = "".join(chunks).strip()

    async def _salvage(
        self,
        messages: Sequence[Message],
        ctx: TurnContext,
        error: str,
        partial: str,
    ) -> str:
        """Last resort when streaming broke: try a non-streaming completion."""
        if partial.strip():
            return partial
        try:
            text = await asyncio.wait_for(
                self._provider.complete(list(messages)), timeout=_VISION_TIMEOUT_S
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("non-streaming salvage also failed: %s", exc)
            return degraded_reply(error)
        return (text or "").strip() or degraded_reply(error)

    # -- tools -------------------------------------------------------------

    async def _resolve_tools(self) -> tuple[dict[str, Any], list[ToolSpec]]:
        """Discover tools through the registry — never a hardcoded list."""
        objects: Any = self._tools_source
        if objects is None:
            try:
                from app.core.tools.base import registry  # local: sibling may lag

                objects = registry
            except Exception as exc:
                log.warning("tool registry unavailable: %s", exc)
                return {}, []
        try:
            if hasattr(objects, "all"):
                objects = objects.all()
            elif callable(objects) and not isinstance(objects, (list, tuple, dict)):
                objects = objects()
            if inspect.isawaitable(objects):
                objects = await objects
            if isinstance(objects, dict):
                objects = list(objects.values())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("tool discovery failed: %s", exc)
            return {}, []

        tools: dict[str, Any] = {}
        specs: list[ToolSpec] = []
        for tool in list(objects or []):
            spec = _to_spec(tool)
            if spec is None:
                continue
            tools[spec.name] = tool
            specs.append(spec)
        return tools, specs

    async def _invoke(self, tools: dict[str, Any], call: _PendingCall) -> ToolResult:
        """Run one tool. Every failure mode becomes a failed ToolResult."""
        tool = tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(tools)) or "none"
            return ToolResult(
                ok=False,
                output=f"No tool named `{call.name}`. Available: {available}.",
                summary="unknown tool",
            )
        try:
            raw = await asyncio.wait_for(tool.run(**call.args), timeout=self._tool_timeout_s)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            return ToolResult(
                ok=False,
                output=f"`{call.name}` timed out after {self._tool_timeout_s:.0f}s.",
                summary="timed out",
            )
        except TypeError as exc:
            return ToolResult(
                ok=False,
                output=f"`{call.name}` rejected those arguments: {exc}",
                summary="bad arguments",
            )
        except Exception as exc:
            log.warning("tool %s raised: %s", call.name, exc)
            return ToolResult(
                ok=False,
                output=f"`{call.name}` failed: {exc}",
                summary=f"{exc.__class__.__name__}",
            )
        return _to_result(raw)

    # -- persistence -------------------------------------------------------

    async def _persist(self, user_text: str, reply: str, ctx: TurnContext) -> None:
        """Write the exchange back to memory, best effort."""
        if not self._persist_memory or self._memory is None:
            return
        meta = {"turn_id": ctx.turn_id}
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    self._memory.remember(user_text, "conv", {**meta, "role": "user"}),
                    self._memory.remember(reply, "conv", {**meta, "role": "assistant"}),
                ),
                timeout=_PERSIST_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("memory persist failed: %s", exc)


# ---------------------------------------------------------------------------
# Adapters for loosely-typed sibling modules
# ---------------------------------------------------------------------------


def _clip(text: str, limit: int) -> str:
    """Truncate with an ellipsis, never mid-surrogate."""
    text = text or ""
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _to_spec(tool: Any) -> ToolSpec | None:
    """Build a :class:`ToolSpec` from any reasonable tool object."""
    existing = getattr(tool, "spec", None)
    if isinstance(existing, ToolSpec):
        return existing
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        return None
    params = getattr(tool, "schema", None) or getattr(tool, "parameters", None)
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    return ToolSpec(
        name=name,
        description=str(getattr(tool, "description", "") or name),
        parameters=params,
    )


def _to_result(raw: Any) -> ToolResult:
    """Coerce whatever a tool returned into a :class:`ToolResult`."""
    if isinstance(raw, ToolResult):
        return raw
    if isinstance(raw, dict) and "output" in raw:
        return ToolResult(
            ok=bool(raw.get("ok", True)),
            output=str(raw.get("output", "")),
            summary=str(raw.get("summary", "")),
            meta=raw.get("meta") if isinstance(raw.get("meta"), dict) else {},
        )
    text = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    return ToolResult(ok=True, output=text, summary=_clip(text, _SUMMARY_CHARS))
