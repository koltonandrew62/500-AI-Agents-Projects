"""Multi-step planner running on the OpenRouter free-tier model chain.

Free-tier models are chatty: they wrap JSON in prose, in markdown fences, and —
in DeepSeek-R1's case — in `<think>` blocks. This module therefore treats model
output as hostile text to be mined, not as JSON to be parsed, and it *never*
raises into the caller: an unparseable plan degrades to a single-step plan so the
turn continues.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from app.models.schemas import Message, Plan, Role, Step, ToolSpec, TurnContext

__all__ = [
    "PLANNER_MODEL_CHAIN",
    "Planner",
    "complete_with_model",
    "extract_json_object",
    "parse_plan_payload",
    "single_step_plan",
    "strip_reasoning",
]

log = logging.getLogger("jarvis.agent.planner")

#: Ordered free-tier chain from docs/CONTRACTS.md §5.
PLANNER_MODEL_CHAIN: tuple[str, ...] = (
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
)

_MAX_STEPS = 8
_DEFAULT_TIMEOUT_S = 45.0

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*|\s*```")
_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*\Z", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


PLANNER_SYSTEM_PROMPT = """\
You are the planning subsystem of a local AI assistant. You do not talk to the user.
You convert one request into a short, ordered, executable plan.

Rules:
- Emit between 1 and 8 steps. Fewer is better. Never pad.
- Each step is one concrete action, phrased as an imperative.
- Set "tool" to a tool name ONLY if that exact name appears in the tool list you were
  given. Otherwise set it to null (reasoning or direct-answer steps need no tool).
- "args" must match that tool's parameters. Use null/omit when unknown at plan time.
- If the request is really a single action or plain conversation, return exactly one
  step and set "multi_step" to false.

Reply with ONE JSON object and nothing else. No prose, no markdown fences, no
commentary before or after. Schema:

{"goal": "<restated objective>",
 "multi_step": true,
 "steps": [{"index": 0, "intent": "<imperative action>", "tool": "<tool name or null>",
            "args": {}, "rationale": "<one short clause>"}]}"""


# ---------------------------------------------------------------------------
# Provider adaptation
# ---------------------------------------------------------------------------


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """Best-effort check that ``fn`` will tolerate keyword ``name``."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # C-implemented or exotic callable
        return True
    for param in sig.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == name and param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


async def complete_with_model(
    provider: Any,
    messages: Sequence[Message],
    model: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> str:
    """Call ``provider.complete`` pinning a model when the provider supports it.

    The contract signature is ``complete(messages) -> str``; providers that also
    accept a ``model`` keyword get the pin, everyone else gets the plain call.
    """
    fn = provider.complete
    payload = list(messages)
    use_model = bool(model) and _accepts_kwarg(fn, "model")
    try:
        if use_model:
            return await asyncio.wait_for(fn(payload, model=model), timeout=timeout_s)
        return await asyncio.wait_for(fn(payload), timeout=timeout_s)
    except TypeError:
        if not use_model:
            raise
        log.debug("provider.complete rejected model kwarg; retrying unpinned")
        return await asyncio.wait_for(fn(payload), timeout=timeout_s)


# ---------------------------------------------------------------------------
# JSON salvage
# ---------------------------------------------------------------------------


def strip_reasoning(text: str) -> str:
    """Remove `<think>` blocks and markdown fences from raw model output."""
    cleaned = _THINK_RE.sub(" ", text or "")
    cleaned = _OPEN_THINK_RE.sub(" ", cleaned)
    cleaned = _FENCE_RE.sub("\n", cleaned)
    return cleaned.strip()


def _scan_balanced(text: str, start: int, opener: str, closer: str) -> str | None:
    """Return the balanced ``opener``/``closer`` span beginning at ``start``."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _candidates(text: str) -> list[str]:
    """All balanced JSON object/array spans in ``text``, longest first."""
    found: list[str] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        for idx, ch in enumerate(text):
            if ch != opener:
                continue
            span = _scan_balanced(text, idx, opener, closer)
            if span and len(span) > 2:
                found.append(span)
    found.sort(key=len, reverse=True)
    return found


def _loads(blob: str) -> Any:
    """Parse JSON, repairing the two failures free-tier models actually make."""
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    repaired = _TRAILING_COMMA_RE.sub(r"\1", blob)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Python-literal-ish output: True/False/None and single quotes.
    swapped = (
        repaired.replace("'", '"')
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null")
    )
    return json.loads(swapped)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Mine the first plausible JSON plan object out of arbitrary model prose."""
    cleaned = strip_reasoning(text)
    if not cleaned:
        return None
    for blob in _candidates(cleaned):
        try:
            value = _loads(blob)
        except (json.JSONDecodeError, ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
            return {"steps": value}
    return None


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def single_step_plan(goal: str, *, model_used: str = "", intent: str = "") -> Plan:
    """The always-available degraded plan: answer the request directly."""
    text = " ".join((intent or goal).split()) or "Respond to the user."
    return Plan(
        goal=" ".join(goal.split()) or text,
        steps=[Step(index=0, intent=text, tool=None, args={}, rationale="direct response")],
        multi_step=False,
        model_used=model_used,
    )


def _coerce_step(raw: Any, index: int, tool_names: set[str]) -> Step | None:
    """Turn one loosely-shaped step dict into a typed :class:`Step`."""
    if isinstance(raw, str):
        raw = {"intent": raw}
    if not isinstance(raw, dict):
        return None

    intent = raw.get("intent") or raw.get("action") or raw.get("description") or raw.get("step")
    if not isinstance(intent, str) or not intent.strip():
        return None

    tool = raw.get("tool") or raw.get("tool_name") or raw.get("name")
    if not isinstance(tool, str) or not tool.strip():
        tool = None
    elif tool_names and tool not in tool_names:
        log.debug("planner hallucinated tool %r; demoting step to reasoning", tool)
        tool = None

    args = raw.get("args") or raw.get("arguments") or raw.get("params") or {}
    if not isinstance(args, dict):
        args = {"value": args}

    rationale = raw.get("rationale") or raw.get("why") or raw.get("reason") or ""
    if not isinstance(rationale, str):
        rationale = str(rationale)

    return Step(
        index=index,
        intent=" ".join(intent.split()),
        tool=tool,
        args=args,
        rationale=" ".join(rationale.split())[:280],
    )


def parse_plan_payload(
    payload: dict[str, Any],
    *,
    goal_fallback: str,
    tool_names: set[str] | None = None,
    model_used: str = "",
    max_steps: int = _MAX_STEPS,
) -> Plan | None:
    """Validate and normalise a raw plan dict into a typed :class:`Plan`."""
    raw_steps = payload.get("steps") or payload.get("plan") or payload.get("actions")
    if isinstance(raw_steps, dict):
        raw_steps = list(raw_steps.values())
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    names = tool_names or set()
    steps: list[Step] = []
    for raw in raw_steps:
        step = _coerce_step(raw, len(steps), names)
        if step is not None:
            steps.append(step)
        if len(steps) >= max_steps:
            break
    if not steps:
        return None

    goal = payload.get("goal") or payload.get("objective") or goal_fallback
    if not isinstance(goal, str) or not goal.strip():
        goal = goal_fallback

    declared = payload.get("multi_step")
    multi = bool(declared) if isinstance(declared, bool) else len(steps) > 1
    return Plan(
        goal=" ".join(goal.split()),
        steps=steps,
        multi_step=multi and len(steps) > 1,
        model_used=model_used,
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """Produces a typed :class:`Plan` from a user request.

    Walks :data:`PLANNER_MODEL_CHAIN` in order, taking the first model that both
    responds and yields a parseable plan. Rate limits, timeouts and garbage
    output all fall through to the next model; exhausting the chain yields a
    single-step plan rather than an exception.
    """

    def __init__(
        self,
        provider: Any,
        *,
        models: Sequence[str] = PLANNER_MODEL_CHAIN,
        max_steps: int = _MAX_STEPS,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        self._provider = provider
        self._models: tuple[str, ...] = tuple(models) or PLANNER_MODEL_CHAIN
        self._max_steps = max(1, max_steps)
        self._timeout_s = timeout_s

    @property
    def models(self) -> tuple[str, ...]:
        """The model chain this planner will walk, in order."""
        return self._models

    async def plan(
        self,
        user_text: str,
        ctx: TurnContext | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> Plan:
        """Plan ``user_text``. Always returns a usable Plan; never raises."""
        goal = " ".join(user_text.split())
        if not goal:
            return single_step_plan("Respond to the user.", model_used="empty-input")
        if self._provider is None:
            return single_step_plan(goal, model_used="no-provider")

        tool_names = {spec.name for spec in (tools or [])}
        messages = self._build_messages(goal, ctx, tools)

        for model in self._models:
            try:
                raw = await complete_with_model(
                    self._provider, messages, model, timeout_s=self._timeout_s
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                log.warning("planner model %s timed out after %.0fs", model, self._timeout_s)
                continue
            except Exception as exc:  # provider/network/rate-limit — try the next model
                log.warning("planner model %s failed: %s", model, exc)
                continue

            payload = extract_json_object(raw or "")
            if payload is None:
                log.warning("planner model %s returned no parseable JSON", model)
                continue

            plan = parse_plan_payload(
                payload,
                goal_fallback=goal,
                tool_names=tool_names,
                model_used=model,
                max_steps=self._max_steps,
            )
            if plan is None:
                log.warning("planner model %s returned an unusable plan shape", model)
                continue
            log.info("plan from %s: %d step(s)", model, len(plan.steps))
            return plan

        log.warning("planner chain exhausted; falling back to a single step")
        return single_step_plan(goal, model_used="fallback")

    # -- prompt assembly ---------------------------------------------------

    def _build_messages(
        self,
        goal: str,
        ctx: TurnContext | None,
        tools: Sequence[ToolSpec] | None,
    ) -> list[Message]:
        """Compose the planning conversation (system + one dense user turn)."""
        blocks: list[str] = []

        if tools:
            lines = ["AVAILABLE TOOLS (use these exact names or null):"]
            for spec in tools:
                params = sorted((spec.parameters or {}).get("properties", {}).keys())
                sig = ", ".join(params) if params else "no parameters"
                desc = " ".join(spec.description.split())
                lines.append(f"- {spec.name}({sig}) — {desc}")
            blocks.append("\n".join(lines))
        else:
            blocks.append("AVAILABLE TOOLS: none. Every step must have \"tool\": null.")

        if ctx is not None:
            if ctx.recalled:
                known = "; ".join(
                    " ".join(hit.text.split())[:160] for hit in ctx.recalled[:5]
                )
                blocks.append(f"KNOWN CONTEXT ABOUT THE USER: {known}")
            if ctx.vision is not None and (ctx.vision.scene or ctx.vision.text):
                seen = ctx.vision.scene or ctx.vision.text
                blocks.append(f"CAMERA SEES: {' '.join(seen.split())[:300]}")
            if ctx.history:
                recent = [
                    f"{m.role.value}: {' '.join(m.content.split())[:200]}"
                    for m in ctx.history[-4:]
                    if m.content
                ]
                if recent:
                    blocks.append("RECENT CONVERSATION:\n" + "\n".join(recent))

        blocks.append(f"REQUEST TO PLAN:\n{goal}")
        blocks.append("Return only the JSON object.")

        return [
            Message(role=Role.SYSTEM, content=PLANNER_SYSTEM_PROMPT),
            Message(role=Role.USER, content="\n\n".join(blocks)),
        ]
