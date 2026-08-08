"""The reasoning core of J.A.R.V.I.S.

Public surface:

- `AgentLoop`      — per-turn orchestration: recall, classify, plan, tool-loop, stream.
- `Planner`        — multi-step planning on the OpenRouter free tier (see CONTRACTS.md §5).
- `TurnClassifier` — decides whether a turn is worth planning at all.
- `build_system_prompt` — assembles the persona with live context injected.
"""

from __future__ import annotations

from .classifier import Classification, TurnClassifier, classify_heuristic, needs_vision
from .loop import AgentLoop
from .persona import build_system_prompt, degraded_reply, format_telemetry, format_vision
from .planner import Planner, single_step_plan

__all__ = [
    "AgentLoop",
    "Planner",
    "TurnClassifier",
    "Classification",
    "build_system_prompt",
    "classify_heuristic",
    "degraded_reply",
    "format_telemetry",
    "format_vision",
    "needs_vision",
    "single_step_plan",
]
