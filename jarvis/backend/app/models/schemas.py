"""Shared data contracts for J.A.R.V.I.S.

This module is the single source of truth for types crossing module boundaries.
It is READ-ONLY for build agents: import from it, never edit it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Conversation primitives
# ---------------------------------------------------------------------------


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    images: list[str] = Field(default_factory=list)  # base64 JPEG, vision turns


class LLMDelta(BaseModel):
    """One streamed chunk from a provider."""

    text: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    finished: bool = False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


class ToolResult(BaseModel):
    ok: bool
    output: str
    summary: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Planning (runs on the OpenRouter free tier)
# ---------------------------------------------------------------------------


class Step(BaseModel):
    index: int
    intent: str
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class Plan(BaseModel):
    goal: str
    steps: list[Step]
    multi_step: bool = True
    model_used: str = ""


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryKind(str, Enum):
    FACT = "fact"           # durable truth about the user
    EVENT = "event"         # something that happened
    PREFERENCE = "pref"     # how the user likes things
    CONVERSATION = "conv"   # raw dialogue turn
    OBSERVATION = "obs"     # vision / sensor derived


class MemoryHit(BaseModel):
    id: str
    text: str
    kind: MemoryKind
    score: float
    created_at: float
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sensing
# ---------------------------------------------------------------------------


class Telemetry(BaseModel):
    cpu: float
    mem: float
    disk: float
    net_up: float
    net_down: float
    battery: float | None = None
    uptime_s: int
    processes: int = 0
    temp_c: float | None = None


class DetectedObject(BaseModel):
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h


class VisionResult(BaseModel):
    objects: list[DetectedObject] = Field(default_factory=list)
    faces: int = 0
    hands: int = 0
    text: str = ""      # OCR
    scene: str = ""     # LLM scene description
    width: int = 0
    height: int = 0


# ---------------------------------------------------------------------------
# WebSocket envelope
# ---------------------------------------------------------------------------

EventType = Literal[
    "token", "done", "thinking", "tool_call", "tool_result",
    "telemetry", "vision", "speak", "log", "error",
]


class Event(BaseModel):
    type: EventType
    id: str | None = None
    text: str | None = None
    stage: str | None = None
    name: str | None = None
    ok: bool | None = None
    summary: str | None = None
    args: dict[str, Any] | None = None
    level: str | None = None
    voice: str | None = None
    data: dict[str, Any] | None = None


class TurnContext(BaseModel):
    """Everything the agent loop knows when handling one user turn."""

    turn_id: str
    history: list[Message] = Field(default_factory=list)
    recalled: list[MemoryHit] = Field(default_factory=list)
    telemetry: Telemetry | None = None
    vision: VisionResult | None = None
    last_frame_b64: str | None = None
