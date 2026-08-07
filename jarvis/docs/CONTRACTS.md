# J.A.R.V.I.S. — Shared Build Contracts

**Every agent MUST read this file before writing code.** It defines the interfaces
between modules so ten agents can build in parallel without colliding.

Isolation rule: this project imports nothing from FinGPT, trading, or research code.
It is entirely standalone.

---

## 1. Stack

| Layer | Choice |
|-------|--------|
| Backend | Python 3.11+, FastAPI, uvicorn, WebSockets |
| LLM | OpenRouter free tier (DeepSeek-R1 primary, Qwen3-235B fallback) |
| Memory | SQLite + sentence-transformers embeddings, cosine recall |
| Vision | OpenCV + MediaPipe (faces/hands), RapidOCR (text), LLM vision for scenes |
| Telemetry | psutil |
| Frontend | React 18 + TypeScript + Vite |
| 3D core | three.js (particle sphere) |
| HUD | Canvas 2D + SVG, no UI framework |

---

## 2. Directory ownership (DO NOT WRITE OUTSIDE YOUR OWN)

| Group | Agent | Owns exactly |
|-------|-------|--------------|
| CODING | `core-agent` | `backend/app/core/agent/` |
| CODING | `core-llm` | `backend/app/core/llm/` |
| FILES | `files-tools` | `backend/app/core/tools/` |
| FILES | `files-config` | `backend/app/config.py`, `backend/requirements.txt`, `.env.example` |
| MEMORY | `memory-store` | `backend/app/core/memory/` |
| MEMORY | `memory-profile` | `backend/app/core/memory/profile.py`, `backend/app/core/memory/recall.py` |
| FRONTEND | `fe-hud` | `frontend/src/hud/` |
| FRONTEND | `fe-shell` | `frontend/src/{App.tsx,main.tsx,panels,state,styles}` |
| BACKEND | `be-api` | `backend/app/{main.py,api/}` |
| BACKEND | `be-sense` | `backend/app/core/{vision,voice,telemetry}/` |

`backend/app/models/schemas.py` is the shared contract file — **read-only for
everyone**; it is written once by the lead before agents start.

---

## 3. WebSocket protocol (`/ws`)

Single duplex channel. Every frame is JSON with a `type` discriminator.

### Client → Server

```jsonc
{ "type": "chat",     "text": "what am I holding?", "id": "uuid" }
{ "type": "frame",    "jpeg_b64": "...", "id": "uuid" }   // webcam still
{ "type": "voice",    "text": "transcript from browser STT", "id": "uuid" }
{ "type": "cancel",   "id": "uuid" }
{ "type": "ping" }
```

### Server → Client

```jsonc
{ "type": "token",     "id": "uuid", "text": "partial " }       // stream chunk
{ "type": "done",      "id": "uuid", "text": "full reply" }
{ "type": "thinking",  "id": "uuid", "stage": "planning|tool|vision|recall" }
{ "type": "tool_call", "id": "uuid", "name": "read_file", "args": {} }
{ "type": "tool_result","id": "uuid","name": "read_file", "ok": true, "summary": "" }
{ "type": "telemetry", "cpu": 41.2, "mem": 63.0, "disk": 22.1, "net_up": 0.4,
                       "net_down": 2.1, "battery": 88, "uptime_s": 91234 }
{ "type": "vision",    "objects": [], "faces": 0, "text": "", "scene": "" }
{ "type": "speak",     "text": "...", "voice": "jarvis" }       // TTS cue
{ "type": "log",       "level": "info", "text": "..." }         // HUD console
{ "type": "error",     "id": "uuid", "text": "..." }
```

Telemetry is pushed unprompted every 1000 ms. Vision events fire only after a
`frame` message.

---

## 4. Core Python interfaces

```python
# core/llm/provider.py
class LLMProvider(Protocol):
    async def stream(self, messages: list[Message], tools: list[ToolSpec] | None
                     ) -> AsyncIterator[LLMDelta]: ...
    async def complete(self, messages: list[Message]) -> str: ...
    async def vision(self, messages: list[Message], image_b64: str) -> str: ...

# core/agent/loop.py
class AgentLoop:
    async def handle(self, user_text: str, ctx: TurnContext) -> AsyncIterator[Event]: ...

# core/memory/store.py
class MemoryStore(Protocol):
    async def remember(self, text: str, kind: str, meta: dict) -> str: ...
    async def recall(self, query: str, k: int = 6) -> list[MemoryHit]: ...
    async def history(self, limit: int = 20) -> list[Message]: ...

# core/tools/base.py
class Tool(Protocol):
    name: str
    description: str
    schema: dict          # JSON Schema for params
    async def run(self, **kwargs) -> ToolResult: ...
```

Every tool self-registers via the `@register` decorator in `core/tools/base.py`.
The agent loop discovers tools through `registry.all()` — never by hardcoding.

---

## 5. Planning on the free API

Planning runs on OpenRouter's free tier, **not** Gemini. Model chain, in order:

1. `deepseek/deepseek-r1:free` — deep multi-step planning
2. `qwen/qwen3-235b-a22b:free` — fallback on rate limit
3. `meta-llama/llama-3.3-70b-instruct:free` — last resort

The planner emits a typed `Plan` (see `models/schemas.py`) with ordered `Step`s.
Fast conversational turns skip the planner entirely — it only engages when the
agent loop classifies a request as multi-step.

---

## 6. Frontend design tokens (authoritative — do not invent colors)

```css
--j-bg:        #030a1a;   /* deep navy void          */
--j-bg-2:      #05122b;   /* panel fill              */
--j-cyan:      #4fd8ff;   /* primary HUD line        */
--j-cyan-dim:  #1c6f92;   /* inactive line           */
--j-blue:      #1b8fe0;   /* secondary               */
--j-glow:      #7ee8ff;   /* emissive / text glow    */
--j-amber:     #ffb545;   /* warning                 */
--j-red:       #ff4d5e;   /* critical                */
--j-grid:      rgba(79,216,255,0.10);
--j-text:      #b8ecff;
--j-text-dim:  #4d7f9c;
font: 'Rajdhani', 'Share Tech Mono', ui-monospace, monospace;
```

All HUD text is uppercase, letter-spaced `0.12em`. Every panel has a corner-bracket
frame (not a plain border). Nothing uses rounded rectangles except the core reticle.

### Layout map (matches the reference HUD)

```
┌──────────────────────────────────────────────────────────────┐
│ [brand + subtitle]                            [clock/date]   │
│ [status chips]                                [geo/coords]   │
│                                                              │
│ ┌SYSTEM VITALS┐        ╭───────────╮         ┌─RADAR/GLOBE─┐ │
│ │ bars        │      ╭─┤  PARTICLE ├─╮       └─────────────┘ │
│ └─────────────┘      │ │   CORE    │ │       ┌─WAVEFORM────┐ │
│ ┌TELEMETRY────┐      ╰─┤  SPHERE   ├─╯       └─────────────┘ │
│ │ scroll list │        ╰───────────╯         ┌─VISION FEED─┐ │
│ └─────────────┘   (orbiting reticle rings)   └─────────────┘ │
│                                                              │
│ ┌DIAGNOSTICS──┐   ┌────── COMMAND BAR ──────┐                │
│ └─────────────┘   └─────────────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

The bottom-center panel is the **command bar** (input + transcript + waveform).
There is deliberately **no financial/money panel** anywhere in this UI.

---

## 7. Non-negotiables

- Type hints on every Python public function; TS `strict: true`.
- No file over 500 lines — split instead.
- Validate all input at boundaries (Pydantic on backend, zod-free manual guards on FE).
- Never commit secrets. All keys come from `.env` (gitignored).
- Sandbox `exec_python` tool: no network, 5 s timeout, temp cwd, denylist on
  `os.system`, `subprocess`, `socket`, `shutil.rmtree`.
- Webcam frames are processed in-memory and never written to disk.
