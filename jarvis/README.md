<div align="center">

# J.A.R.V.I.S.

### Just A Rather Very Intelligent System

**A voice-driven, webcam-seeing, memory-keeping AI assistant with a cinematic Iron Man HUD.**

`Python 3.11+` · `FastAPI` · `React 18` · `TypeScript` · `three.js` · `OpenRouter (free tier)`

</div>

---

## What this is

A standalone AI assistant that sees through your webcam, hears you, remembers you, and
helps with real-world problems — wrapped in a deep-navy holographic interface built to
look like it came out of a Marvel film rather than a dashboard template.

It runs on **OpenRouter's free tier** (DeepSeek-R1 for planning, Qwen3 and Llama vision
models for everything else). No paid API key required.

> This project is deliberately **self-contained**. It shares no code, data, memory, or
> configuration with any other project in this repository or elsewhere.

---

## Features

### Sees
- **Live webcam vision** — object, face and hand detection through MediaPipe
- **Reads anything you hold up** — OCR via RapidOCR
- **Real-world assistance** — "what am I looking at", "what's wrong with this",
  "how do I fix this" answered from what the camera actually sees, with actionable steps
- **Targeting overlay** — bounding boxes and a scanning sweep rendered over the live feed
- Frames are analyzed in memory and **never written to disk**

### Hears and speaks
- Wake-word activation ("Jarvis") with continuous speech recognition
- Streaming text-to-speech in a British voice
- Live audio waveform driving the HUD's particle core

### Remembers
- Persistent long-term memory in SQLite with vector recall (384-dim embeddings)
- A durable **user profile** — who you are, what you prefer, people and devices you mention
- Recall blends semantic similarity, recency and importance rather than raw cosine alone
- Query expansion with reciprocal-rank fusion for retrieval that survives vague phrasing
- Automatic consolidation dedupes near-identical memories over time

### Thinks and acts
- Multi-step autonomous planning on the free reasoning tier, with a fast path that skips
  planning entirely for ordinary conversation
- Tool-calling loop: file operations, sandboxed code execution, web search, URL fetch,
  weather, news, reminders and scheduling, system control
- Every tool is sandboxed — workspace-confined paths, AST denylists, command allowlists,
  execution timeouts

### Watches itself
- Live CPU / memory / disk / network / battery telemetry driving the HUD gauges
- Rolling diagnostic event log
- `/health` endpoint reporting every subsystem's real status

---

## The interface

A single-screen HUD. No scrolling, no chrome, no menus.

```
┌──────────────────────────────────────────────────────────────┐
│ J.A.R.V.I.S.                                  19:13:32.67    │
│ [ONLINE] [VISION] [MEMORY] [VOICE]            42.36N 71.05W  │
│                                                              │
│ ┌SYSTEM VITALS┐        ╭───────────╮         ┌─RADAR/GLOBE─┐ │
│ │ ▓▓▓▓▓░░ CPU │      ╭─┤  PARTICLE ├─╮       └─────────────┘ │
│ └─────────────┘      │ │   CORE    │ │       ┌─WAVEFORM────┐ │
│ ┌TELEMETRY────┐      ╰─┤  SPHERE   ├─╯       └─────────────┘ │
│ │ ▸ event log │        ╰───────────╯         ┌─VISION FEED─┐ │
│ └─────────────┘   (orbiting reticle rings)   └─────────────┘ │
│                                                              │
│ ┌DIAGNOSTICS──┐   ┌────── COMMAND BAR ──────┐                │
│ └─────────────┘   └─────────────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

The centerpiece is a **4,000-particle three.js sphere** that pulses with your voice and
churns while Jarvis is thinking, ringed by counter-rotating dotted orbital reticles.
Everything is cyan-on-navy wireframe with corner-bracket panel frames, scanlines and glow.

---

## Quick start

### 1. Get a free API key

Sign up at **[openrouter.ai/keys](https://openrouter.ai/keys)** — the models this project
defaults to are free.

### 2. Run it

```bash
cd server
cp .env.example .env   # then set OPENROUTER_API_KEY in .env
npm install && npm run build && npm start
```

Open **http://localhost:8000** — the server serves `jarvis.html` directly, so that one
process is the whole app. Allow camera and microphone access, and say **"Jarvis."**

> `jarvis.html` also opens standalone with no server at all (double-click it, or drag it
> into a browser) — the HUD renders and animates on simulated data, but chat/vision/memory
> won't respond without the server behind it.

> The original Python/FastAPI + React/Vite implementation still lives in `backend/` and
> `frontend/` as a reference — `server/` + `jarvis.html` (Node/TypeScript + one static file)
> is the current, maintained stack.

---

## Deploy it for free

Everything below is optional — running it locally per above is the zero-account path. This
is for a real `https://` URL reachable from any device, at the cost of one thing only you
can do: creating an account on a hosting platform. There's no version of this with zero
action from you — every host ties a running service to *someone's* account, and it has to
be yours, not something done on your behalf.

**Render** (free web-service tier, no card, GitHub sign-in) is a good fit — it runs a real
long-running Node process with WebSockets, unlike pure static/serverless hosts. One
trade-off worth knowing going in: the free tier's disk is ephemeral, so the memory database
(`memory.db`/`tasks.db`) resets on every restart or redeploy — it won't persist the way it
does running on your own machine.

A `render.yaml` blueprint is at the repo root as a starting point, but it's best-effort —
unverified against Render's current schema. The manual path below relies only on stable,
long-standing dashboard concepts and is the more reliable one:

1. Sign up at [render.com](https://render.com) with **"Sign in with GitHub."**
2. **New +** → **Web Service** → connect this repo, branch `claude/spiderman-jarvis-ai-ffdqot`.
3. Set:
   - **Root Directory:** `jarvis/server`
   - **Runtime:** Node
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `npm start`
4. **Environment** tab, add:
   - `OPENROUTER_API_KEY` — your key from step 1 above
   - `JARVIS_HOST` = `0.0.0.0` — required; the server defaults to loopback-only,
     which is unreachable from Render's routing layer
5. **Advanced** → Health Check Path: `/health` (optional, but lets Render detect a bad deploy)
6. Create Web Service. First deploy takes a few minutes; the URL Render assigns is live once
   it's done.

The free tier spins down after ~15 minutes idle and takes 30–60s to wake on the next
request — expected, not a bug.

---

## Architecture

```
frontend/                     React 18 + TypeScript + Vite
├── src/hud/                  HUD components (particle core, reticle, gauges, globe)
├── src/panels/               Command bar, vision feed, chat log
├── src/lib/                  WebSocket client, webcam, audio (STT/TTS/analyser)
└── src/state/                Dependency-free store via useSyncExternalStore

backend/                      FastAPI + WebSockets
└── app/
    ├── api/                  /ws duplex channel + REST routes
    ├── models/schemas.py     Shared contracts (single source of truth)
    └── core/
        ├── agent/            Agent loop, planner, persona, classifier
        ├── llm/              OpenRouter provider + free-tier fallback router
        ├── memory/           SQLite store, embeddings, profile, recall engine
        ├── tools/            Files, sandbox, scheduler, web, system
        ├── vision/           Pipeline, detection, OCR, scene understanding
        ├── voice/            Optional server-side TTS/STT
        └── telemetry/        psutil system monitor
```

Frontend and backend talk over **one duplex WebSocket** carrying typed, discriminated
events — token streams, tool calls, telemetry pushes, vision results and speech cues.
The full protocol is specified in [`docs/CONTRACTS.md`](docs/CONTRACTS.md).

### Model routing

| Purpose | Chain (free tier, in fallback order) |
|---------|--------------------------------------|
| Planning | `deepseek-r1` → `qwen3-235b` → `llama-3.3-70b` |
| Chat | fast free chat models |
| Vision | `llama-3.2-11b-vision` → `qwen2.5-vl-72b` |

Rate limits and provider outages fall through the chain transparently with backoff.

---

## Graceful degradation

Every heavy dependency is optional and lazily imported. The app starts and runs with none
of them installed:

| Missing | What degrades |
|---------|---------------|
| `mediapipe` | Falls back to OpenCV Haar cascades for faces; no hand tracking |
| `rapidocr-onnxruntime` | No OCR; scene understanding still works |
| `sentence-transformers` | Hash-based fallback embeddings; recall gets fuzzier |
| `pyttsx3` / `faster-whisper` | Browser handles TTS/STT instead (the default anyway) |

---

## Security

- Webcam frames processed in memory, never persisted
- File tools confined to a workspace root; traversal and symlink escapes rejected
- `exec_python` runs in a subprocess with an AST denylist, 5s timeout and a temp cwd
- `exec_shell` restricted to a read-only command allowlist
- API keys read from `.env` only — never committed, never logged

---

## Built with Ruflo

Planned and implemented by a ten-agent Ruflo swarm in five specialized groups —
coding, files, memory, frontend, backend — coordinated through a shared contract
document with strict per-agent file ownership so no two agents ever wrote the same file.

---

<div align="center">
<sub>Standalone project. No connection to trading, research, or any other work in this repository.</sub>
</div>
