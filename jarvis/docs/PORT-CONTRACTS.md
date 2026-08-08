# J.A.R.V.I.S. — TypeScript Port Contracts

**Every agent MUST read this file and `server/src/types.ts` before writing code.**

We are porting the Python backend to **Node.js + TypeScript**, and collapsing the
React/Vite frontend into **one self-contained HTML file**.

The Python backend under `backend/` is the reference implementation. Read it for
behavior, then write idiomatic TypeScript — do not transliterate Python.

---

## 1. Stack

| Layer | Choice |
|-------|--------|
| Runtime | Node.js 22+ (ESM, `"type": "module"`) |
| Language | TypeScript 5, `strict: true` |
| HTTP/WS | `express` + `ws` |
| DB | `better-sqlite3` (synchronous, fast, no native async needed) |
| LLM | OpenRouter free tier — unchanged model chains |
| Telemetry | `systeminformation` |
| Frontend | ONE file: `jarvis.html`. No build step, no npm, no CDN. |

**No three.js.** The particle core is hand-rolled Canvas 2D with real 3D
projection. A 460KB library cannot be inlined sensibly, and "self-contained"
must mean zero external requests.

---

## 2. Directory ownership (DO NOT WRITE OUTSIDE YOUR OWN)

| Group | Agent | Owns exactly |
|-------|-------|--------------|
| CODING | `ts-core` | `server/src/llm/`, `server/src/agent/` |
| MEMORY | `ts-memory` | `server/src/memory/` |
| FILES | `ts-tools` | `server/src/tools/` |
| BACKEND | `ts-server` | `server/src/index.ts`, `server/src/ws.ts`, `server/src/config.ts`, `server/src/vision/`, `server/src/telemetry/`, `server/package.json`, `server/tsconfig.json` |
| FRONTEND | `html-ui` | `jarvis.html` (the entire single file) |

`server/src/types.ts` is the shared contract — **read-only for everyone**.

---

## 3. WebSocket protocol — UNCHANGED

Identical to the Python edition so the client stays compatible. See
`server/src/types.ts` (`ClientMessage`, `ServerEvent`) for the exact discriminated
unions. Telemetry is pushed unprompted every 1000 ms. Vision events fire only
after a `frame` message.

---

## 4. Model chains (free tier, unchanged)

```ts
planning: ['deepseek/deepseek-r1:free', 'qwen/qwen3-235b-a22b:free',
           'meta-llama/llama-3.3-70b-instruct:free']
chat:     ['meta-llama/llama-3.3-70b-instruct:free', 'qwen/qwen3-235b-a22b:free']
vision:   ['meta-llama/llama-3.2-11b-vision-instruct:free',
           'qwen/qwen2.5-vl-72b-instruct:free']
```

No Gemini, no Google models anywhere. Fall through the chain on 429/5xx with
exponential backoff plus jitter, max 3 attempts per model.

---

## 5. Vision strategy (changed from Python)

Node has no practical MediaPipe. Split the work:

- **Browser** does cheap geometry: face detection via the native `FaceDetector`
  API where available, degrading silently to none.
- **Server** does semantics: the vision model chain describes the scene, reads
  any visible text, and returns actionable next steps.

`VisionResult.faces`/`hands` may be 0 when the browser cannot detect — that is
expected, not an error. Keep the frame-similarity short-circuit from the Python
version so a static scene does not burn free-tier calls.

---

## 6. `jarvis.html` requirements

One file. Opening it directly must render the full HUD with simulated data;
connecting to the server makes it live. Everything inlined — CSS, JS, fonts
(system font stack, no webfont fetches).

### Design tokens (authoritative — do not invent colors)

```css
--j-bg:#030a1a; --j-bg-2:#05122b; --j-cyan:#4fd8ff; --j-cyan-dim:#1c6f92;
--j-blue:#1b8fe0; --j-glow:#7ee8ff; --j-amber:#ffb545; --j-red:#ff4d5e;
--j-grid:rgba(79,216,255,0.10); --j-text:#b8ecff; --j-text-dim:#4d7f9c;
```

All HUD text uppercase, letter-spacing `0.12em`, monospace. Every panel framed
with corner brackets, never a plain border.

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│ [brand + subtitle]                            [clock/date]   │
│ [status chips]                                [geo/coords]   │
│ ┌SYSTEM VITALS┐        ╭───────────╮         ┌─RADAR/GLOBE─┐ │
│ │ bars        │      ╭─┤  PARTICLE ├─╮       └─────────────┘ │
│ └─────────────┘      │ │   CORE    │ │       ┌─WAVEFORM────┐ │
│ ┌TELEMETRY────┐      ╰─┤  SPHERE   ├─╯       └─────────────┘ │
│ │ scroll list │        ╰───────────╯         ┌─VISION FEED─┐ │
│ └─────────────┘   (orbiting reticle rings)   └─────────────┘ │
│ ┌DIAGNOSTICS──┐   ┌────── COMMAND BAR ──────┐                │
│ └─────────────┘   └─────────────────────────┘                │
└──────────────────────────────────────────────────────────────┘
```

The bottom-center panel is the **command bar**. There is deliberately **no
financial/money panel anywhere** — this is an explicit user requirement.

### Required client features

Webcam capture + throttled frame send · mic level metering for the waveform ·
`SpeechRecognition` with "jarvis" wake word · `SpeechSynthesis` TTS preferring a
deep en-GB voice · auto-reconnecting WebSocket with backoff · graceful
DISCONNECTED state. Every browser API guarded — missing APIs no-op, never throw.

---

## 7. Non-negotiables

- `strict: true`; no `any` in exported signatures.
- No file over 500 lines (`jarvis.html` is exempt but must stay organized with
  clear section banners).
- Validate all input at boundaries.
- Never commit secrets; key comes from `.env` via `OPENROUTER_API_KEY`.
- Sandbox tool: no network, 5s timeout, temp cwd, denylist on `child_process`,
  `fs` writes outside temp, `process.exit`.
- Webcam frames processed in memory, never written to disk.
- Every optional dependency lazily imported in try/catch and degrading with a
  logged warning — the server must start with none of them installed.
