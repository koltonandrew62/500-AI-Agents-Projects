/**
 * J.A.R.V.I.S. server entrypoint: express app + http server + `/ws` upgrade.
 *
 * Boots every subsystem, wires them together, serves the single-file
 * `jarvis.html` client at `/`, starts the telemetry broadcast + reminder
 * loops, and shuts down cleanly on SIGINT/SIGTERM.
 *
 * Integration contract with sibling modules (built in parallel by other
 * agents, per PORT-CONTRACTS.md §2):
 *   - `./llm/index.js`    exports `ModelRouter` implementing `LLMProvider` (types.ts).
 *   - `./memory/index.js` exports `SQLiteMemoryStore` implementing `MemoryStore`.
 *   - `./tools/index.js`  exports `registry` (`{ all(): ToolSpec[] }`) and `runTool`.
 *   - `./agent/index.js`  exports `AgentLoop`, whose `handle(text, ctx, signal)`
 *     is an async generator of `ServerEvent` — the same stream `ws.ts` relays
 *     live and this file's `POST /api/chat` drains non-streaming.
 *
 * This module owns none of those directories; it only composes them.
 */

import { existsSync } from 'node:fs';
import { createServer } from 'node:http';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import express, { type NextFunction, type Request, type Response } from 'express';

import { config } from './config.js';
import { attachWebSocket, type AgentLoopLike } from './ws.js';
import { VisionPipeline } from './vision/index.js';
import { TelemetryMonitor } from './telemetry/index.js';
import type { Message, ServerEvent, TurnContext } from './types.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Subsystem construction — every sibling import is soft: a missing/broken
// module degrades that one subsystem (and /health reports it) rather than
// preventing the process from starting at all.
// ---------------------------------------------------------------------------

interface LLMProviderLike {
  stream(messages: Message[], tools?: unknown, signal?: AbortSignal): AsyncIterable<unknown>;
  complete(messages: Message[], signal?: AbortSignal): Promise<string>;
  vision(messages: Message[], imageB64: string, signal?: AbortSignal): Promise<string>;
}

interface MemoryStoreLike {
  history(limit?: number): Promise<Message[]>;
  recall(query: string, k?: number): Promise<unknown[]>;
}

interface ToolRegistryLike {
  all(): unknown[];
}

async function loadLLMProvider(): Promise<LLMProviderLike | null> {
  try {
    const mod = await import('./llm/index.js');
    const Ctor = (mod as Record<string, unknown>).ModelRouter as
      | (new (opts: { apiKey?: string; chains?: typeof config.models }) => LLMProviderLike)
      | undefined;
    if (!Ctor) throw new Error("llm/index.js did not export 'ModelRouter'");
    return new Ctor({ apiKey: config.openrouterApiKey || undefined, chains: config.models });
  } catch (err) {
    console.warn('[index] llm subsystem unavailable at startup:', errMsg(err));
    return null;
  }
}

async function loadMemoryStore(): Promise<MemoryStoreLike | null> {
  try {
    const mod = await import('./memory/index.js');
    const Ctor = (mod as Record<string, unknown>).SqliteMemoryStore as
      | (new (dbPath: string) => MemoryStoreLike)
      | undefined;
    if (!Ctor) throw new Error("memory/index.js did not export 'SqliteMemoryStore'");
    return new Ctor(config.memoryDbPath);
  } catch (err) {
    console.warn('[index] memory subsystem unavailable at startup:', errMsg(err));
    return null;
  }
}

async function loadToolRegistry(): Promise<ToolRegistryLike | null> {
  try {
    const mod = await import('./tools/index.js');
    const registry = (mod as Record<string, unknown>).registry as ToolRegistryLike | undefined;
    if (!registry) throw new Error("tools/index.js did not export 'registry'");
    return registry;
  } catch (err) {
    console.warn('[index] tools subsystem unavailable at startup:', errMsg(err));
    return null;
  }
}

async function loadAgentLoop(
  llm: LLMProviderLike | null,
  memory: MemoryStoreLike | null,
  tools: ToolRegistryLike | null,
): Promise<AgentLoopLike | null> {
  if (!llm) return null;
  try {
    const mod = await import('./agent/index.js');
    const Ctor = (mod as Record<string, unknown>).AgentLoop as
      | (new (
          router: LLMProviderLike,
          opts: { memory?: MemoryStoreLike | null; tools?: ToolRegistryLike | null; maxIterations?: number; emitSpeak?: boolean },
        ) => AgentLoopLike)
      | undefined;
    if (!Ctor) throw new Error("agent/index.js did not export 'AgentLoop'");
    return new Ctor(llm, {
      memory,
      tools,
      maxIterations: config.maxToolIterations,
      emitSpeak: config.voiceEnabled,
    });
  } catch (err) {
    console.warn('[index] agent subsystem unavailable at startup:', errMsg(err));
    return null;
  }
}

/** Answers every `chat`/`voice` turn with a clear, typed error — used when the agent loop never came up. */
function unavailableAgentLoop(): AgentLoopLike {
  return {
    async *handle(_text: string, ctx: TurnContext): AsyncIterable<ServerEvent> {
      yield { type: 'error', id: ctx.turnId, text: 'agent subsystem is unavailable (missing OPENROUTER_API_KEY or a sibling module failed to load)' };
      yield { type: 'done', id: ctx.turnId, text: '' };
    },
  };
}

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// ---------------------------------------------------------------------------
// jarvis.html resolution — one process serves everything.
// ---------------------------------------------------------------------------

function resolveJarvisHtml(): string | null {
  const candidates = [
    resolve(process.cwd(), '../jarvis.html'),
    resolve(process.cwd(), 'jarvis.html'),
    resolve(__dirname, '../../jarvis.html'),
    resolve(__dirname, '../../../jarvis.html'),
  ];
  return candidates.find((p) => existsSync(p)) ?? null;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  config.ensureDirs();

  const [llmProvider, memoryStore, toolRegistry] = await Promise.all([
    loadLLMProvider(),
    loadMemoryStore(),
    loadToolRegistry(),
  ]);
  const agentLoop = (await loadAgentLoop(llmProvider, memoryStore, toolRegistry)) ?? unavailableAgentLoop();

  const telemetryMonitor = new TelemetryMonitor();

  let visionPipeline: VisionPipeline | null = null;
  if (config.visionEnabled) {
    if (llmProvider) {
      visionPipeline = new VisionPipeline({
        visionComplete: (messages, imageB64, signal) => llmProvider.vision(messages, imageB64, signal),
      });
    } else {
      console.warn('[index] vision enabled but no LLM provider is available; vision subsystem disabled');
    }
  }

  const app = express();
  app.use(express.json({ limit: '2mb' }));

  app.get('/health', async (_req: Request, res: Response) => {
    res.json({
      llm: llmHealth(),
      db: await dbHealth(memoryStore),
      vision: subsystemHealth(visionPipeline, config.visionEnabled),
      voice: { status: config.voiceEnabled ? 'ok' : 'disabled', detail: 'voice runs client-side (browser SpeechRecognition/SpeechSynthesis)' },
    });
  });

  app.get('/api/tools', (_req: Request, res: Response) => {
    res.json(toolRegistry ? toolRegistry.all() : []);
  });

  app.get('/api/telemetry', async (_req: Request, res: Response) => {
    try {
      res.json(await telemetryMonitor.sample());
    } catch (err) {
      res.status(500).json({ detail: errMsg(err) });
    }
  });

  app.post('/api/chat', async (req: Request, res: Response) => {
    const text = typeof req.body?.text === 'string' ? req.body.text.trim() : '';
    if (!text || text.length > 8000) {
      res.status(400).json({ detail: 'text must be a non-empty string up to 8000 chars' });
      return;
    }
    const turnId = typeof req.body?.id === 'string' && req.body.id ? req.body.id : crypto.randomUUID();
    const ctx: TurnContext = { turnId, history: [], recalled: [], telemetry: null, vision: null, lastFrameB64: null };
    const events: ServerEvent[] = [];
    let reply = '';
    try {
      for await (const event of agentLoop.handle(text, ctx, new AbortController().signal)) {
        events.push(event);
        if (event.type === 'done') reply = event.text;
      }
    } catch (err) {
      res.status(500).json({ detail: errMsg(err) });
      return;
    }
    res.json({ id: turnId, text: reply, events });
  });

  app.get('/api/memory/search', async (req: Request, res: Response) => {
    const q = typeof req.query.q === 'string' ? req.query.q : '';
    const k = Math.min(50, Math.max(1, Number.parseInt(String(req.query.k ?? '6'), 10) || 6));
    if (!q) {
      res.status(400).json({ detail: "query parameter 'q' is required" });
      return;
    }
    if (!memoryStore) {
      res.status(503).json({ detail: 'memory subsystem unavailable' });
      return;
    }
    try {
      res.json(await memoryStore.recall(q, k));
    } catch (err) {
      res.status(500).json({ detail: errMsg(err) });
    }
  });

  const jarvisHtmlPath = resolveJarvisHtml();
  if (jarvisHtmlPath) {
    console.log(`[index] serving frontend from ${jarvisHtmlPath}`);
    app.get('/', (_req: Request, res: Response) => res.sendFile(jarvisHtmlPath));
  } else {
    console.warn('[index] jarvis.html not found; GET / will 404 until it is built');
  }

  // Boundary safety net: an internal error becomes typed JSON, never a crash.
  app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
    console.error('[index] unhandled request error', err);
    if (!res.headersSent) res.status(500).json({ detail: errMsg(err) });
  });

  const httpServer = createServer(app);
  const ws = attachWebSocket({
    httpServer,
    agentLoop,
    visionPipeline,
    telemetryMonitor,
    telemetryIntervalMs: config.telemetryIntervalMs,
  });

  await new Promise<void>((resolve) => httpServer.listen(config.port, config.host, resolve));
  console.log(
    `[index] jarvis backend ready on http://${config.host}:${config.port} ` +
      `(llm=${!!llmProvider}, memory=${!!memoryStore}, tools=${!!toolRegistry}, vision=${!!visionPipeline}, voice=${config.voiceEnabled})`,
  );

  let shuttingDown = false;
  const shutdown = (signal: string): void => {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`[index] received ${signal}, shutting down`);
    ws.close();
    httpServer.close(() => {
      console.log('[index] shutdown complete');
      process.exit(0);
    });
    // Force-exit if close() hangs (e.g. a stuck keep-alive socket).
    setTimeout(() => process.exit(0), 5000).unref();
  };
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
}

function llmHealth(): { status: string; detail: string } {
  const configured = Boolean(config.openrouterApiKey);
  return { status: configured ? 'ok' : 'unconfigured', detail: configured ? '' : 'OPENROUTER_API_KEY not set' };
}

async function dbHealth(memoryStore: MemoryStoreLike | null): Promise<{ status: string; detail: string }> {
  if (!memoryStore) return { status: 'error', detail: 'memory store not initialized' };
  try {
    await memoryStore.history(1);
    return { status: 'ok', detail: '' };
  } catch (err) {
    return { status: 'error', detail: errMsg(err) };
  }
}

function subsystemHealth(instance: unknown, enabled: boolean): { status: string; detail: string } {
  if (!enabled) return { status: 'disabled', detail: '' };
  if (instance === null || instance === undefined) return { status: 'unavailable', detail: 'module not loaded' };
  return { status: 'ok', detail: '' };
}

main().catch((err: unknown) => {
  console.error('[index] fatal startup error', err);
  process.exit(1);
});
