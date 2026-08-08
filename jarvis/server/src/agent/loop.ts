/**
 * The agent loop -- JARVIS's turn-handling brain.
 *
 * One turn runs: recall memory -> classify simple vs multi-step -> plan when
 * multi-step -> optionally look through the camera -> then a bounded
 * tool-calling loop that streams tokens out as they arrive. Everything is
 * emitted as `ServerEvent` values (types.ts) via an async generator, which
 * the WebSocket layer forwards verbatim.
 *
 * Failure policy: the loop degrades, it does not hard-fail. Memory, the
 * planner, vision, tools and even the streaming provider may each break
 * independently; the user still gets a reply and a `done` event.
 */

import type {
  LLMDelta,
  Message,
  MemoryStore,
  Plan,
  ServerEvent,
  ToolResult,
  ToolSpec,
  TurnContext,
  VisionResult,
} from '../types.js';
import type { ModelRouter } from '../llm/router.js';
import type { Classification } from './classifier.js';
import { TurnClassifier, needsVision } from './classifier.js';
import { buildSystemPrompt, degradedReply } from './persona.js';
import { Planner } from './planner.js';

export const MAX_ITERATIONS = 8;
const TOOL_TIMEOUT_MS = 30_000;
const RECALL_TIMEOUT_MS = 6_000;
const VISION_TIMEOUT_MS = 45_000;
const PERSIST_TIMEOUT_MS = 5_000;
const REPEAT_LIMIT = 3;
const SUMMARY_CHARS = 240;
const TOOL_OUTPUT_CHARS = 8_000;

// ---------------------------------------------------------------------------
// Tool registry discovery
// ---------------------------------------------------------------------------

/**
 * The tool-registry contract owned by `../tools/index.js` (a sibling agent's
 * package). Declared locally -- rather than statically imported -- so a
 * registry that hasn't landed yet never blocks this package's typecheck.
 */
export interface ToolRegistryLike {
  all(): readonly ToolSpec[];
  runTool(name: string, args: Record<string, unknown>): Promise<ToolResult>;
}

function isToolRegistry(value: unknown): value is ToolRegistryLike {
  const candidate = value as Partial<ToolRegistryLike> | null | undefined;
  return typeof candidate?.all === 'function' && typeof candidate?.runTool === 'function';
}

// Routed through a non-literal specifier so TypeScript treats the import as
// `Promise<any>` and skips static module resolution -- the tools package is
// owned by another agent and may not exist yet.
const TOOLS_MODULE_PATH = '../tools/index.js';

async function loadDefaultToolRegistry(): Promise<ToolRegistryLike | null> {
  try {
    const mod = (await import(TOOLS_MODULE_PATH)) as { registry?: unknown; default?: unknown };
    const candidate = mod.registry ?? mod.default;
    return isToolRegistry(candidate) ? candidate : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Turn-scoped helpers
// ---------------------------------------------------------------------------

interface PendingCall {
  name: string;
  args: Record<string, unknown>;
  id: string;
}

function newPendingCall(name: string, args: Record<string, unknown>): PendingCall {
  return { name, args, id: cryptoRandomId() };
}

function cryptoRandomId(): string {
  return globalThis.crypto.randomUUID().replace(/-/g, '').slice(0, 12);
}

function callSignature(call: PendingCall): string {
  try {
    return `${call.name}:${JSON.stringify(call.args, Object.keys(call.args).sort())}`;
  } catch {
    return `${call.name}:${String(call.args)}`;
  }
}

interface StreamOutcome {
  text: string;
  calls: PendingCall[];
  error: string | null;
  cancelled: boolean;
}

function clip(text: string, limit: number): string {
  const value = text ?? '';
  return value.length <= limit ? value : `${value.slice(0, limit).trimEnd()}…`;
}

export interface AgentLoopOptions {
  memory?: MemoryStore | null;
  tools?: ToolRegistryLike | null;
  planner?: Planner | null;
  classifier?: TurnClassifier | null;
  maxIterations?: number;
  recallK?: number;
  persistMemory?: boolean;
  emitSpeak?: boolean;
  toolTimeoutMs?: number;
}

/**
 * Handles one user turn end to end, yielding protocol events as it goes.
 *
 * @param router - The `ModelRouter` used for chat streaming, planning and
 *   vision, all with free-tier fallback baked in.
 */
export class AgentLoop {
  private readonly router: ModelRouter;
  private readonly memory: MemoryStore | null;
  private readonly toolsOverride: ToolRegistryLike | null;
  private readonly planner: Planner;
  private readonly classifier: TurnClassifier;
  private readonly maxIterations: number;
  private readonly recallK: number;
  private readonly persistMemory: boolean;
  private readonly emitSpeak: boolean;
  private readonly toolTimeoutMs: number;

  constructor(router: ModelRouter, opts: AgentLoopOptions = {}) {
    this.router = router;
    this.memory = opts.memory ?? null;
    this.toolsOverride = opts.tools ?? null;
    this.planner = opts.planner ?? new Planner(router);
    this.classifier = opts.classifier ?? new TurnClassifier(router);
    this.maxIterations = Math.max(1, opts.maxIterations ?? MAX_ITERATIONS);
    this.recallK = Math.max(0, opts.recallK ?? 6);
    this.persistMemory = opts.persistMemory ?? true;
    this.emitSpeak = opts.emitSpeak ?? false;
    this.toolTimeoutMs = opts.toolTimeoutMs ?? TOOL_TIMEOUT_MS;
  }

  /** Run one turn, yielding `ServerEvent`s until `done`. Honors `signal`. */
  async *handle(userText: string, ctx: TurnContext, signal?: AbortSignal): AsyncGenerator<ServerEvent> {
    const turn = ctx.turnId;
    const text = (userText ?? '').trim();
    if (!text) {
      yield { type: 'done', id: turn, text: '' };
      return;
    }

    // 1. Recall ------------------------------------------------------------
    yield { type: 'thinking', id: turn, stage: 'recall' };
    await this.recall(text, ctx);
    if (signal?.aborted) {
      yield this.cancelDone(turn);
      return;
    }

    const { tools, specs } = await this.resolveTools();

    // 2. Classify + 3. Plan --------------------------------------------------
    const verdict = await this.classify(text, ctx);
    let plan: Plan | null = null;
    if (verdict.multiStep) {
      yield { type: 'thinking', id: turn, stage: 'planning' };
      plan = await this.planTurn(text, ctx, specs);
    }
    if (signal?.aborted) {
      yield this.cancelDone(turn);
      return;
    }

    // 4. Vision --------------------------------------------------------------
    if (this.shouldLook(text, ctx)) {
      yield { type: 'thinking', id: turn, stage: 'vision' };
      const described = await this.describeFrame(text, ctx, signal);
      if (described) yield { type: 'vision', scene: described };
    }

    // 5. Tool-calling / streaming loop ----------------------------------------
    const messages = this.buildMessages(text, ctx, specs, plan);
    let reply = '';
    const seen = new Map<string, number>();

    for (let iteration = 0; iteration < this.maxIterations; iteration += 1) {
      if (signal?.aborted) {
        yield this.cancelDone(turn, reply);
        return;
      }

      const outcome: StreamOutcome = { text: '', calls: [], error: null, cancelled: false };
      const isLastIteration = iteration === this.maxIterations - 1;
      for await (const event of this.streamPass(messages, isLastIteration ? undefined : specs, turn, signal, outcome)) {
        yield event;
      }

      if (outcome.text) reply = outcome.text;
      if (outcome.cancelled) {
        yield this.cancelDone(turn, reply);
        return;
      }
      if (outcome.error && outcome.calls.length === 0) {
        reply = await this.salvage(messages, outcome.error, reply, signal);
        break;
      }
      if (outcome.calls.length === 0) break;

      messages.push({ role: 'assistant', content: outcome.text || '', name: 'jarvis' });
      for (const call of outcome.calls) {
        if (signal?.aborted) {
          yield this.cancelDone(turn, reply);
          return;
        }
        yield { type: 'tool_call', id: turn, name: call.name, args: { ...call.args } };
        const sig = callSignature(call);
        const count = (seen.get(sig) ?? 0) + 1;
        seen.set(sig, count);

        const result =
          count > REPEAT_LIMIT
            ? {
                ok: false,
                output: `Refused: \`${call.name}\` has already been called with these exact arguments. Use the result you already have, or change approach.`,
                summary: 'repeated call suppressed',
              }
            : await this.invoke(tools, call);

        yield {
          type: 'tool_result',
          id: turn,
          name: call.name,
          ok: result.ok,
          summary: result.summary || clip(result.output, SUMMARY_CHARS),
        };
        messages.push({
          role: 'tool',
          content: clip(result.output, TOOL_OUTPUT_CHARS),
          name: call.name,
          toolCallId: call.id,
        });
      }
    }

    if (!reply.trim()) reply = degradedReply();

    await this.persist(text, reply, ctx);

    if (this.emitSpeak) yield { type: 'speak', text: reply, voice: 'jarvis' };
    yield { type: 'done', id: turn, text: reply };
  }

  // -- stage helpers ---------------------------------------------------------

  private cancelDone(turn: string, partial = ''): ServerEvent {
    return { type: 'done', id: turn, text: partial };
  }

  private async recall(text: string, ctx: TurnContext): Promise<void> {
    if (!this.memory || this.recallK === 0) return;
    try {
      ctx.recalled = (await withTimeout(this.memory.recall(text, this.recallK), RECALL_TIMEOUT_MS)) ?? [];
    } catch {
      // best-effort -- recall failure never blocks the turn
    }
    if (ctx.history.length > 0) return;
    try {
      ctx.history = (await withTimeout(this.memory.history(20), RECALL_TIMEOUT_MS)) ?? [];
    } catch {
      // best-effort
    }
  }

  private async classify(text: string, ctx: TurnContext): Promise<Classification> {
    try {
      return await this.classifier.classify(text, ctx);
    } catch (exc) {
      return { multiStep: false, score: 0, reason: `classifier error: ${(exc as Error).message}`, source: 'default' };
    }
  }

  private async planTurn(text: string, ctx: TurnContext, specs: readonly ToolSpec[]): Promise<Plan | null> {
    try {
      const plan = await this.planner.plan(text, ctx, specs);
      return plan.steps.length > 0 ? plan : null;
    } catch {
      return null;
    }
  }

  private shouldLook(text: string, ctx: TurnContext): boolean {
    if (!ctx.lastFrameB64) return false;
    if (ctx.vision?.scene?.trim()) return false;
    return needsVision(text);
  }

  private async describeFrame(text: string, ctx: TurnContext, signal: AbortSignal | undefined): Promise<string> {
    const prompt: Message[] = [
      {
        role: 'system',
        content:
          'Describe this webcam frame factually and concisely for another assistant to ' +
          'reason over: people, objects, activity, legible text, setting. No speculation, no commentary.',
      },
      { role: 'user', content: text.split(/\s+/).join(' ').slice(0, 500) },
    ];
    try {
      const routed = await withTimeout(
        this.router.vision(prompt, ctx.lastFrameB64 ?? '', signal),
        VISION_TIMEOUT_MS,
      );
      const scene = (routed.text ?? '').trim();
      if (!scene) return '';
      ctx.vision = ctx.vision ? { ...ctx.vision, scene } : emptyVisionResult(scene);
      return scene;
    } catch {
      return '';
    }
  }

  private buildMessages(text: string, ctx: TurnContext, specs: readonly ToolSpec[], plan: Plan | null): Message[] {
    const system = buildSystemPrompt(ctx, { tools: specs, plan });
    const messages: Message[] = [{ role: 'system', content: system }];
    messages.push(...ctx.history.filter((m) => m.role !== 'system'));

    const last = messages[messages.length - 1];
    const already = messages.length > 1 && last?.role === 'user' && last.content.trim() === text;
    if (!already) {
      messages.push({ role: 'user', content: text, images: ctx.lastFrameB64 ? [ctx.lastFrameB64] : [] });
    }
    return messages;
  }

  // -- streaming ---------------------------------------------------------------

  private async *streamPass(
    messages: Message[],
    specs: readonly ToolSpec[] | undefined,
    turn: string,
    signal: AbortSignal | undefined,
    outcome: StreamOutcome,
  ): AsyncGenerator<ServerEvent> {
    let pending: PendingCall | null = null;
    const chunks: string[] = [];

    try {
      const stream = this.router.streamChat(messages, specs ? [...specs] : undefined, signal);
      for await (const delta of stream) {
        if (signal?.aborted) {
          outcome.cancelled = true;
          break;
        }
        pending = this.absorbDelta(delta, pending, outcome);
        if (delta.text) {
          chunks.push(delta.text);
          yield { type: 'token', id: turn, text: delta.text };
        }
        if (delta.finished) break;
      }
    } catch (exc) {
      outcome.error = (exc as Error).message || (exc as Error).name || 'stream failed';
      yield { type: 'log', level: 'warn', text: `stream error: ${outcome.error}` };
    }

    if (pending) outcome.calls.push(pending);
    outcome.text = chunks.join('').trim();
  }

  private absorbDelta(delta: LLMDelta, pending: PendingCall | null, outcome: StreamOutcome): PendingCall | null {
    if (delta.toolName) {
      const args = delta.toolArgs ?? {};
      const mergeable =
        pending && pending.name === delta.toolName && !Object.keys(args).some((k) => k in pending!.args);
      if (mergeable && pending) {
        Object.assign(pending.args, args);
        return pending;
      }
      if (pending) outcome.calls.push(pending);
      return newPendingCall(delta.toolName, { ...args });
    }
    if (delta.toolArgs && pending) {
      Object.assign(pending.args, delta.toolArgs);
    }
    return pending;
  }

  private async salvage(messages: Message[], error: string, partial: string, signal: AbortSignal | undefined): Promise<string> {
    if (partial.trim()) return partial;
    try {
      const routed = await withTimeout(this.router.completeChat(messages, signal), VISION_TIMEOUT_MS);
      return (routed.text ?? '').trim() || degradedReply(error);
    } catch {
      return degradedReply(error);
    }
  }

  // -- tools -------------------------------------------------------------------

  private async resolveTools(): Promise<{ tools: ToolRegistryLike | null; specs: ToolSpec[] }> {
    const registry = this.toolsOverride ?? (await loadDefaultToolRegistry());
    if (!registry) return { tools: null, specs: [] };
    try {
      return { tools: registry, specs: [...registry.all()] };
    } catch {
      return { tools: null, specs: [] };
    }
  }

  private async invoke(tools: ToolRegistryLike | null, call: PendingCall): Promise<ToolResult> {
    if (!tools) {
      return { ok: false, output: `No tool registry available to run \`${call.name}\`.`, summary: 'no registry' };
    }
    try {
      return await withTimeout(tools.runTool(call.name, call.args), this.toolTimeoutMs);
    } catch (exc) {
      const message = (exc as Error).message || String(exc);
      if (message.includes('timed out')) {
        return { ok: false, output: `\`${call.name}\` timed out after ${(this.toolTimeoutMs / 1000).toFixed(0)}s.`, summary: 'timed out' };
      }
      return { ok: false, output: `\`${call.name}\` failed: ${message}`, summary: (exc as Error).name ?? 'error' };
    }
  }

  // -- persistence ---------------------------------------------------------------

  private async persist(userText: string, reply: string, ctx: TurnContext): Promise<void> {
    if (!this.persistMemory || !this.memory) return;
    try {
      await withTimeout(
        Promise.all([
          this.memory.appendTurn('user', userText, ctx.turnId),
          this.memory.appendTurn('assistant', reply, ctx.turnId),
        ]),
        PERSIST_TIMEOUT_MS,
      );
    } catch {
      // best-effort -- a failed persist never fails the turn
    }
  }
}

function emptyVisionResult(scene: string): VisionResult {
  return { objects: [], faces: 0, hands: 0, text: '', scene, width: 0, height: 0 };
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`timed out after ${ms}ms`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err: unknown) => {
        clearTimeout(timer);
        reject(err as Error);
      },
    );
  });
}
