/**
 * Multi-step planner running on the OpenRouter free-tier planning chain.
 *
 * Free-tier models are chatty: they wrap JSON in prose, in markdown fences,
 * and (DeepSeek-R1) in `<think>` blocks. This module treats model output as
 * hostile text to be mined, not as JSON to be parsed, and it never throws
 * into the caller: an unparseable plan degrades to a single-step plan so the
 * turn continues.
 */

import type { Message, Plan, Step, ToolSpec, TurnContext } from '../types.js';
import type { ModelRouter } from '../llm/router.js';

const MAX_STEPS = 8;
const DEFAULT_TIMEOUT_MS = 45_000;

const FENCE_RE = /```[a-zA-Z0-9_+-]*\s*|\s*```/g;
const THINK_RE = /<(think|thinking|reasoning)>[\s\S]*?<\/\1>/gi;
const OPEN_THINK_RE = /<(think|thinking|reasoning)>[\s\S]*$/gi;
const TRAILING_COMMA_RE = /,(\s*[}\]])/g;

export const PLANNER_SYSTEM_PROMPT = `You are the planning subsystem of a local AI assistant. You do not talk to the user.
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
            "args": {}, "rationale": "<one short clause>"}]}`;

// ---------------------------------------------------------------------------
// JSON salvage
// ---------------------------------------------------------------------------

/** Remove `<think>` blocks and markdown fences from raw model output. */
export function stripReasoning(text: string): string {
  let cleaned = (text ?? '').replace(THINK_RE, ' ');
  cleaned = cleaned.replace(OPEN_THINK_RE, ' ');
  cleaned = cleaned.replace(FENCE_RE, '\n');
  return cleaned.trim();
}

/** Return the balanced `opener`/`closer` span beginning at `start`. */
function scanBalanced(text: string, start: number, opener: string, closer: string): string | null {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === opener) depth += 1;
    else if (ch === closer) {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return null;
}

/** All balanced JSON object/array spans in `text`, longest first. */
function candidates(text: string): string[] {
  const found: string[] = [];
  for (const [opener, closer] of [
    ['{', '}'],
    ['[', ']'],
  ] as const) {
    for (let idx = 0; idx < text.length; idx += 1) {
      if (text[idx] !== opener) continue;
      const span = scanBalanced(text, idx, opener, closer);
      if (span && span.length > 2) found.push(span);
    }
  }
  found.sort((a, b) => b.length - a.length);
  return found;
}

/** Parse JSON, repairing the two failures free-tier models actually make. */
function loadsRepaired(blob: string): unknown {
  try {
    return JSON.parse(blob);
  } catch {
    // fall through to repair attempts
  }
  const repaired = blob.replace(TRAILING_COMMA_RE, '$1');
  try {
    return JSON.parse(repaired);
  } catch {
    // fall through
  }
  // Python-literal-ish output: True/False/None and single quotes.
  const swapped = repaired
    .replace(/'/g, '"')
    .replace(/\bTrue\b/g, 'true')
    .replace(/\bFalse\b/g, 'false')
    .replace(/\bNone\b/g, 'null');
  return JSON.parse(swapped);
}

/** Mine the first plausible JSON plan object out of arbitrary model prose. */
export function extractJsonObject(text: string): Record<string, unknown> | null {
  const cleaned = stripReasoning(text);
  if (!cleaned) return null;
  for (const blob of candidates(cleaned)) {
    let value: unknown;
    try {
      value = loadsRepaired(blob);
    } catch {
      continue;
    }
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
    if (Array.isArray(value) && value.length > 0 && value.every((i) => i && typeof i === 'object')) {
      return { steps: value };
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Plan construction
// ---------------------------------------------------------------------------

/** The always-available degraded plan: answer the request directly. */
export function singleStepPlan(goal: string, opts: { modelUsed?: string; intent?: string } = {}): Plan {
  const text = (opts.intent || goal).split(/\s+/).filter(Boolean).join(' ') || 'Respond to the user.';
  return {
    goal: goal.split(/\s+/).filter(Boolean).join(' ') || text,
    steps: [{ index: 0, intent: text, tool: undefined, args: {}, rationale: 'direct response' }],
    multiStep: false,
    modelUsed: opts.modelUsed,
  };
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

/** Turn one loosely-shaped step value into a typed `Step`. */
function coerceStep(raw: unknown, index: number, toolNames: Set<string>): Step | null {
  const record = typeof raw === 'string' ? { intent: raw } : asRecord(raw);
  if (!record) return null;

  const intentRaw = record.intent ?? record.action ?? record.description ?? record.step;
  if (typeof intentRaw !== 'string' || !intentRaw.trim()) return null;

  let tool: string | undefined;
  const toolRaw = record.tool ?? record.tool_name ?? record.name;
  if (typeof toolRaw === 'string' && toolRaw.trim()) {
    tool = toolNames.size > 0 && !toolNames.has(toolRaw) ? undefined : toolRaw;
  }

  const argsRaw = record.args ?? record.arguments ?? record.params;
  const args = asRecord(argsRaw) ?? (argsRaw !== undefined ? { value: argsRaw } : {});

  const rationaleRaw = record.rationale ?? record.why ?? record.reason ?? '';
  const rationale = (typeof rationaleRaw === 'string' ? rationaleRaw : String(rationaleRaw))
    .split(/\s+/)
    .filter(Boolean)
    .join(' ')
    .slice(0, 280);

  return {
    index,
    intent: intentRaw.split(/\s+/).filter(Boolean).join(' '),
    tool,
    args,
    rationale,
  };
}

export interface ParsePlanPayloadOptions {
  goalFallback: string;
  toolNames?: Set<string>;
  modelUsed?: string;
  maxSteps?: number;
}

/** Validate and normalise a raw plan payload into a typed `Plan`. */
export function parsePlanPayload(payload: Record<string, unknown>, opts: ParsePlanPayloadOptions): Plan | null {
  let rawSteps = payload.steps ?? payload.plan ?? payload.actions;
  if (rawSteps && typeof rawSteps === 'object' && !Array.isArray(rawSteps)) {
    rawSteps = Object.values(rawSteps as Record<string, unknown>);
  }
  if (!Array.isArray(rawSteps) || rawSteps.length === 0) return null;

  const names = opts.toolNames ?? new Set<string>();
  const maxSteps = opts.maxSteps ?? MAX_STEPS;
  const steps: Step[] = [];
  for (const raw of rawSteps) {
    const step = coerceStep(raw, steps.length, names);
    if (step) steps.push(step);
    if (steps.length >= maxSteps) break;
  }
  if (steps.length === 0) return null;

  const goalRaw = payload.goal ?? payload.objective;
  const goal = typeof goalRaw === 'string' && goalRaw.trim() ? goalRaw : opts.goalFallback;

  const declared = payload.multi_step;
  const multi = typeof declared === 'boolean' ? declared : steps.length > 1;

  return {
    goal: goal.split(/\s+/).filter(Boolean).join(' '),
    steps,
    multiStep: multi && steps.length > 1,
    modelUsed: opts.modelUsed,
  };
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

export interface PlannerOptions {
  maxSteps?: number;
  timeoutMs?: number;
}

/**
 * Produces a typed `Plan` from a user request.
 *
 * Routes the request through the router's `planning` chain (which owns its
 * own network-level fallback and backoff). If the winning model's reply
 * can't be mined for a usable plan, this degrades to a single-step plan
 * rather than throwing.
 */
export class Planner {
  private readonly router: ModelRouter;
  private readonly maxSteps: number;
  private readonly timeoutMs: number;

  constructor(router: ModelRouter, opts: PlannerOptions = {}) {
    this.router = router;
    this.maxSteps = Math.max(1, opts.maxSteps ?? MAX_STEPS);
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  /** Plan `userText`. Always returns a usable Plan; never throws. */
  async plan(userText: string, ctx?: TurnContext | null, tools?: readonly ToolSpec[]): Promise<Plan> {
    const goal = userText.split(/\s+/).filter(Boolean).join(' ');
    if (!goal) return singleStepPlan('Respond to the user.', { modelUsed: 'empty-input' });

    const toolNames = new Set((tools ?? []).map((spec) => spec.name));
    const messages = this.buildMessages(goal, ctx, tools);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error('planner timed out')), this.timeoutMs);
    let raw: string;
    let modelUsed: string;
    try {
      const routed = await this.router.completePlanning(messages, controller.signal);
      raw = routed.text;
      modelUsed = routed.modelUsed;
    } catch {
      return singleStepPlan(goal, { modelUsed: 'fallback' });
    } finally {
      clearTimeout(timer);
    }

    const payload = extractJsonObject(raw ?? '');
    if (!payload) return singleStepPlan(goal, { modelUsed: 'fallback' });

    const plan = parsePlanPayload(payload, {
      goalFallback: goal,
      toolNames,
      modelUsed,
      maxSteps: this.maxSteps,
    });
    return plan ?? singleStepPlan(goal, { modelUsed: 'fallback' });
  }

  // -- prompt assembly -----------------------------------------------------

  private buildMessages(goal: string, ctx: TurnContext | null | undefined, tools: readonly ToolSpec[] | undefined): Message[] {
    const blocks: string[] = [];

    if (tools && tools.length > 0) {
      const lines = ['AVAILABLE TOOLS (use these exact names or null):'];
      for (const spec of tools) {
        const properties = (spec.parameters as { properties?: Record<string, unknown> } | undefined)?.properties;
        const params = properties ? Object.keys(properties).sort() : [];
        const sig = params.length > 0 ? params.join(', ') : 'no parameters';
        const desc = spec.description.split(/\s+/).join(' ');
        lines.push(`- ${spec.name}(${sig}) — ${desc}`);
      }
      blocks.push(lines.join('\n'));
    } else {
      blocks.push('AVAILABLE TOOLS: none. Every step must have "tool": null.');
    }

    if (ctx) {
      if (ctx.recalled.length > 0) {
        const known = ctx.recalled
          .slice(0, 5)
          .map((hit) => hit.text.split(/\s+/).join(' ').slice(0, 160))
          .join('; ');
        blocks.push(`KNOWN CONTEXT ABOUT THE USER: ${known}`);
      }
      if (ctx.vision && (ctx.vision.scene || ctx.vision.text)) {
        const seen = ctx.vision.scene || ctx.vision.text;
        blocks.push(`CAMERA SEES: ${seen.split(/\s+/).join(' ').slice(0, 300)}`);
      }
      if (ctx.history.length > 0) {
        const recent = ctx.history
          .slice(-4)
          .filter((m) => m.content)
          .map((m) => `${m.role}: ${m.content.split(/\s+/).join(' ').slice(0, 200)}`);
        if (recent.length > 0) blocks.push(`RECENT CONVERSATION:\n${recent.join('\n')}`);
      }
    }

    blocks.push(`REQUEST TO PLAN:\n${goal}`);
    blocks.push('Return only the JSON object.');

    return [
      { role: 'system', content: PLANNER_SYSTEM_PROMPT },
      { role: 'user', content: blocks.join('\n\n') },
    ];
  }
}
