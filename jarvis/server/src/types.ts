/**
 * Shared data contracts for J.A.R.V.I.S. (TypeScript edition).
 *
 * This module is the single source of truth for types crossing module
 * boundaries. It is READ-ONLY for build agents: import from it, never edit it.
 *
 * Ported from the original Python `app/models/schemas.py`. The wire protocol
 * is unchanged, so the HTML client and this server stay compatible.
 */

// ---------------------------------------------------------------------------
// Conversation primitives
// ---------------------------------------------------------------------------

export type Role = 'system' | 'user' | 'assistant' | 'tool';

export interface Message {
  role: Role;
  content: string;
  name?: string;
  toolCallId?: string;
  /** base64 JPEG, vision turns only */
  images?: string[];
}

export interface LLMDelta {
  text: string;
  toolName?: string;
  toolArgs?: Record<string, unknown>;
  finished: boolean;
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export interface ToolSpec {
  name: string;
  description: string;
  /** JSON Schema object */
  parameters: Record<string, unknown>;
}

export interface ToolResult {
  ok: boolean;
  output: string;
  summary?: string;
  meta?: Record<string, unknown>;
}

export interface Tool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  run(args: Record<string, unknown>): Promise<ToolResult>;
}

// ---------------------------------------------------------------------------
// Planning (runs on the OpenRouter free tier)
// ---------------------------------------------------------------------------

export interface Step {
  index: number;
  intent: string;
  tool?: string;
  args?: Record<string, unknown>;
  rationale?: string;
}

export interface Plan {
  goal: string;
  steps: Step[];
  multiStep: boolean;
  modelUsed?: string;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

export type MemoryKind = 'fact' | 'event' | 'pref' | 'conv' | 'obs';

export interface MemoryHit {
  id: string;
  text: string;
  kind: MemoryKind;
  score: number;
  createdAt: number;
  meta?: Record<string, unknown>;
}

export interface MemoryStore {
  remember(text: string, kind: MemoryKind, meta?: Record<string, unknown>): Promise<string>;
  recall(query: string, k?: number): Promise<MemoryHit[]>;
  history(limit?: number): Promise<Message[]>;
  appendTurn(role: Role, content: string, turnId?: string): Promise<string>;
  forget(id: string): Promise<boolean>;
  consolidate(threshold?: number): Promise<number>;
}

// ---------------------------------------------------------------------------
// Sensing
// ---------------------------------------------------------------------------

export interface Telemetry {
  cpu: number;
  mem: number;
  disk: number;
  /** KB/s, computed from counter deltas — never cumulative totals */
  net_up: number;
  net_down: number;
  battery: number | null;
  uptime_s: number;
  processes: number;
  temp_c: number | null;
}

export interface DetectedObject {
  label: string;
  confidence: number;
  /** x, y, w, h in source-image pixels */
  box: [number, number, number, number];
}

export interface VisionResult {
  objects: DetectedObject[];
  faces: number;
  hands: number;
  /** OCR */
  text: string;
  /** LLM scene description */
  scene: string;
  width: number;
  height: number;
}

// ---------------------------------------------------------------------------
// WebSocket protocol — unchanged from the Python edition
// ---------------------------------------------------------------------------

export type ClientMessage =
  | { type: 'chat'; text: string; id: string }
  | { type: 'frame'; jpeg_b64: string; id: string }
  | { type: 'voice'; text: string; id: string }
  | { type: 'cancel'; id: string }
  | { type: 'ping' };

export type ServerEvent =
  | { type: 'token'; id: string; text: string }
  | { type: 'done'; id: string; text: string }
  | { type: 'thinking'; id: string; stage: 'planning' | 'tool' | 'vision' | 'recall' }
  | { type: 'tool_call'; id: string; name: string; args: Record<string, unknown> }
  | { type: 'tool_result'; id: string; name: string; ok: boolean; summary: string }
  | ({ type: 'telemetry' } & Telemetry)
  | ({ type: 'vision' } & Partial<VisionResult>)
  | { type: 'speak'; text: string; voice: string }
  | { type: 'log'; level: 'debug' | 'info' | 'warn' | 'error' | 'ok'; text: string }
  | { type: 'error'; id?: string; text: string };

export interface TurnContext {
  turnId: string;
  history: Message[];
  recalled: MemoryHit[];
  telemetry?: Telemetry | null;
  vision?: VisionResult | null;
  lastFrameB64?: string | null;
}

// ---------------------------------------------------------------------------
// LLM provider
// ---------------------------------------------------------------------------

export type ChainName = 'planning' | 'chat' | 'vision';

export interface RoutedCompletion {
  text: string;
  modelUsed: string;
}

export interface LLMProvider {
  stream(
    messages: Message[],
    tools?: ToolSpec[],
    signal?: AbortSignal,
  ): AsyncIterable<LLMDelta>;
  complete(messages: Message[], signal?: AbortSignal): Promise<string>;
  vision(messages: Message[], imageB64: string, signal?: AbortSignal): Promise<string>;
}
