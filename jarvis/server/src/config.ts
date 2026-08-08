/**
 * Runtime configuration for J.A.R.V.I.S. — env-driven, with safe defaults.
 *
 * MUST be import-safe with no `.env` file and no `OPENROUTER_API_KEY` present:
 * every other module does `import { config } from './config.js'` at import
 * time, so a missing key must never throw here. The LLM layer is responsible
 * for raising a clear, user-facing error the first time it actually tries to
 * call OpenRouter without a key.
 *
 * Ported from `backend/app/config.py`. Model chains per PORT-CONTRACTS.md §4.
 */

import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { homedir } from 'node:os';
import * as dotenv from 'dotenv';

dotenv.config();

// ---------------------------------------------------------------------------
// Free-tier model chains (PORT-CONTRACTS.md §4 — unchanged, no Google models)
// ---------------------------------------------------------------------------

export const DEFAULT_PLANNING_MODELS: readonly string[] = [
  'deepseek/deepseek-r1:free',
  'qwen/qwen3-235b-a22b:free',
  'meta-llama/llama-3.3-70b-instruct:free',
];

export const DEFAULT_CHAT_MODELS: readonly string[] = [
  'meta-llama/llama-3.3-70b-instruct:free',
  'qwen/qwen3-235b-a22b:free',
];

export const DEFAULT_VISION_MODELS: readonly string[] = [
  'meta-llama/llama-3.2-11b-vision-instruct:free',
  'qwen/qwen2.5-vl-72b-instruct:free',
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function env(name: string): string | undefined {
  const value = process.env[name];
  return value && value.trim() !== '' ? value.trim() : undefined;
}

function envInt(name: string, fallback: number): number {
  const raw = env(name);
  if (raw === undefined) return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function envBool(name: string, fallback: boolean): boolean {
  const raw = env(name);
  if (raw === undefined) return fallback;
  return ['1', 'true', 'yes', 'on'].includes(raw.toLowerCase());
}

/** `FOO=a,b,c` or a JSON array string -> string[]. Empty/missing -> the given default. */
function envList(name: string, fallback: readonly string[]): string[] {
  const raw = env(name);
  if (raw === undefined) return [...fallback];
  const trimmed = raw.trim();
  if (!trimmed) return [];
  if (trimmed.startsWith('[')) {
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (Array.isArray(parsed)) return parsed.map(String);
    } catch {
      // fall through to CSV parsing
    }
  }
  return trimmed
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function expandPath(raw: string): string {
  const withHome = raw.startsWith('~') ? raw.replace(/^~/, homedir()) : raw;
  return resolve(withHome);
}

// ---------------------------------------------------------------------------
// Config shape
// ---------------------------------------------------------------------------

export interface ModelChains {
  planning: string[];
  chat: string[];
  vision: string[];
}

export interface Config {
  /** OpenRouter API key. Required at call time, not at import time. */
  openrouterApiKey: string;
  openrouterBaseUrl: string;

  /**
   * xAI (Grok) API key — optional. Unlike OpenRouter this is NOT a free
   * tier; calls are billed by xAI per token. Required at call time only, not
   * at import time, matching the OpenRouter key's degrade-gracefully pattern.
   */
  xaiApiKey: string;
  xaiBaseUrl: string;
  xaiModel: string;
  xaiVisionModel: string;
  /**
   * Which named chains ('planning'|'chat'|'vision') route through xAI
   * instead of the free OpenRouter chain. Empty by default -- xAI is wired
   * in but fully inert until the user opts a chain in explicitly.
   */
  xaiChains: string[];

  models: ModelChains;

  workspaceRoot: string;
  memoryDbPath: string;
  tasksDbPath: string;

  host: string;
  port: number;
  corsOrigins: string[];

  telemetryIntervalMs: number;
  visionEnabled: boolean;
  voiceEnabled: boolean;

  maxToolIterations: number;
  logLevel: string;

  /** Create the workspace root and DB parent directories if missing. Idempotent, never throws. */
  ensureDirs(): void;
}

const defaultWorkspaceRoot = resolve(homedir(), 'jarvis-workspace');

function buildConfig(): Config {
  const workspaceRoot = expandPath(env('JARVIS_WORKSPACE_ROOT') ?? defaultWorkspaceRoot);
  const memoryDbPath = expandPath(
    env('JARVIS_MEMORY_DB_PATH') ?? resolve(workspaceRoot, 'memory.db'),
  );
  const tasksDbPath = expandPath(
    env('JARVIS_TASKS_DB_PATH') ?? resolve(workspaceRoot, 'tasks.db'),
  );

  return {
    openrouterApiKey: env('OPENROUTER_API_KEY') ?? '',
    openrouterBaseUrl: env('OPENROUTER_BASE_URL') ?? 'https://openrouter.ai/api/v1',

    xaiApiKey: env('XAI_API_KEY') ?? '',
    xaiBaseUrl: env('XAI_BASE_URL') ?? 'https://api.x.ai/v1',
    // Model names current as of this integration -- xAI's lineup moves
    // faster than this file gets revisited. Check https://docs.x.ai/docs/models
    // if either default 404s, and override via env rather than editing here.
    xaiModel: env('XAI_MODEL') ?? 'grok-4',
    xaiVisionModel: env('XAI_VISION_MODEL') ?? 'grok-4',
    xaiChains: envList('JARVIS_XAI_CHAINS', []).filter((c) =>
      (['planning', 'chat', 'vision'] as const).includes(c as 'planning' | 'chat' | 'vision'),
    ),

    models: {
      planning: envList('JARVIS_PLANNING_MODELS', DEFAULT_PLANNING_MODELS),
      chat: envList('JARVIS_CHAT_MODELS', DEFAULT_CHAT_MODELS),
      vision: envList('JARVIS_VISION_MODELS', DEFAULT_VISION_MODELS),
    },

    workspaceRoot,
    memoryDbPath,
    tasksDbPath,

    host: env('JARVIS_HOST') ?? '127.0.0.1',
    port: envInt('JARVIS_PORT', 8000),
    corsOrigins: envList('JARVIS_CORS_ORIGINS', [
      'http://localhost:5173',
      'http://127.0.0.1:5173',
    ]),

    telemetryIntervalMs: envInt('JARVIS_TELEMETRY_INTERVAL_MS', 1000),
    visionEnabled: envBool('JARVIS_VISION_ENABLED', true),
    voiceEnabled: envBool('JARVIS_VOICE_ENABLED', true),

    maxToolIterations: envInt('JARVIS_MAX_TOOL_ITERATIONS', 8),
    logLevel: env('JARVIS_LOG_LEVEL') ?? 'info',

    ensureDirs(): void {
      for (const p of [workspaceRoot, dirname(memoryDbPath), dirname(tasksDbPath)]) {
        try {
          mkdirSync(p, { recursive: true });
        } catch {
          // best-effort; a read-only FS shouldn't crash import/startup here —
          // callers that actually need the directory will surface their own error.
        }
      }
    },
  };
}

export const config: Config = buildConfig();
