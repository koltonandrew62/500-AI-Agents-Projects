/**
 * Tool registry: registration, a dependency-free JSON-Schema-subset
 * validator, and the `runTool` dispatcher the agent loop calls.
 *
 * Ported from `backend/app/core/tools/base.py`. Tools are never hardcoded
 * into the agent loop; it always calls `registry.all()` / `runTool(name, args)`.
 */

import type { Tool, ToolResult, ToolSpec } from '../types.js';

/** Safety-net ceiling for any single tool call (individual tools may enforce tighter timeouts). */
const DEFAULT_TOOL_TIMEOUT_MS = 30_000;

class ToolRegistry {
  private readonly tools = new Map<string, Tool>();

  register(tool: Tool): Tool {
    if (!tool.name) {
      throw new Error(`tool ${JSON.stringify(tool)} has no non-empty 'name'`);
    }
    if (this.tools.has(tool.name)) {
      throw new Error(`duplicate tool name: ${tool.name}`);
    }
    this.tools.set(tool.name, tool);
    return tool;
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  all(): ToolSpec[] {
    return [...this.tools.values()].map((t) => ({
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    }));
  }

  names(): string[] {
    return [...this.tools.keys()];
  }
}

export const registry = new ToolRegistry();

// ---------------------------------------------------------------------------
// JSON-Schema-subset validation (no ajv dependency, by design)
// ---------------------------------------------------------------------------
//
// Supports the subset of JSON Schema actually used by tool authors here:
// type, properties, required, enum, minimum/maximum, minLength/maxLength,
// items, additionalProperties. Unknown keywords are ignored rather than
// rejected — permissive by design so schemas can carry descriptive-only keys.

type JsonSchema = Record<string, unknown>;

function jsTypeOf(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value; // 'string' | 'number' | 'boolean' | 'object' | 'undefined' | ...
}

function checkType(value: unknown, expected: string): boolean {
  switch (expected) {
    case 'string':
      return typeof value === 'string';
    case 'boolean':
      return typeof value === 'boolean';
    case 'integer':
      return typeof value === 'number' && Number.isInteger(value);
    case 'number':
      return typeof value === 'number' && Number.isFinite(value);
    case 'object':
      return jsTypeOf(value) === 'object';
    case 'array':
      return Array.isArray(value);
    case 'null':
      return value === null;
    default:
      return true; // unrecognized type keyword — don't block on it
  }
}

function validateValue(value: unknown, schema: JsonSchema, path: string): string | null {
  const enumVals = schema.enum as unknown[] | undefined;
  if (enumVals && !enumVals.includes(value)) {
    return `${path}: must be one of ${JSON.stringify(enumVals)}`;
  }

  const expectedType = schema.type as string | undefined;
  if (expectedType && !checkType(value, expectedType)) {
    return `${path}: expected type '${expectedType}', got ${jsTypeOf(value)}`;
  }

  if (expectedType === 'string' && typeof value === 'string') {
    const minLength = schema.minLength as number | undefined;
    const maxLength = schema.maxLength as number | undefined;
    if (minLength !== undefined && value.length < minLength) {
      return `${path}: shorter than minLength ${minLength}`;
    }
    if (maxLength !== undefined && value.length > maxLength) {
      return `${path}: longer than maxLength ${maxLength}`;
    }
  }

  if ((expectedType === 'integer' || expectedType === 'number') && typeof value === 'number') {
    const minimum = schema.minimum as number | undefined;
    const maximum = schema.maximum as number | undefined;
    if (minimum !== undefined && value < minimum) {
      return `${path}: below minimum ${minimum}`;
    }
    if (maximum !== undefined && value > maximum) {
      return `${path}: above maximum ${maximum}`;
    }
  }

  if (expectedType === 'array' && Array.isArray(value)) {
    const itemSchema = schema.items as JsonSchema | undefined;
    if (itemSchema) {
      for (let i = 0; i < value.length; i++) {
        const err = validateValue(value[i], itemSchema, `${path}[${i}]`);
        if (err) return err;
      }
    }
  }

  if (expectedType === 'object' && jsTypeOf(value) === 'object') {
    const err = validateObject(value as Record<string, unknown>, schema, path);
    if (err) return err;
  }

  return null;
}

function validateObject(args: Record<string, unknown>, schema: JsonSchema, path = '$'): string | null {
  if (jsTypeOf(args) !== 'object') {
    return `${path}: expected an object, got ${jsTypeOf(args)}`;
  }

  const required = (schema.required as string[] | undefined) ?? [];
  for (const key of required) {
    if (!(key in args)) {
      return `${path}: missing required field '${key}'`;
    }
  }

  const properties = (schema.properties as Record<string, JsonSchema> | undefined) ?? {};
  // Closed-schema guard: additionalProperties:false rejects any argument the
  // tool doesn't declare — stops a confused/adversarial caller from smuggling
  // extra kwargs into `run()`.
  if (schema.additionalProperties === false) {
    const extra = Object.keys(args).filter((k) => !(k in properties));
    if (extra.length > 0) {
      return `${path}: unexpected field(s) ${JSON.stringify(extra.sort())}`;
    }
  }

  for (const [key, value] of Object.entries(args)) {
    const propSchema = properties[key];
    if (!propSchema) continue;
    const err = validateValue(value, propSchema, `${path}.${key}`);
    if (err) return err;
  }

  return null;
}

/** Validate `args` against `schema`. Returns an error string, or null if valid. */
export function validateSchema(schema: JsonSchema, args: Record<string, unknown>): string | null {
  const type = (schema.type as string | undefined) ?? 'object';
  if (type !== 'object') return null;
  return validateObject(args, schema);
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

function withTimeout<T>(promise: Promise<T>, ms: number, onTimeout: () => T): Promise<T> {
  return new Promise<T>((resolve) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        resolve(onTimeout());
      }
    }, ms);
    promise.then(
      (value) => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve(value);
        }
      },
      () => {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          // Errors are handled by the caller's own try/catch around `tool.run`,
          // so this branch only fires if that guard is somehow bypassed.
          resolve(onTimeout());
        }
      },
    );
  });
}

/**
 * Look up, schema-validate, and execute a tool by name.
 *
 * Single entrypoint the agent loop should call. Never throws: unknown tool
 * names, schema violations, timeouts, and exceptions raised by the tool
 * implementation itself are all normalized into `{ ok: false, ... }` so one
 * misbehaving tool can never take down a turn.
 */
export async function runTool(
  name: string,
  args?: Record<string, unknown>,
  timeoutMs: number = DEFAULT_TOOL_TIMEOUT_MS,
): Promise<ToolResult> {
  const callArgs = { ...(args ?? {}) };

  const tool = registry.get(name);
  if (!tool) {
    return { ok: false, output: '', summary: `Unknown tool: ${JSON.stringify(name)}` };
  }

  const error = validateSchema(tool.parameters, callArgs);
  if (error !== null) {
    return { ok: false, output: '', summary: `Invalid arguments for ${JSON.stringify(name)}: ${error}` };
  }

  let ran: Promise<ToolResult>;
  try {
    ran = tool.run(callArgs);
  } catch (exc) {
    // Synchronous throw before the tool's own promise even starts.
    return { ok: false, output: '', summary: `Tool ${JSON.stringify(name)} failed: ${String(exc)}` };
  }

  const result = await withTimeout(
    ran.catch(
      (exc): ToolResult => ({
        ok: false,
        output: '',
        summary: `Tool ${JSON.stringify(name)} failed: ${exc instanceof Error ? exc.message : String(exc)}`,
      }),
    ),
    timeoutMs,
    (): ToolResult => ({
      ok: false,
      output: '',
      summary: `Tool ${JSON.stringify(name)} timed out after ${Math.round(timeoutMs / 1000)}s`,
    }),
  );

  if (typeof result !== 'object' || result === null || typeof (result as ToolResult).ok !== 'boolean') {
    return { ok: false, output: '', summary: `Tool ${JSON.stringify(name)} returned an invalid result type` };
  }
  return result;
}
