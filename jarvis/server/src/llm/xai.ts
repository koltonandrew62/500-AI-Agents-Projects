/**
 * xAI (Grok) implementation of the `LLMProvider` interface (types.ts).
 *
 * xAI's API at `https://api.x.ai/v1` is OpenAI-compatible — same request and
 * SSE response shape as OpenRouter — so this mirrors `openrouter.ts`'s proven
 * parsing logic closely. Kept as an independent file rather than factored
 * into a shared base: `openrouter.ts` is already shipped and tested, and this
 * integration doesn't need to risk it for a DRY improvement nobody asked for.
 *
 * Unlike OpenRouter, xAI is NOT a free tier — every call here is billed by
 * xAI per token. This class only performs single-model calls; it deliberately
 * does not own fallback/retry chain logic the way `ModelRouter` does for
 * OpenRouter (see router.ts) — that's a decision for whoever wires this
 * provider in, not baked in here.
 */

import type { LLMDelta, LLMProvider, Message, ToolSpec } from '../types.js';
import { BadResponse, NoKey, ProviderDown, RateLimited } from './errors.js';

const DEFAULT_BASE_URL = 'https://api.x.ai/v1';
const DEFAULT_TIMEOUT_MS = 90_000;

export interface XaiOptions {
  apiKey?: string;
  baseUrl?: string;
  timeoutMs?: number;
}

type WireContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

interface WireMessage {
  role: string;
  content: string | WireContentPart[];
  name?: string;
  tool_call_id?: string;
}

interface WireToolCallDelta {
  index?: number;
  id?: string;
  function?: { name?: string; arguments?: string };
}

function resolveApiKey(explicit: string | undefined): string {
  const key = explicit ?? process.env.XAI_API_KEY;
  if (!key) throw new NoKey();
  return key;
}

function toDataUri(image: string): string {
  return image.startsWith('data:') ? image : `data:image/jpeg;base64,${image}`;
}

function messageToWire(msg: Message, extraImage?: string): WireMessage {
  const images = [...(msg.images ?? []), ...(extraImage ? [extraImage] : [])];
  const wire: WireMessage = { role: msg.role, content: msg.content };
  if (images.length > 0) {
    const parts: WireContentPart[] = [];
    if (msg.content) parts.push({ type: 'text', text: msg.content });
    for (const image of images) parts.push({ type: 'image_url', image_url: { url: toDataUri(image) } });
    wire.content = parts;
  }
  if (msg.name) wire.name = msg.name;
  if (msg.toolCallId) wire.tool_call_id = msg.toolCallId;
  return wire;
}

function toolsToWire(tools: ToolSpec[] | undefined): unknown[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  return tools.map((t) => ({
    type: 'function',
    function: { name: t.name, description: t.description, parameters: t.parameters },
  }));
}

function parseRetryAfter(headers: Headers): number | null {
  const raw = headers.get('retry-after');
  if (raw === null) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

/** Accumulates fragmented OpenAI-style streamed tool-call deltas by index — see openrouter.ts for the full rationale. */
class ToolCallAccumulator {
  private readonly byIndex = new Map<number, { id: string; name: string; args: string }>();
  private readonly order: number[] = [];

  add(deltas: WireToolCallDelta[]): void {
    for (const delta of deltas) {
      const index = delta.index ?? 0;
      let entry = this.byIndex.get(index);
      if (!entry) {
        entry = { id: '', name: '', args: '' };
        this.byIndex.set(index, entry);
        this.order.push(index);
      }
      if (delta.id) entry.id = delta.id;
      const fn = delta.function;
      if (fn?.name) entry.name += fn.name;
      if (fn?.arguments) entry.args += fn.arguments;
    }
  }

  finalize(model: string): LLMDelta[] {
    const deltas: LLMDelta[] = [];
    for (const index of this.order) {
      const entry = this.byIndex.get(index);
      if (!entry) continue;
      if (!entry.name) {
        throw new BadResponse(model, `streamed tool call at index ${index} has no function name`);
      }
      let args: Record<string, unknown>;
      try {
        args = JSON.parse(entry.args || '{}') as Record<string, unknown>;
      } catch (exc) {
        throw new BadResponse(
          model,
          `could not parse arguments for tool '${entry.name}': ${(exc as Error).message}`,
        );
      }
      deltas.push({ text: '', toolName: entry.name, toolArgs: args, finished: false });
    }
    this.byIndex.clear();
    this.order.length = 0;
    return deltas;
  }
}

export class XaiProvider implements LLMProvider {
  readonly model: string;
  private readonly apiKey: string | undefined;
  private readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(model: string, opts: XaiOptions = {}) {
    this.model = model;
    this.apiKey = opts.apiKey;
    this.baseUrl = (opts.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, '');
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  private get completionsUrl(): string {
    return `${this.baseUrl}/chat/completions`;
  }

  private headers(): Record<string, string> {
    const key = resolveApiKey(this.apiKey);
    return {
      Authorization: `Bearer ${key}`,
      'Content-Type': 'application/json',
    };
  }

  private buildPayload(
    messages: Message[],
    opts: { tools?: ToolSpec[]; stream: boolean; extraImageForLast?: string },
  ): Record<string, unknown> {
    const lastIndex = messages.length - 1;
    const wireMessages = messages.map((m, i) =>
      messageToWire(m, opts.extraImageForLast && i === lastIndex ? opts.extraImageForLast : undefined),
    );
    const payload: Record<string, unknown> = {
      model: this.model,
      messages: wireMessages,
      stream: opts.stream,
    };
    const wireTools = toolsToWire(opts.tools);
    if (wireTools) {
      payload.tools = wireTools;
      payload.tool_choice = 'auto';
    }
    return payload;
  }

  /** Combine a caller signal with our own request timeout. */
  private linkedSignal(signal: AbortSignal | undefined): { signal: AbortSignal; cleanup: () => void } {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(new Error('timeout')), this.timeoutMs);
    const onAbort = (): void => controller.abort(signal?.reason);
    signal?.addEventListener('abort', onAbort);
    return {
      signal: controller.signal,
      cleanup: () => {
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
      },
    };
  }

  private async post(payload: Record<string, unknown>, signal: AbortSignal | undefined): Promise<Response> {
    const linked = this.linkedSignal(signal);
    let response: Response;
    try {
      response = await fetch(this.completionsUrl, {
        method: 'POST',
        headers: this.headers(),
        body: JSON.stringify(payload),
        signal: linked.signal,
      });
    } catch (exc) {
      if (exc instanceof NoKey) throw exc;
      throw new ProviderDown(this.model, (exc as Error).message);
    } finally {
      linked.cleanup();
    }
    await this.raiseForErrorStatus(response);
    return response;
  }

  private async raiseForErrorStatus(response: Response): Promise<void> {
    if (response.status === 429) {
      throw new RateLimited(this.model, parseRetryAfter(response.headers));
    }
    if (response.status >= 500) {
      const body = await response.text();
      throw new ProviderDown(this.model, `HTTP ${response.status}: ${body.slice(0, 500)}`);
    }
    if (response.status >= 400) {
      const body = await response.text();
      throw new BadResponse(this.model, `HTTP ${response.status}: ${body.slice(0, 500)}`);
    }
  }

  private extractContent(data: unknown): string {
    const choices = (data as { choices?: Array<{ message?: { content?: string } }> })?.choices;
    const content = choices?.[0]?.message?.content;
    if (typeof content !== 'string') {
      throw new BadResponse(this.model, 'unexpected response shape (no message content)');
    }
    return content;
  }

  async *stream(
    messages: Message[],
    tools?: ToolSpec[],
    signal?: AbortSignal,
  ): AsyncIterable<LLMDelta> {
    const payload = this.buildPayload(messages, { tools, stream: true });
    const response = await this.post(payload, signal);
    if (!response.body) throw new BadResponse(this.model, 'streaming response had no body');

    const accumulator = new ToolCallAccumulator();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let newlineIndex: number;
        while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
          const rawLine = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 1);
          const line = rawLine.trim();
          if (!line || line.startsWith(':') || !line.startsWith('data:')) continue;
          const data = line.slice('data:'.length).trim();
          if (data === '[DONE]') {
            for (const toolDelta of accumulator.finalize(this.model)) yield toolDelta;
            yield { text: '', finished: true };
            return;
          }

          let chunk: unknown;
          try {
            chunk = JSON.parse(data);
          } catch (exc) {
            throw new BadResponse(this.model, `malformed SSE JSON: ${(exc as Error).message}`);
          }

          const choice = (
            chunk as {
              choices?: Array<{
                delta?: { content?: string; tool_calls?: WireToolCallDelta[] };
                finish_reason?: string | null;
              }>;
            }
          ).choices?.[0];
          if (!choice) continue;

          const delta = choice.delta ?? {};
          if (delta.content) yield { text: delta.content, finished: false };
          if (delta.tool_calls) accumulator.add(delta.tool_calls);
          if (choice.finish_reason) {
            for (const toolDelta of accumulator.finalize(this.model)) yield toolDelta;
          }
        }
      }
    } catch (exc) {
      if (exc instanceof BadResponse || exc instanceof RateLimited || exc instanceof ProviderDown) throw exc;
      throw new ProviderDown(this.model, (exc as Error).message);
    } finally {
      reader.releaseLock();
    }

    // Stream closed without an explicit [DONE]/finish_reason -- flush any
    // tool calls that never got one and terminate the sequence anyway.
    for (const toolDelta of accumulator.finalize(this.model)) yield toolDelta;
    yield { text: '', finished: true };
  }

  async complete(messages: Message[], signal?: AbortSignal): Promise<string> {
    const payload = this.buildPayload(messages, { stream: false });
    const response = await this.post(payload, signal);
    let data: unknown;
    try {
      data = await response.json();
    } catch (exc) {
      throw new BadResponse(this.model, `invalid JSON body: ${(exc as Error).message}`);
    }
    return this.extractContent(data);
  }

  async vision(messages: Message[], imageB64: string, signal?: AbortSignal): Promise<string> {
    if (messages.length === 0) throw new BadResponse(this.model, 'vision() requires at least one message');
    const payload = this.buildPayload(messages, { stream: false, extraImageForLast: imageB64 });
    const response = await this.post(payload, signal);
    let data: unknown;
    try {
      data = await response.json();
    } catch (exc) {
      throw new BadResponse(this.model, `invalid JSON body: ${(exc as Error).message}`);
    }
    return this.extractContent(data);
  }
}
