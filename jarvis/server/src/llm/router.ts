/**
 * Free-tier model routing with transparent fallback.
 *
 * Three named chains -- `planning`, `chat`, `vision` -- each an ordered list
 * of OpenRouter free models (PORT-CONTRACTS.md §4). On a rate limit (429) or
 * server error (5xx) the router retries the current model with exponential
 * backoff + jitter (up to `maxAttemptsPerModel`), then transparently falls
 * to the next model in the chain. A malformed response (`BadResponse`) is
 * treated as non-transient and moves straight to the next model. A missing
 * API key (`NoKey`) is a configuration error that no fallback fixes, so it
 * propagates immediately.
 *
 * Which model actually served a request is always exposed to the caller:
 * `RoutedCompletion.modelUsed` for one-shot calls, `RoutedStream.modelUsed`
 * for streaming calls (set as soon as the winning model is known, before the
 * first chunk is yielded).
 */

import type { LLMDelta, Message, RoutedCompletion, ToolSpec } from '../types.js';
import { AllModelsExhausted, BadResponse, NoKey, ProviderDown, RateLimited } from './errors.js';
import { OpenRouterProvider } from './openrouter.js';

export type ChainName = 'planning' | 'chat' | 'vision';

export const PLANNING_CHAIN: readonly string[] = [
  'deepseek/deepseek-r1:free',
  'qwen/qwen3-235b-a22b:free',
  'meta-llama/llama-3.3-70b-instruct:free',
];

export const CHAT_CHAIN: readonly string[] = [
  'meta-llama/llama-3.3-70b-instruct:free',
  'qwen/qwen3-235b-a22b:free',
];

export const VISION_CHAIN: readonly string[] = [
  'meta-llama/llama-3.2-11b-vision-instruct:free',
  'qwen/qwen2.5-vl-72b-instruct:free',
];

const DEFAULT_MAX_ATTEMPTS_PER_MODEL = 3;
const DEFAULT_BASE_DELAY_MS = 500;
const DEFAULT_MAX_DELAY_MS = 8_000;

export interface ModelRouterOptions {
  apiKey?: string;
  chains?: Partial<Record<ChainName, readonly string[]>>;
  maxAttemptsPerModel?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
}

/** An `AsyncIterable<LLMDelta>` that also exposes which model served it. */
export interface RoutedStream extends AsyncIterable<LLMDelta> {
  /**
   * `null` until the router has committed to a model (i.e. it yielded at
   * least one chunk successfully); stable for the rest of the stream.
   */
  readonly modelUsed: string | null;
}

class RoutedStreamImpl implements RoutedStream {
  modelUsed: string | null = null;
  private generator: AsyncGenerator<LLMDelta> | null = null;

  attach(generator: AsyncGenerator<LLMDelta>): void {
    this.generator = generator;
  }

  [Symbol.asyncIterator](): AsyncIterator<LLMDelta> {
    if (!this.generator) throw new Error('RoutedStream used before its generator was attached');
    return this.generator;
  }
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new Error('aborted'));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(signal?.reason ?? new Error('aborted'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Routes calls across a fallback chain of free OpenRouter models. */
export class ModelRouter {
  private readonly chains: Record<ChainName, readonly string[]>;
  private readonly maxAttempts: number;
  private readonly baseDelayMs: number;
  private readonly maxDelayMs: number;
  private readonly providers = new Map<string, OpenRouterProvider>();
  private readonly apiKey: string | undefined;

  /**
   * Best-effort, process-wide "who served last" convenience field. Not
   * concurrency-safe -- under concurrent turns, prefer the `modelUsed` on
   * the individual `RoutedCompletion` / `RoutedStream` instead.
   */
  lastModelUsed: string | null = null;

  constructor(opts: ModelRouterOptions = {}) {
    this.apiKey = opts.apiKey;
    this.chains = {
      planning: opts.chains?.planning ?? PLANNING_CHAIN,
      chat: opts.chains?.chat ?? CHAT_CHAIN,
      vision: opts.chains?.vision ?? VISION_CHAIN,
    };
    this.maxAttempts = Math.max(1, opts.maxAttemptsPerModel ?? DEFAULT_MAX_ATTEMPTS_PER_MODEL);
    this.baseDelayMs = opts.baseDelayMs ?? DEFAULT_BASE_DELAY_MS;
    this.maxDelayMs = opts.maxDelayMs ?? DEFAULT_MAX_DELAY_MS;
  }

  chainFor(name: ChainName): readonly string[] {
    return this.chains[name];
  }

  private providerFor(model: string): OpenRouterProvider {
    let provider = this.providers.get(model);
    if (!provider) {
      provider = new OpenRouterProvider(model, { apiKey: this.apiKey });
      this.providers.set(model, provider);
    }
    return provider;
  }

  private async backoff(attempt: number, signal: AbortSignal | undefined, retryAfterS: number | null): Promise<void> {
    const delay =
      retryAfterS && retryAfterS > 0
        ? Math.min(retryAfterS * 1000, this.maxDelayMs)
        : Math.min(this.maxDelayMs, this.baseDelayMs * 2 ** (attempt - 1));
    const jitter = Math.random() * delay * 0.25;
    await sleep(delay + jitter, signal);
  }

  // -- Non-streaming (complete / vision) -------------------------------------

  private async runChain(
    chainName: ChainName,
    signal: AbortSignal | undefined,
    call: (provider: OpenRouterProvider) => Promise<string>,
  ): Promise<RoutedCompletion> {
    const errors = new Map<string, Error>();
    for (const model of this.chains[chainName]) {
      const provider = this.providerFor(model);
      let attempt = 1;
      while (attempt <= this.maxAttempts) {
        try {
          const text = await call(provider);
          this.lastModelUsed = model;
          return { text, modelUsed: model };
        } catch (exc) {
          if (exc instanceof NoKey) throw exc; // no model in any chain will help
          if (exc instanceof RateLimited) {
            errors.set(model, exc);
            if (attempt < this.maxAttempts) {
              await this.backoff(attempt, signal, exc.retryAfter);
              attempt += 1;
              continue;
            }
            break;
          }
          if (exc instanceof ProviderDown) {
            errors.set(model, exc);
            if (attempt < this.maxAttempts) {
              await this.backoff(attempt, signal, null);
              attempt += 1;
              continue;
            }
            break;
          }
          // BadResponse or anything else: not transient, try the next model.
          errors.set(model, exc as Error);
          break;
        }
      }
    }
    throw new AllModelsExhausted(chainName, errors);
  }

  async completeChat(messages: Message[], signal?: AbortSignal): Promise<RoutedCompletion> {
    return this.runChain('chat', signal, (p) => p.complete(messages, signal));
  }

  async completePlanning(messages: Message[], signal?: AbortSignal): Promise<RoutedCompletion> {
    return this.runChain('planning', signal, (p) => p.complete(messages, signal));
  }

  async vision(messages: Message[], imageB64: string, signal?: AbortSignal): Promise<RoutedCompletion> {
    return this.runChain('vision', signal, (p) => p.vision(messages, imageB64, signal));
  }

  // -- Streaming --------------------------------------------------------------

  private async *streamChainGen(
    chainName: ChainName,
    messages: Message[],
    tools: ToolSpec[] | undefined,
    signal: AbortSignal | undefined,
    routed: RoutedStreamImpl,
  ): AsyncGenerator<LLMDelta> {
    const errors = new Map<string, Error>();
    for (const model of this.chains[chainName]) {
      const provider = this.providerFor(model);
      let attempt = 1;
      while (attempt <= this.maxAttempts) {
        const gen = provider.stream(messages, tools, signal)[Symbol.asyncIterator]();
        let first: IteratorResult<LLMDelta>;
        try {
          first = await gen.next();
        } catch (exc) {
          if (exc instanceof NoKey) throw exc;
          if (exc instanceof RateLimited) {
            errors.set(model, exc);
            if (attempt < this.maxAttempts) {
              await this.backoff(attempt, signal, exc.retryAfter);
              attempt += 1;
              continue;
            }
            break;
          }
          if (exc instanceof ProviderDown) {
            errors.set(model, exc);
            if (attempt < this.maxAttempts) {
              await this.backoff(attempt, signal, null);
              attempt += 1;
              continue;
            }
            break;
          }
          errors.set(model, exc as Error);
          break;
        }

        if (first.done) {
          errors.set(model, new BadResponse(model, 'stream produced no output'));
          break;
        }

        // First chunk arrived -- this model is live. Commit to it: no
        // further fallback once output has reached the caller, since
        // partial tokens can't be un-yielded.
        routed.modelUsed = model;
        this.lastModelUsed = model;
        yield first.value;
        for (;;) {
          const next = await gen.next();
          if (next.done) return;
          yield next.value;
        }
      }
    }
    throw new AllModelsExhausted(chainName, errors);
  }

  streamChat(messages: Message[], tools?: ToolSpec[], signal?: AbortSignal): RoutedStream {
    const routed = new RoutedStreamImpl();
    routed.attach(this.streamChainGen('chat', messages, tools, signal, routed));
    return routed;
  }

  streamPlanning(messages: Message[], tools?: ToolSpec[], signal?: AbortSignal): RoutedStream {
    const routed = new RoutedStreamImpl();
    routed.attach(this.streamChainGen('planning', messages, tools, signal, routed));
    return routed;
  }
}
