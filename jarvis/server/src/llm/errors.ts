/**
 * Typed errors for the LLM layer.
 *
 * Every failure mode that can occur while talking to an LLM provider gets its
 * own class so callers (the router, the agent loop) can `instanceof`-branch
 * instead of string-matching a generic `Error`.
 */

/** Base class for every error raised by the LLM layer. */
export class LLMError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * No API key is configured. A configuration error, not a transient one --
 * retrying will never fix it.
 */
export class NoKey extends LLMError {
  constructor(message?: string) {
    super(
      message ??
        'No OpenRouter API key configured. Set OPENROUTER_API_KEY in your ' +
          '.env file (see .env.example) and restart the server.',
    );
  }
}

/** HTTP 429 (or an equivalent rate-limit signal) from a provider. */
export class RateLimited extends LLMError {
  readonly model: string;
  readonly retryAfter: number | null;

  constructor(model: string, retryAfter: number | null = null) {
    const suffix = retryAfter ? ` (retry after ${retryAfter.toFixed(1)}s)` : '';
    super(`Rate limited by model '${model}'${suffix}`);
    this.model = model;
    this.retryAfter = retryAfter;
  }
}

/** HTTP 5xx, timeouts, or connection failures -- the provider is unavailable. */
export class ProviderDown extends LLMError {
  readonly model: string;
  readonly detail: string;

  constructor(model: string, detail = '') {
    super(`Provider unavailable for model '${model}'${detail ? `: ${detail}` : ''}`);
    this.model = model;
    this.detail = detail;
  }
}

/**
 * A 2xx response we could not parse: malformed JSON, missing `choices`, a
 * stream that never terminates cleanly, or an unexpected response shape.
 */
export class BadResponse extends LLMError {
  readonly model: string;
  readonly detail: string;

  constructor(model: string, detail = '') {
    super(`Bad response from model '${model}'${detail ? `: ${detail}` : ''}`);
    this.model = model;
    this.detail = detail;
  }
}

/** Every model in a fallback chain failed. Carries the per-model errors. */
export class AllModelsExhausted extends LLMError {
  readonly chainName: string;
  readonly errors: ReadonlyMap<string, Error>;

  constructor(chainName: string, errors: ReadonlyMap<string, Error>) {
    const detail = Array.from(errors, ([model, err]) => `${model}: ${err.message}`).join('; ');
    super(`All models in chain '${chainName}' failed: ${detail}`);
    this.chainName = chainName;
    this.errors = errors;
  }
}
