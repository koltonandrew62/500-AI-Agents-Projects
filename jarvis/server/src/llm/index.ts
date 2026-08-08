/**
 * LLM layer: OpenRouter provider and the fallback router.
 *
 * Everything downstream (the agent loop, API routes) should depend on this
 * package's public surface -- `ModelRouter` for normal use, `LLMProvider`
 * (re-exported from `types.ts`) for typing, `OpenRouterProvider` only when a
 * single fixed model is needed without fallback -- and never reach into
 * submodules directly.
 */

export type { LLMProvider } from '../types.js';

export { OpenRouterProvider } from './openrouter.js';
export type { OpenRouterOptions } from './openrouter.js';

export {
  ModelRouter,
  PLANNING_CHAIN,
  CHAT_CHAIN,
  VISION_CHAIN,
} from './router.js';
export type { ChainName, ModelRouterOptions, RoutedStream } from './router.js';

export {
  LLMError,
  NoKey,
  RateLimited,
  ProviderDown,
  BadResponse,
  AllModelsExhausted,
} from './errors.js';
