/**
 * The reasoning core of J.A.R.V.I.S.
 *
 * Public surface:
 *  - `AgentLoop`      -- per-turn orchestration: recall, classify, plan, tool-loop, stream.
 *  - `Planner`        -- multi-step planning on the OpenRouter free tier (PORT-CONTRACTS.md §4).
 *  - `TurnClassifier` -- decides whether a turn is worth planning at all.
 *  - `buildSystemPrompt` -- assembles the persona with live context injected.
 */

export { AgentLoop, MAX_ITERATIONS } from './loop.js';
export type { AgentLoopOptions, ToolRegistryLike } from './loop.js';

export { Planner, singleStepPlan, extractJsonObject, parsePlanPayload, stripReasoning } from './planner.js';
export type { PlannerOptions, ParsePlanPayloadOptions } from './planner.js';

export { TurnClassifier, classifyHeuristic, needsVision } from './classifier.js';
export type { Classification, ClassificationSource, TurnClassifierOptions } from './classifier.js';

export {
  buildSystemPrompt,
  degradedReply,
  formatTelemetry,
  formatVision,
  formatMemories,
  formatTools,
  formatPlan,
  JARVIS_IDENTITY,
  JARVIS_STYLE,
  JARVIS_OPERATING_RULES,
} from './persona.js';
export type { BuildSystemPromptOptions } from './persona.js';
