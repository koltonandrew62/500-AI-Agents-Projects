/**
 * Turn classification: does this request need the planner?
 *
 * Planning costs a full round-trip to a free-tier reasoning model, which is
 * the slowest thing in the turn. The overwhelming majority of turns are
 * conversation and must never pay for it. So: a pure-heuristic fast path
 * decides confidently at both ends of the spectrum, and only the genuinely
 * ambiguous middle band escalates to a one-token LLM judgement.
 */

import type { Message, TurnContext } from '../types.js';
import type { ModelRouter } from '../llm/router.js';

export type ClassificationSource = 'heuristic' | 'llm' | 'default';

export interface Classification {
  multiStep: boolean;
  score: number;
  reason: string;
  source: ClassificationSource;
}

const CLASSIFIER_TIMEOUT_MS = 12_000;
const AMBIGUOUS_LOW = 1;
const AMBIGUOUS_HIGH = 3;

// ---------------------------------------------------------------------------
// Lexicons
// ---------------------------------------------------------------------------

// Turns that are unambiguously conversation, whatever else they contain.
const CHITCHAT = new Set([
  'hi', 'hey', 'hello', 'yo', 'sup', 'morning', 'good morning', 'good evening',
  'good night', 'thanks', 'thank you', 'ta', 'cheers', 'ok', 'okay', 'k', 'cool',
  'nice', 'great', 'lol', 'hmm', 'yes', 'no', 'yep', 'nope', 'sure', 'stop',
  'cancel', 'never mind', 'nevermind', 'wake up', 'you there', 'you awake',
  'jarvis', 'hey jarvis', 'hello jarvis', 'shut up', 'quiet', 'bye', 'goodbye',
]);

// Explicit sequencing language -- the strongest multi-step signal there is.
const SEQUENCE_PATTERNS = [
  /\bstep[- ]by[- ]step\b/,
  /\bfirst\b[^.?!]{0,80}\bthen\b/,
  /\band then\b/,
  /\bafter (?:that|which|you)\b/,
  /\bonce (?:you|that|it)(?:'ve| have| is| are)?\b[^.?!]{0,40}\b(?:then|next)\b/,
  /\bfollowed by\b/,
  /\bfinally,\b/,
  /\bnext,?\s+(?:you|please)?\s*\w+/,
  /\bfor each\b/,
  /\bone by one\b/,
  /\bmake (?:me )?a plan\b/,
  /\bplan (?:out|this|a)\b/,
  /\bwork (?:out|through)\b/,
  /\bbreak (?:it|this) down\b/,
  /\bin order to\b[^.?!]{0,60}\bthen\b/,
];

// Verbs that imply reaching outside the model -- each one is roughly one tool call.
const ACTION_VERBS = new Set([
  'analyse', 'analyze', 'audit', 'backup', 'benchmark', 'build', 'check',
  'clean', 'compare', 'compile', 'configure', 'convert', 'copy', 'create',
  'debug', 'delete', 'deploy', 'diagnose', 'download', 'edit', 'execute',
  'export', 'fetch', 'find', 'fix', 'generate', 'grep', 'implement', 'index',
  'inspect', 'install', 'kill', 'launch', 'list', 'measure', 'migrate',
  'monitor', 'move', 'open', 'optimise', 'optimize', 'parse', 'patch', 'ping',
  'profile', 'pull', 'read', 'refactor', 'rename', 'render', 'research',
  'restart', 'run', 'save', 'scan', 'search', 'set', 'setup', 'sort', 'start',
  'summarise', 'summarize', 'sync', 'test', 'trace', 'update', 'upgrade',
  'validate', 'verify', 'write',
]);

// Objects that only exist behind a tool.
const TOOL_NOUNS = new Set([
  'file', 'files', 'folder', 'folders', 'directory', 'directories', 'repo',
  'repository', 'script', 'scripts', 'log', 'logs', 'process', 'processes',
  'port', 'ports', 'package', 'packages', 'dependency', 'dependencies',
  'database', 'table', 'endpoint', 'api', 'url', 'website', 'codebase',
  'project', 'test', 'tests', 'commit', 'branch', 'disk', 'memory', 'cpu',
]);

const NUMBERED_LIST_RE = /^\s*(?:\d+[.)]\s+|[-*•]\s+)/gm;
const QUESTION_OPENERS = [
  'what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose', 'is', 'are',
  'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should',
  'am', 'have', 'has', 'tell me', 'explain', 'define', 'remind me',
];

const VISION_PATTERNS = [
  /\b(?:web)?cam(?:era)?\b/,
  /\bwhat (?:do|can) you see\b/,
  /\bcan you see\b/,
  /\blook at (?:this|me|that)\b/,
  /\btake a look\b/,
  /\bwhat am i (?:holding|wearing|doing|pointing)\b/,
  /\bhow many (?:fingers|people|faces)\b/,
  /\bin front of (?:me|the camera)\b/,
  /\bwho (?:am i|is (?:this|that))\b/,
  /\bread (?:this|that|my screen|the screen)\b/,
  /\bwhat(?:'s| is) (?:this|that|on (?:my|the) screen)\b/,
  /\bdescribe (?:this|the scene|what you see)\b/,
  /\bmy face\b/,
  /\bhow do i look\b/,
];
const VISION_RE = new RegExp(VISION_PATTERNS.map((r) => r.source).join('|'), 'i');
const SEQUENCE_RE = new RegExp(SEQUENCE_PATTERNS.map((r) => r.source).join('|'), 'i');
const WORD_RE = /[a-z][a-z'-]*/g;

const CLASSIFIER_PROMPT = `You route requests inside an AI assistant. Decide whether a request needs a
multi-step plan (several dependent actions or tool calls) or can be answered in a
single response or single tool call.

Reply with exactly one word: SIMPLE or COMPLEX. No punctuation, no explanation.`;

// ---------------------------------------------------------------------------
// Heuristics
// ---------------------------------------------------------------------------

/** True when the turn plainly refers to what the camera can see. */
export function needsVision(text: string): boolean {
  return VISION_RE.test(text ?? '');
}

function normalise(text: string): string {
  return (text ?? '').toLowerCase().split(/\s+/).filter(Boolean).join(' ');
}

/**
 * Score a turn without touching the network.
 *
 * A score at or below `AMBIGUOUS_LOW` is confidently simple; at or above
 * `AMBIGUOUS_HIGH` is confidently multi-step; the band between is ambiguous
 * and reported with `multiStep=false` so callers may choose to escalate.
 */
export function classifyHeuristic(text: string): Classification {
  const norm = normalise(text);
  if (!norm) return { multiStep: false, score: 0, reason: 'empty input', source: 'heuristic' };

  const stripped = norm.replace(/[!?.\s]+$/, '');
  if (CHITCHAT.has(stripped)) {
    return { multiStep: false, score: -5, reason: 'conversational filler', source: 'heuristic' };
  }

  const words = norm.match(WORD_RE) ?? [];
  const nWords = words.length;
  const reasons: string[] = [];
  let score = 0;

  if (SEQUENCE_RE.test(norm)) {
    score += 3;
    reasons.push('explicit sequencing language');
  }

  const listItems = (text ?? '').match(NUMBERED_LIST_RE)?.length ?? 0;
  if (listItems >= 2) {
    score += 3;
    reasons.push(`${listItems} enumerated items`);
  }

  const verbs = new Set(words.filter((w) => ACTION_VERBS.has(w)));
  if (verbs.size >= 2) {
    score += 2;
    reasons.push(`${verbs.size} distinct action verbs`);
  } else if (verbs.size === 1) {
    score += 1;
    reasons.push('one action verb');
  }

  const nouns = new Set(words.filter((w) => TOOL_NOUNS.has(w)));
  if (verbs.size > 0 && nouns.size > 0) {
    score += 1;
    reasons.push('action applied to a tool-backed object');
  }

  // " ... and <verb> ..." conjoins two separate jobs.
  if (verbs.size > 0 && /\b(?:and|also|plus|as well as)\b\s+\w+/.test(norm)) {
    const conjoinRe = /\b(?:and|also|plus|as well as)\b\s+([a-z']+)/g;
    let match: RegExpExecArray | null;
    while ((match = conjoinRe.exec(norm)) !== null) {
      if (ACTION_VERBS.has(match[1])) {
        score += 2;
        reasons.push('conjoined second action');
        break;
      }
    }
  }

  const sentences = norm.split(/[.!?\n]+/).filter((s) => s.trim());
  if (sentences.length >= 3 && verbs.size > 0) {
    score += 1;
    reasons.push('multiple imperative sentences');
  }

  if (nWords >= 45 && verbs.size > 0) {
    score += 1;
    reasons.push('long directive');
  }

  // Pull back toward "simple" for plain questions and short turns.
  const isQuestion = norm.endsWith('?') || QUESTION_OPENERS.some((q) => norm.startsWith(q));
  if (isQuestion && score < AMBIGUOUS_HIGH) {
    score -= 1;
    reasons.push('phrased as a question');
  }
  if (nWords <= 6) {
    score -= 2;
    reasons.push('very short');
  }
  if (needsVision(norm) && nWords <= 14) {
    score -= 2;
    reasons.push('single vision lookup');
  }

  const multiStep = score >= AMBIGUOUS_HIGH;
  const reason = reasons.length > 0 ? reasons.join('; ') : 'no strong signals';
  return { multiStep, score, reason, source: 'heuristic' };
}

function isAmbiguous(result: Classification): boolean {
  return result.score > AMBIGUOUS_LOW && result.score < AMBIGUOUS_HIGH;
}

/** Extract SIMPLE/COMPLEX from a model reply, tolerating stray prose. */
function readVerdict(raw: string): boolean | null {
  let text = (raw ?? '').trim().toLowerCase();
  if (!text) return null;
  text = text.replace(/<(think|thinking|reasoning)>[\s\S]*?<\/\1>/gi, ' ');
  if (/\b(complex|multi[- ]?step|plan)\b/.test(text)) return true;
  if (/\b(simple|single[- ]?step|direct|chat)\b/.test(text)) return false;
  return null;
}

async function withTimeout<T>(promise: Promise<T>, ms: number, signal: AbortController): Promise<T> {
  const timer = setTimeout(() => signal.abort(new Error('classifier timed out')), ms);
  try {
    return await promise;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

export interface TurnClassifierOptions {
  useLlmFallback?: boolean;
  timeoutMs?: number;
}

/**
 * Decides whether a turn is worth planning.
 *
 * The heuristic answers alone whenever it is confident. Ambiguous turns fall
 * through to a cheap single-word LLM judgement (via the router's `chat`
 * chain -- cheaper and faster than the planning chain for a one-word
 * verdict), and if that path is unavailable or fails, the turn is treated as
 * simple: a missed plan costs a slightly worse answer, a spurious plan costs
 * the user several seconds.
 */
export class TurnClassifier {
  private readonly router: ModelRouter | null;
  private readonly useLlm: boolean;
  private readonly timeoutMs: number;

  constructor(router: ModelRouter | null = null, opts: TurnClassifierOptions = {}) {
    this.router = router;
    this.useLlm = (opts.useLlmFallback ?? true) && router !== null;
    this.timeoutMs = opts.timeoutMs ?? CLASSIFIER_TIMEOUT_MS;
  }

  /** Classify one turn. Never throws; degrades to the heuristic verdict. */
  async classify(userText: string, ctx?: TurnContext | null): Promise<Classification> {
    const result = classifyHeuristic(userText);
    if (!this.useLlm || !isAmbiguous(result)) return result;

    const verdict = await this.askLlm(userText, ctx);
    if (verdict === null) {
      return {
        multiStep: result.multiStep,
        score: result.score,
        reason: `${result.reason}; llm arbitration unavailable`,
        source: 'default',
      };
    }
    return {
      multiStep: verdict,
      score: result.score,
      reason: `${result.reason}; llm says ${verdict ? 'complex' : 'simple'}`,
      source: 'llm',
    };
  }

  private async askLlm(userText: string, ctx: TurnContext | null | undefined): Promise<boolean | null> {
    if (!this.router) return null;
    const messages: Message[] = [
      { role: 'system', content: CLASSIFIER_PROMPT },
      { role: 'user', content: this.contextLine(userText, ctx) },
    ];
    const controller = new AbortController();
    try {
      const { text } = await withTimeout(
        this.router.completeChat(messages, controller.signal),
        this.timeoutMs,
        controller,
      );
      return readVerdict(text);
    } catch {
      return null;
    }
  }

  private contextLine(userText: string, ctx: TurnContext | null | undefined): string {
    const request = (userText ?? '').split(/\s+/).join(' ').slice(0, 1000);
    if (!ctx || ctx.history.length === 0) return `REQUEST: ${request}`;
    const previous = ctx.history.slice(-2);
    const lines = previous
      .filter((m) => m.content)
      .map((m) => `${m.role}: ${m.content.split(/\s+/).join(' ').slice(0, 200)}`);
    const prefix = lines.length > 0 ? `RECENT:\n${lines.join('\n')}\n\n` : '';
    return `${prefix}REQUEST: ${request}`;
  }
}
