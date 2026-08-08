/**
 * Smarter retrieval than raw vector search.
 *
 * A single embedding of the user's literal message is a weak query: people
 * ask "what was that thing I mentioned about the camera" when the stored
 * memory says "prefers the webcam feed muted on startup". `RecallEngine`
 * widens the net by rewriting one message into several retrieval queries,
 * running them all, and fusing the rankings.
 *
 * Every LLM call here is optional. If the provider is down, rate-limited, or
 * has no key, each stage degrades to a deterministic heuristic rather than
 * failing the turn — recall gets fuzzier, the assistant keeps working.
 */

import type { MemoryHit } from '../types.js';

// Reciprocal-rank-fusion damping. 60 is the value from the original RRF
// paper; it keeps a rank-1 hit from a single query from dominating a result
// that placed respectably across all of them.
const RRF_K = 60;

const MAX_EXPANSIONS = 3;

// Turns that never justify a retrieval round-trip.
const TRIVIAL = new Set<string>([
  'hi', 'hey', 'hello', 'yo', 'sup', 'thanks', 'thank you', 'ty', 'ok',
  'okay', 'k', 'cool', 'nice', 'got it', 'sure', 'yes', 'no', 'yep',
  'nope', 'stop', 'cancel', 'nevermind', 'never mind', 'quiet', 'shut up',
  'wake up', 'jarvis', 'hey jarvis', 'good morning', 'good night', 'bye',
]);

const EXPANSION_PROMPT = (n: number, text: string): string =>
  `Rewrite the user's message into up to ${n} short search queries for a personal memory database. Each query should capture a different angle: literal keywords, the underlying topic, and any implied subject.

Return ONLY the queries, one per line, no numbering, no commentary.

User message: ${text}`;

/** The slice of `SqliteMemoryStore` this module needs. */
export interface RecallStore {
  recall(query: string, k?: number): Promise<MemoryHit[]>;
}

/** The slice of an `LLMProvider` this module needs, for query expansion. */
export interface ExpansionClient {
  complete(messages: { role: 'user'; content: string }[]): Promise<string>;
}

/** Multi-query retrieval with reciprocal-rank fusion over a memory store. */
export class RecallEngine {
  private readonly store: RecallStore;
  private readonly llm: ExpansionClient | null;
  private readonly maxExpansions: number;

  constructor(store: RecallStore, llm?: ExpansionClient | null, maxExpansions = MAX_EXPANSIONS) {
    this.store = store;
    this.llm = llm ?? null;
    this.maxExpansions = Math.max(1, maxExpansions);
  }

  // -- gating ----------------------------------------------------------

  /**
   * Fast heuristic: is this turn worth a retrieval round-trip?
   *
   * Deliberately cheap and conservative — when unsure, recall. The cost of
   * a needless lookup is milliseconds; the cost of missing context is the
   * assistant forgetting who it is talking to.
   */
  static shouldRecall(text: string): boolean {
    const cleaned = (text ?? '').trim().toLowerCase().replace(/[^\w\s]/g, '');
    if (!cleaned) return false;
    if (TRIVIAL.has(cleaned)) return false;
    // Very short utterances carry too little signal to retrieve against,
    // unless they are a question ("why?", "when?") that leans on context.
    if (cleaned.split(/\s+/).filter(Boolean).length <= 2 && !(text ?? '').trim().endsWith('?')) {
      return false;
    }
    return true;
  }

  // -- query expansion ---------------------------------------------------

  /** Rewrite one message into several retrieval queries. */
  async expand(query: string): Promise<string[]> {
    const queries = [query];
    if (!this.llm) {
      return queries.concat(heuristicExpansions(query, this.maxExpansions - 1));
    }

    const prompt = EXPANSION_PROMPT(this.maxExpansions, query);
    let raw: string;
    try {
      raw = await this.llm.complete([{ role: 'user', content: prompt }]);
    } catch {
      // provider down / rate limited / no key
      return queries.concat(heuristicExpansions(query, this.maxExpansions - 1));
    }

    const seen = new Set(queries.map((q) => q.toLowerCase()));
    for (const line of (raw ?? '').split('\n')) {
      const candidate = line.trim().replace(/^[-•*0-9.\s]+/, '').trim();
      if (!candidate || candidate.length < 3) continue;
      if (seen.has(candidate.toLowerCase())) continue;
      queries.push(candidate);
      seen.add(candidate.toLowerCase());
      if (queries.length >= this.maxExpansions + 1) break;
    }

    if (queries.length === 1) {
      return queries.concat(heuristicExpansions(query, this.maxExpansions - 1));
    }
    return queries;
  }

  // -- retrieval -----------------------------------------------------------

  /** Multi-query recall fused by reciprocal rank. */
  async recall(query: string, k = 6): Promise<MemoryHit[]> {
    if (!RecallEngine.shouldRecall(query)) return [];

    const queries = await this.expand(query);

    // Over-fetch per query so fusion has material to work with.
    const perQuery = Math.max(k, 8);
    const rankings: MemoryHit[][] = [];
    for (const q of queries) {
      try {
        rankings.push(await this.store.recall(q, perQuery));
      } catch {
        // one query's failure shouldn't sink the whole recall
      }
    }

    return fuse(rankings, k);
  }

  /** Render the winning memories as a compact prompt block. */
  async buildContext(query: string, k = 6): Promise<string> {
    return renderMemories(await this.recall(query, k));
  }
}

// ---------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------

/**
 * Reciprocal-rank fusion, deduplicated by memory id.
 *
 * Scoring by rank rather than raw similarity is what makes this robust: the
 * stores' cosine scores are not comparable across differently-phrased
 * queries, but positions are.
 */
function fuse(rankings: readonly MemoryHit[][], k: number): MemoryHit[] {
  const scores = new Map<string, number>();
  const best = new Map<string, MemoryHit>();

  for (const ranking of rankings) {
    ranking.forEach((hit, position) => {
      scores.set(hit.id, (scores.get(hit.id) ?? 0) + 1 / (RRF_K + position + 1));
      // Keep the representation that scored highest on its own query.
      const current = best.get(hit.id);
      if (!current || hit.score > current.score) {
        best.set(hit.id, hit);
      }
    });
  }

  const ordered = [...scores.entries()].sort((a, b) => b[1] - a[1]);

  const fused: MemoryHit[] = [];
  const seenText = new Set<string>();
  for (const [id] of ordered) {
    const hit = best.get(id);
    if (!hit) continue;
    // Near-duplicate text can survive as distinct rows; collapse it here so
    // the prompt block does not repeat itself.
    const fingerprint = hit.text.toLowerCase().replace(/\W+/g, '').slice(0, 120);
    if (seenText.has(fingerprint)) continue;
    seenText.add(fingerprint);
    fused.push(hit);
    if (fused.length >= k) break;
  }
  return fused;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/** Render memories as a compact prompt block with relative timestamps. */
export function renderMemories(hits: readonly MemoryHit[], now?: number): string {
  if (hits.length === 0) return '';
  const current = now ?? Date.now() / 1000;
  const lines = hits
    .filter((hit) => hit.text && hit.text.trim())
    .map((hit) => `- [${relativeTime(hit.createdAt, current)}] ${hit.text.trim()}`);
  if (lines.length === 0) return '';
  return `RELEVANT MEMORY:\n${lines.join('\n')}`;
}

/** Human-readable age: 'just now', 'yesterday', '3 weeks ago'. */
export function relativeTime(then: number, now?: number): string {
  const current = now ?? Date.now() / 1000;
  const delta = Math.max(0, current - then);

  const minutes = delta / 60;
  if (minutes < 2) return 'just now';
  if (minutes < 60) return `${Math.floor(minutes)} minutes ago`;

  const hours = minutes / 60;
  if (hours < 24) return `${Math.floor(hours)} hour${Math.floor(hours) !== 1 ? 's' : ''} ago`;

  const days = hours / 24;
  if (days < 2) return 'yesterday';
  if (days < 7) return `${Math.floor(days)} days ago`;

  const weeks = days / 7;
  if (weeks < 5) return `${Math.floor(weeks)} week${Math.floor(weeks) !== 1 ? 's' : ''} ago`;

  const months = days / 30;
  if (months < 12) return `${Math.floor(months)} month${Math.floor(months) !== 1 ? 's' : ''} ago`;

  const years = Math.floor(days / 365);
  return `${years} year${years !== 1 ? 's' : ''} ago`;
}

// ---------------------------------------------------------------------------
// Heuristic fallback
// ---------------------------------------------------------------------------

const STOPWORDS = new Set<string>([
  'a', 'an', 'the', 'and', 'or', 'but', 'if', 'is', 'are', 'was', 'were',
  'do', 'does', 'did', 'have', 'has', 'had', 'i', 'you', 'me', 'my',
  'your', 'it', 'that', 'this', 'what', 'when', 'where', 'who', 'how',
  'to', 'of', 'in', 'on', 'for', 'with', 'about', 'can', 'could', 'would',
  'should', 'please', 'jarvis', 'tell', 'again',
]);

/** Keyword-only expansion for when the LLM is unavailable. */
function heuristicExpansions(query: string, limit: number): string[] {
  if (limit <= 0) return [];

  const words = query.toLowerCase().match(/[a-z][a-z0-9'-]+/g) ?? [];
  const keywords = words.filter((w) => !STOPWORDS.has(w) && w.length > 2);
  if (keywords.length === 0) return [];

  const out: string[] = [];
  // The content words alone — drops question scaffolding that dilutes the
  // vector. Dedupe while preserving first-seen order.
  const joined = [...new Set(keywords)].join(' ');
  if (joined && joined !== query.toLowerCase()) out.push(joined);
  // The longest single term, as a narrow high-precision probe.
  if (keywords.length > 1) {
    out.push(keywords.reduce((a, b) => (b.length > a.length ? b : a)));
  }

  return out.slice(0, limit);
}
