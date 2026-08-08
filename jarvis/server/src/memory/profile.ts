/**
 * The durable model of who the user is.
 *
 * `UserProfile` maintains a small set of structured fields (name, location,
 * timezone, occupation, interests, people the user mentions, devices,
 * stated preferences, communication style) backed by individual FACT / PREF
 * memories in the shared memory store.
 *
 * This module depends only on the small structural `ProfileStore` /
 * `ProfileLLM` interfaces below (subsets of `MemoryStore` / `LLMProvider`
 * from `../types.ts`), so it stays usable even if the concrete store or LLM
 * router implementation changes shape elsewhere.
 *
 * Persistence convention: every fact this module writes carries structured
 * `meta`:
 *
 *     { profileField: '<field name>', value: '<value>', key?: '<optional>' }
 *
 * `key` is used for the two dict-shaped fields (`people`, `preferences`) to
 * identify which person / topic the value is about. `load()` trusts this
 * meta as the source of truth (rather than parsing the human-readable
 * sentence text) and reconstructs state via targeted `recall()` queries per
 * field — the store exposes semantic recall, not a list-all-of-kind
 * primitive, so this is the best reconstruction available under the current
 * contract.
 */

import type { MemoryHit, MemoryKind, Message } from '../types.js';

// ---------------------------------------------------------------------------
// Structural dependencies
// ---------------------------------------------------------------------------

/** The slice of `SqliteMemoryStore` this module needs. */
export interface ProfileStore {
  remember(text: string, kind: MemoryKind, meta: Record<string, unknown>): Promise<string>;
  recall(query: string, k?: number): Promise<MemoryHit[]>;
}

/** The slice of an `LLMProvider` this module needs, for fact extraction. */
export interface ProfileLLM {
  complete(messages: Message[]): Promise<string>;
}

// ---------------------------------------------------------------------------
// Field taxonomy
// ---------------------------------------------------------------------------

const SCALAR_FIELDS = ['name', 'location', 'timezone', 'occupation'] as const;
const LIST_FIELDS = ['interests', 'devices', 'communicationStyle'] as const;
const DICT_FIELDS = ['people', 'preferences'] as const;

type ScalarField = (typeof SCALAR_FIELDS)[number];
type ListField = (typeof LIST_FIELDS)[number];
type DictField = (typeof DICT_FIELDS)[number];
type ProfileField = ScalarField | ListField | DictField;

const FIELD_KIND: Record<ProfileField, MemoryKind> = {
  name: 'fact',
  location: 'fact',
  timezone: 'fact',
  occupation: 'fact',
  interests: 'fact',
  people: 'fact',
  devices: 'fact',
  preferences: 'pref',
  communicationStyle: 'pref',
};

// Targeted recall query per field, used by load() to reconstruct state.
const FIELD_QUERIES: Record<ProfileField, string> = {
  name: "the user's name",
  location: "where the user lives, their location",
  timezone: "the user's timezone",
  occupation: "the user's job, occupation, profession",
  interests: 'things the user is interested in, hobbies',
  people: 'people the user has mentioned: family, friends, colleagues, pets',
  devices: 'devices the user owns or uses',
  preferences: "the user's stated preferences, likes and dislikes",
  communicationStyle: "the user's preferred communication style or tone",
};

const SENTENCE_TEMPLATES: Record<ProfileField, (value: string, key?: string) => string> = {
  name: (v) => `The user's name is ${v}.`,
  location: (v) => `The user lives in ${v}.`,
  timezone: (v) => `The user's timezone is ${v}.`,
  occupation: (v) => `The user works as ${v}.`,
  interests: (v) => `The user is interested in ${v}.`,
  devices: (v) => `The user uses a device: ${v}.`,
  people: (v, k) => `The user mentioned ${k}, described as: ${v}.`,
  preferences: (v, k) => `Regarding ${k}, the user prefers: ${v}.`,
  communicationStyle: (v) => `The user's communication style: ${v}.`,
};

const EXTRACTION_SYSTEM_PROMPT = `You extract durable, self-reported facts about a user from one turn of a conversation with their personal AI assistant.

Rules (follow strictly):
1. Only extract facts the USER explicitly and unambiguously stated about THEMSELVES in their own message. Never extract facts about the assistant, and never extract something about another person unless the user is directly stating their relationship to that person.
2. Never infer or guess. A single ambiguous, hypothetical, or third-hand mention ("what if I lived in Texas", "my friend loves hiking") is NOT a fact to extract. Only clear, direct, first-person statements are ("I live in Texas", "I love hiking").
3. Do not repeat anything already present in "Known profile" below -- only report facts that are genuinely NEW or that CHANGE a known value.
4. If nothing new and durable was stated, return an empty JSON object: {}
5. Respond with ONLY a single JSON object. No prose, no markdown fences, no commentary.

JSON shape (all keys optional -- include only what is new or changed):
{
  "name": "string",
  "location": "string",
  "timezone": "string",
  "occupation": "string",
  "interests": ["string", ...],
  "people": [{"name": "string", "relation": "string"}, ...],
  "devices": ["string", ...],
  "preferences": [{"topic": "string", "value": "string"}, ...],
  "communication_style": ["string", ...]
}`;

interface ExtractedProfile {
  name?: string;
  location?: string;
  timezone?: string;
  occupation?: string;
  interests?: string[];
  people?: { name?: string; relation?: string }[];
  devices?: string[];
  preferences?: { topic?: string; value?: string }[];
  communication_style?: string[];
}

function normalize(text: string): string {
  return text.trim().toLowerCase().split(/\s+/).filter(Boolean).join(' ');
}

/**
 * Best-effort extraction of the first top-level JSON object in `text`.
 *
 * LLMs sometimes wrap JSON in prose or markdown fences despite explicit
 * instructions not to; this scans for a balanced `{...}` span (respecting
 * string quoting) instead of assuming the whole response is valid JSON.
 */
function extractJsonObject(text: string): Record<string, unknown> | null {
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escape) {
        escape = false;
      } else if (ch === '\\') {
        escape = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        const candidate = text.slice(start, i + 1);
        try {
          const parsed: unknown = JSON.parse(candidate);
          return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
            ? (parsed as Record<string, unknown>)
            : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// UserProfile
// ---------------------------------------------------------------------------

/**
 * The durable, structured model of who the user is.
 *
 * Construct once per session with a store (and, optionally, an LLM client
 * for fact extraction) and call `load()` before first use.
 */
export class UserProfile {
  private readonly store: ProfileStore;
  private readonly llm: ProfileLLM | null;
  private loaded = false;

  name: string | null = null;
  location: string | null = null;
  timezone: string | null = null;
  occupation: string | null = null;
  interests: string[] = [];
  people: Record<string, string> = {};
  devices: string[] = [];
  preferences: Record<string, string> = {};
  communicationStyle: string[] = [];

  constructor(store: ProfileStore, llm?: ProfileLLM | null) {
    this.store = store;
    this.llm = llm ?? null;
  }

  // -- loading ------------------------------------------------------

  private resetFields(): void {
    this.name = null;
    this.location = null;
    this.timezone = null;
    this.occupation = null;
    this.interests = [];
    this.people = {};
    this.devices = [];
    this.preferences = {};
    this.communicationStyle = [];
  }

  private apply(fieldName: ProfileField, value: string, key?: string): void {
    if ((SCALAR_FIELDS as readonly string[]).includes(fieldName)) {
      (this as unknown as Record<ScalarField, string | null>)[fieldName as ScalarField] = value;
    } else if ((LIST_FIELDS as readonly string[]).includes(fieldName)) {
      const items = this[fieldName as ListField];
      const norms = new Set(items.map(normalize));
      if (!norms.has(normalize(value))) items.push(value);
    } else if ((DICT_FIELDS as readonly string[]).includes(fieldName)) {
      if (key) this[fieldName as DictField][key] = value;
    }
  }

  /**
   * (Re)populate profile fields from the memory store.
   *
   * The store only exposes semantic `recall()`, not a list-all-of-kind
   * primitive, so this issues one targeted query per field and trusts the
   * structured `meta` this module writes (`profileField`, `value`, `key`)
   * as ground truth rather than parsing sentence text.
   */
  async load(): Promise<void> {
    this.resetFields();
    // slot -> (createdAt, value); keeps the most recent write per slot.
    const best = new Map<string, { createdAt: number; value: string }>();

    for (const fieldName of Object.keys(FIELD_QUERIES) as ProfileField[]) {
      let hits: MemoryHit[];
      try {
        hits = await this.store.recall(FIELD_QUERIES[fieldName], 10);
      } catch {
        // provider/store failure -> skip field
        continue;
      }
      for (const hit of hits) {
        const meta = hit.meta ?? {};
        if (meta.profileField !== fieldName) continue;
        const value = meta.value;
        if (!value) continue;
        const key = typeof meta.key === 'string' ? meta.key : undefined;
        const slot = `${fieldName} ${key ?? ''}`;
        const prev = best.get(slot);
        if (!prev || hit.createdAt > prev.createdAt) {
          best.set(slot, { createdAt: hit.createdAt, value: String(value) });
        }
      }
    }

    for (const [slot, { value }] of best) {
      const [fieldName, key] = slot.split(' ');
      this.apply(fieldName as ProfileField, value, key || undefined);
    }
    this.loaded = true;
  }

  // -- rendering ------------------------------------------------------

  /** Compact block for injection into the persona/system prompt. */
  renderForPrompt(): string {
    const lines: string[] = [];
    if (this.name) lines.push(`Name: ${this.name}`);
    if (this.location) lines.push(`Location: ${this.location}`);
    if (this.timezone) lines.push(`Timezone: ${this.timezone}`);
    if (this.occupation) lines.push(`Occupation: ${this.occupation}`);
    if (this.interests.length) lines.push(`Interests: ${this.interests.join(', ')}`);
    if (Object.keys(this.people).length) {
      const people = Object.entries(this.people)
        .map(([name, rel]) => (rel ? `${name} (${rel})` : name))
        .join('; ');
      lines.push(`People: ${people}`);
    }
    if (this.devices.length) lines.push(`Devices: ${this.devices.join(', ')}`);
    if (Object.keys(this.preferences).length) {
      const prefs = Object.entries(this.preferences)
        .map(([topic, val]) => `${topic}: ${val}`)
        .join('; ');
      lines.push(`Preferences: ${prefs}`);
    }
    if (this.communicationStyle.length) {
      lines.push(`Communication style: ${this.communicationStyle.join(', ')}`);
    }
    if (lines.length === 0) return '';
    return `USER PROFILE:\n${lines.map((l) => `- ${l}`).join('\n')}`;
  }

  // -- direct fact access ------------------------------------------------------

  /**
   * Directly persist a single known fact, bypassing LLM extraction. Used
   * both internally by `updateFromTurn()` and by any caller that already
   * knows a fact for certain (e.g. an explicit "remember that ..." command
   * handled elsewhere).
   */
  async setFact(fieldName: ProfileField, value: string, key?: string): Promise<void> {
    if (!(fieldName in FIELD_KIND)) {
      throw new Error(`Unknown profile field: ${String(fieldName)}`);
    }
    const trimmed = value.trim();
    if (!trimmed) return;
    if ((DICT_FIELDS as readonly string[]).includes(fieldName) && !key) {
      throw new Error(`Field ${fieldName} requires a key`);
    }

    const kind = FIELD_KIND[fieldName];
    const text = SENTENCE_TEMPLATES[fieldName](trimmed, key);
    const meta: Record<string, unknown> = { profileField: fieldName, value: trimmed };
    if (key) meta.key = key;

    try {
      await this.store.remember(text, kind, meta);
    } catch {
      // degrade gracefully: skip persisting rather than crash the turn
      return;
    }
    this.apply(fieldName, trimmed, key);
  }

  /** Read a fact from the in-memory profile (no store round-trip). */
  getFact(fieldName: ProfileField, key?: string): string | string[] | Record<string, string> | null {
    if ((SCALAR_FIELDS as readonly string[]).includes(fieldName)) {
      return this[fieldName as ScalarField];
    }
    if ((LIST_FIELDS as readonly string[]).includes(fieldName)) {
      return [...this[fieldName as ListField]];
    }
    if ((DICT_FIELDS as readonly string[]).includes(fieldName)) {
      const values = this[fieldName as DictField];
      return key !== undefined ? (values[key] ?? null) : { ...values };
    }
    throw new Error(`Unknown profile field: ${String(fieldName)}`);
  }

  // -- extraction ------------------------------------------------------

  private extractionMessages(userText: string, assistantText: string): Message[] {
    const known = this.renderForPrompt() || '(nothing known yet)';
    const userPrompt =
      `Known profile:\n${known}\n\n` +
      `User said: ${JSON.stringify(userText)}\n` +
      `Assistant replied: ${JSON.stringify(assistantText)}\n\n` +
      'Extract only NEW durable facts the user stated about themselves, per the rules. Respond with the JSON object only.';
    return [
      { role: 'system', content: EXTRACTION_SYSTEM_PROMPT },
      { role: 'user', content: userPrompt },
    ];
  }

  /**
   * Extract and persist NEW durable facts from one conversation turn.
   *
   * Conservative by design: only saves things the user actually stated
   * about themselves, deduped against the currently loaded profile. Never
   * re-saves something already known. If the LLM is unavailable or fails,
   * this degrades to a no-op for the turn rather than raising — extraction
   * requires judgement that has no safe non-LLM fallback, so "degrade
   * gracefully" here means "skip, don't guess".
   *
   * Returns a list of short human-readable descriptions of what was saved
   * (for logging/telemetry), e.g. `["location=Austin, TX"]`.
   */
  async updateFromTurn(userText: string, assistantText: string): Promise<string[]> {
    if (!this.loaded) await this.load();
    if (!this.llm || !userText.trim()) return [];

    let raw: string;
    try {
      raw = await this.llm.complete(this.extractionMessages(userText, assistantText));
    } catch {
      return [];
    }

    const extracted = extractJsonObject(raw);
    if (!extracted) return [];
    return this.saveNewFacts(extracted as ExtractedProfile);
  }

  private async saveNewFacts(data: ExtractedProfile): Promise<string[]> {
    const saved: string[] = [];

    for (const fieldName of SCALAR_FIELDS) {
      const value = data[fieldName];
      if (typeof value !== 'string' || !value.trim()) continue;
      const trimmed = value.trim();
      const current = this[fieldName];
      if (current !== null && normalize(current) === normalize(trimmed)) continue;
      await this.setFact(fieldName, trimmed);
      saved.push(`${fieldName}=${trimmed}`);
    }

    const listSource: Record<ListField, string[] | undefined> = {
      interests: data.interests,
      devices: data.devices,
      communicationStyle: data.communication_style,
    };
    for (const fieldName of LIST_FIELDS) {
      const items = listSource[fieldName];
      if (!Array.isArray(items)) continue;
      const existingNorm = new Set(this[fieldName].map(normalize));
      for (const item of items) {
        if (typeof item !== 'string' || !item.trim()) continue;
        const trimmed = item.trim();
        if (existingNorm.has(normalize(trimmed))) continue;
        await this.setFact(fieldName, trimmed);
        existingNorm.add(normalize(trimmed));
        saved.push(`${fieldName}+=${trimmed}`);
      }
    }

    if (Array.isArray(data.people)) {
      for (const entry of data.people) {
        const name = String(entry?.name ?? '').trim();
        const relation = String(entry?.relation ?? '').trim();
        if (!name || !relation) continue;
        const current = this.people[name];
        if (current !== undefined && normalize(current) === normalize(relation)) continue;
        await this.setFact('people', relation, name);
        saved.push(`people[${name}]=${relation}`);
      }
    }

    if (Array.isArray(data.preferences)) {
      for (const entry of data.preferences) {
        const topic = String(entry?.topic ?? '').trim();
        const value = String(entry?.value ?? '').trim();
        if (!topic || !value) continue;
        const current = this.preferences[topic];
        if (current !== undefined && normalize(current) === normalize(value)) continue;
        await this.setFact('preferences', value, topic);
        saved.push(`preferences[${topic}]=${value}`);
      }
    }

    return saved;
  }
}
