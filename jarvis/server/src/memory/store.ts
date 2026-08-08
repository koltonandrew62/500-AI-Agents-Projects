/**
 * Persistent long-term memory store.
 *
 * `SqliteMemoryStore` implements the `MemoryStore` interface from
 * `../types.ts`, plus the extra primitives the rest of the memory subsystem
 * builds on: `forget` and `consolidate` are already part of the shared
 * contract, and `appendTurn` covers raw conversation-turn logging.
 *
 * Recall blends three signals into one ranking score:
 *
 *     finalScore = 0.7 * cosineSimilarity + 0.2 * recency + 0.1 * importance
 *
 * - cosineSimilarity: semantic closeness between the query and stored text,
 *   computed over the stored float32 embedding.
 * - recency: 1.0 for the newest memory in the table, decaying linearly
 *   toward 0.0 for the oldest, relative to the current result set.
 * - importance: the caller-assigned weight (0..1) stored with the memory.
 */

import { randomUUID } from 'node:crypto';
import type Database from 'better-sqlite3';
import type { MemoryHit, MemoryKind, MemoryStore, Message, Role } from '../types.js';
import { initDb } from './db.js';
import { Embedder } from './embeddings.js';

const RECALL_WEIGHT_COSINE = 0.7;
const RECALL_WEIGHT_RECENCY = 0.2;
const RECALL_WEIGHT_IMPORTANCE = 0.1;

const VALID_KINDS: ReadonlySet<MemoryKind> = new Set([
  'fact',
  'event',
  'pref',
  'conv',
  'obs',
]);

const VALID_ROLES: ReadonlySet<Role> = new Set(['system', 'user', 'assistant', 'tool']);

interface MemoryRow {
  id: string;
  text: string;
  kind: string;
  embedding: Buffer;
  created_at: number;
  importance: number;
  meta: string;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i += 1) {
    // len <= a.length and len <= b.length by construction, so both reads
    // are provably in-bounds; TypedArray indexing never yields `undefined`
    // at runtime.
    const ai = a[i] as number;
    const bi = b[i] as number;
    dot += ai * bi;
    normA += ai * ai;
    normB += bi * bi;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/**
 * SQLite-backed implementation of `MemoryStore`. Owns three tables
 * (`memories`, `conversations`, `summaries`) created by `initDb()`, which
 * this class calls lazily on first use so a bare `new SqliteMemoryStore(path)`
 * is cheap and side-effect-free to construct.
 */
export class SqliteMemoryStore implements MemoryStore {
  private readonly dbPath: string;
  private readonly embedder: Embedder;
  private db: Database.Database | null = null;

  constructor(dbPath: string, embedder?: Embedder) {
    this.dbPath = dbPath;
    this.embedder = embedder ?? new Embedder();
  }

  private conn(): Database.Database {
    if (!this.db) {
      this.db = initDb(this.dbPath);
    }
    return this.db;
  }

  /** Close the underlying connection. Safe to call multiple times. */
  close(): void {
    this.db?.close();
    this.db = null;
  }

  // ------------------------------------------------------------------
  // MemoryStore interface
  // ------------------------------------------------------------------

  async remember(
    text: string,
    kind: MemoryKind,
    meta?: Record<string, unknown>,
  ): Promise<string> {
    if (!VALID_KINDS.has(kind)) {
      throw new Error(`Invalid memory kind: ${String(kind)}`);
    }
    const metaObj = { ...(meta ?? {}) };
    const rawImportance = metaObj.importance;
    const importance = Math.min(
      Math.max(typeof rawImportance === 'number' ? rawImportance : 0.5, 0),
      1,
    );

    const id = randomUUID();
    const now = Date.now() / 1000;
    const vector = await this.embedder.encodeOne(text);
    const blob = Embedder.toBytes(vector);

    this.conn()
      .prepare(
        `INSERT INTO memories
           (id, text, kind, embedding, created_at, accessed_at, access_count, importance, meta)
         VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)`,
      )
      .run(id, text, kind, blob, now, now, importance, JSON.stringify(metaObj));

    return id;
  }

  async recall(query: string, k = 6): Promise<MemoryHit[]> {
    const rows = this.conn()
      .prepare(
        'SELECT id, text, kind, embedding, created_at, importance, meta FROM memories',
      )
      .all() as MemoryRow[];

    if (rows.length === 0) return [];

    const queryVec = await this.embedder.encodeOne(query);
    const now = Date.now() / 1000;
    const ages = rows.map((row) => Math.max(now - row.created_at, 0));
    const maxAge = ages.length > 0 ? Math.max(...ages) : 0;

    const scored = rows.map((row, i) => {
      const vec = Embedder.fromBytes(row.embedding);
      const cosine = cosineSimilarity(queryVec, vec);
      // ages is built via rows.map(...), so it has exactly rows.length
      // entries and `i` (from this same rows.map) is always in range.
      const age = ages[i] as number;
      const recency = maxAge > 0 ? 1 - age / maxAge : 1;
      const importance = row.importance;
      const score =
        RECALL_WEIGHT_COSINE * cosine +
        RECALL_WEIGHT_RECENCY * recency +
        RECALL_WEIGHT_IMPORTANCE * importance;
      return { score, row };
    });

    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, Math.max(k, 0));

    if (top.length > 0) {
      const bump = this.conn().prepare(
        'UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?',
      );
      const bumpAll = this.conn().transaction((items: typeof top) => {
        for (const { row } of items) bump.run(now, row.id);
      });
      bumpAll(top);
    }

    return top.map(({ score, row }) => ({
      id: row.id,
      text: row.text,
      kind: row.kind as MemoryKind,
      score,
      createdAt: row.created_at,
      meta: JSON.parse(row.meta || '{}') as Record<string, unknown>,
    }));
  }

  async history(limit = 20): Promise<Message[]> {
    const rows = this.conn()
      .prepare(
        'SELECT role, content FROM conversations ORDER BY created_at DESC LIMIT ?',
      )
      .all(Math.max(limit, 0)) as { role: string; content: string }[];

    return rows.reverse().map((row) => ({
      role: row.role as Role,
      content: row.content,
    }));
  }

  // ------------------------------------------------------------------
  // Extra primitives
  // ------------------------------------------------------------------

  async appendTurn(role: Role, content: string, turnId?: string): Promise<string> {
    if (!VALID_ROLES.has(role)) {
      throw new Error(`Invalid role: ${String(role)}`);
    }
    const id = randomUUID();
    const now = Date.now() / 1000;
    this.conn()
      .prepare(
        'INSERT INTO conversations (id, role, content, turn_id, created_at) VALUES (?, ?, ?, ?, ?)',
      )
      .run(id, role, content, turnId ?? null, now);
    return id;
  }

  async forget(id: string): Promise<boolean> {
    const result = this.conn().prepare('DELETE FROM memories WHERE id = ?').run(id);
    return result.changes > 0;
  }

  async consolidate(threshold = 0.95): Promise<number> {
    const rows = this.conn()
      .prepare('SELECT id, embedding, importance FROM memories ORDER BY created_at ASC')
      .all() as { id: string; embedding: Buffer; importance: number }[];

    if (rows.length < 2) return 0;

    const vectors = rows.map((row) => Embedder.fromBytes(row.embedding));
    const removed = new Set<string>();

    for (let i = 0; i < rows.length; i += 1) {
      const rowI = rows[i];
      const vecI = vectors[i];
      // vectors is rows.map(...), so it's always the same length as rows —
      // this pair is unreachable-undefined, guarded rather than asserted.
      if (rowI === undefined || vecI === undefined) continue;
      if (removed.has(rowI.id)) continue;
      for (let j = i + 1; j < rows.length; j += 1) {
        const rowJ = rows[j];
        const vecJ = vectors[j];
        if (rowJ === undefined || vecJ === undefined) continue;
        if (removed.has(rowJ.id)) continue;
        const similarity = cosineSimilarity(vecI, vecJ);
        if (similarity < threshold) continue;
        const impI = rowI.importance;
        const impJ = rowJ.importance;
        if (impJ > impI) {
          removed.add(rowI.id);
          break; // rowI is gone; nothing left to compare it against
        }
        removed.add(rowJ.id);
      }
    }

    if (removed.size > 0) {
      const del = this.conn().prepare('DELETE FROM memories WHERE id = ?');
      const delAll = this.conn().transaction((ids: string[]) => {
        for (const id of ids) del.run(id);
      });
      delAll([...removed]);
    }

    return removed.size;
  }
}
