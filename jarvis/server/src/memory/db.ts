/**
 * SQLite schema and connection management for the memory subsystem.
 *
 * `better-sqlite3` is synchronous, so — unlike the Python edition's
 * open-per-call `aiosqlite` connections — this module keeps one long-lived
 * connection per process and hands it out via `initDb()`. `initDb()` is
 * idempotent: safe (and expected) to call on every process start, it only
 * ever creates tables/indexes that do not already exist and never drops or
 * rewrites data.
 *
 * Persistence correctness matters more than raw speed here, so every
 * connection runs WAL journaling with a generous busy timeout so concurrent
 * readers/writers back off instead of throwing `SQLITE_BUSY`.
 */

import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import Database from 'better-sqlite3';

export const SCHEMA_VERSION = 1;

const CREATE_SCHEMA_META = `
CREATE TABLE IF NOT EXISTS _schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);`;

const CREATE_MEMORIES = `
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    kind          TEXT NOT NULL,
    embedding     BLOB,
    created_at    REAL NOT NULL,
    accessed_at   REAL NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    importance    REAL NOT NULL DEFAULT 0.5,
    meta          TEXT NOT NULL DEFAULT '{}'
);`;

const CREATE_CONVERSATIONS = `
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    turn_id     TEXT,
    created_at  REAL NOT NULL
);`;

const CREATE_SUMMARIES = `
CREATE TABLE IF NOT EXISTS summaries (
    id          TEXT PRIMARY KEY,
    period      TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  REAL NOT NULL
);`;

const INDEXES: readonly string[] = [
  'CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);',
  'CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);',
  'CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);',
  'CREATE INDEX IF NOT EXISTS idx_conversations_turn_id ON conversations(turn_id);',
  'CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at);',
  'CREATE INDEX IF NOT EXISTS idx_summaries_period ON summaries(period);',
];

/**
 * Open (creating if necessary) the memory database at `dbPath`, apply
 * pragmas, and ensure the schema exists. Calling this repeatedly on the
 * same path is cheap and safe — every statement is `IF NOT EXISTS`.
 */
export function initDb(dbPath: string): Database.Database {
  const dir = dirname(dbPath);
  if (dir && dir !== '.' && dir !== '/') {
    mkdirSync(dir, { recursive: true });
  }

  const db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
  db.pragma('busy_timeout = 5000');

  db.exec(CREATE_SCHEMA_META);
  db.exec(CREATE_MEMORIES);
  db.exec(CREATE_CONVERSATIONS);
  db.exec(CREATE_SUMMARIES);
  for (const statement of INDEXES) {
    db.exec(statement);
  }
  db.prepare(
    "INSERT OR IGNORE INTO _schema_meta (key, value) VALUES ('schema_version', ?)",
  ).run(String(SCHEMA_VERSION));

  return db;
}
