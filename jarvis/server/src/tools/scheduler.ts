/**
 * Reminder scheduling: create_reminder, list_reminders, cancel_reminder,
 * plus `runScheduler(onFire)` — a long-lived ticker that fires due
 * reminders.
 *
 * Reminders persist to SQLite at `config.tasksDbPath` so they survive a
 * process restart. `better-sqlite3` is synchronous, so no locking helper is
 * needed the way the Python edition needs one around its async sqlite3
 * connection.
 */

import Database from 'better-sqlite3';
import { randomUUID } from 'node:crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';

import { config } from '../config.js';
import { registry } from './base.js';
import type { Tool, ToolResult } from '../types.js';

const TICK_INTERVAL_MS = 5_000;

interface ReminderRow {
  id: string;
  text: string;
  due_at: number;
  created_at: number;
  fired: number;
  cancelled: number;
}

let db: Database.Database | null = null;

function getDb(): Database.Database {
  if (db) return db;
  db = new Database(config.tasksDbPath);
  db.pragma('journal_mode = WAL');
  db.exec(`
    CREATE TABLE IF NOT EXISTS reminders (
      id TEXT PRIMARY KEY,
      text TEXT NOT NULL,
      due_at REAL NOT NULL,
      created_at REAL NOT NULL,
      fired INTEGER NOT NULL DEFAULT 0,
      cancelled INTEGER NOT NULL DEFAULT 0
    )
  `);
  return db;
}

async function ensureDbDir(): Promise<void> {
  await fs.mkdir(path.dirname(config.tasksDbPath), { recursive: true });
}

function fail(summary: string): ToolResult {
  return { ok: false, output: '', summary };
}

const createReminderTool: Tool = {
  name: 'create_reminder',
  description: 'Schedule a reminder to fire after a given number of seconds from now.',
  parameters: {
    type: 'object',
    properties: {
      text: { type: 'string', minLength: 1, description: 'Reminder text.' },
      delay_seconds: {
        type: 'number',
        minimum: 1,
        description: 'How many seconds from now the reminder should fire.',
      },
    },
    required: ['text', 'delay_seconds'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const text = args.text as string;
    const delaySeconds = args.delay_seconds as number;

    await ensureDbDir();
    const id = randomUUID();
    const now = Date.now() / 1000;
    const dueAt = now + delaySeconds;

    getDb()
      .prepare(
        'INSERT INTO reminders (id, text, due_at, created_at, fired, cancelled) VALUES (?, ?, ?, ?, 0, 0)',
      )
      .run(id, text, dueAt, now);

    return {
      ok: true,
      output: id,
      summary: `Reminder ${id} scheduled in ${Math.round(delaySeconds)}s`,
      meta: { id, due_at: dueAt },
    };
  },
};

const listRemindersTool: Tool = {
  name: 'list_reminders',
  description: 'List pending (not yet fired or cancelled) reminders.',
  parameters: { type: 'object', properties: {}, required: [], additionalProperties: false },
  async run(): Promise<ToolResult> {
    await ensureDbDir();
    const rows = getDb()
      .prepare('SELECT id, text, due_at FROM reminders WHERE fired = 0 AND cancelled = 0 ORDER BY due_at ASC')
      .all() as Array<Pick<ReminderRow, 'id' | 'text' | 'due_at'>>;

    const lines = rows.map((r) => `${r.id}: ${JSON.stringify(r.text)} due at ${Math.round(r.due_at)}`);
    return {
      ok: true,
      output: lines.join('\n'),
      summary: `${rows.length} pending reminder(s)`,
      meta: { count: rows.length },
    };
  },
};

const cancelReminderTool: Tool = {
  name: 'cancel_reminder',
  description: 'Cancel a pending reminder by id.',
  parameters: {
    type: 'object',
    properties: { id: { type: 'string', minLength: 1 } },
    required: ['id'],
    additionalProperties: false,
  },
  async run(args): Promise<ToolResult> {
    const id = args.id as string;
    await ensureDbDir();
    const info = getDb()
      .prepare('UPDATE reminders SET cancelled = 1 WHERE id = ? AND fired = 0 AND cancelled = 0')
      .run(id);

    if (info.changes === 0) {
      return fail(`No pending reminder with id ${JSON.stringify(id)}`);
    }
    return { ok: true, output: '', summary: `Cancelled reminder ${JSON.stringify(id)}` };
  },
};

registry.register(createReminderTool);
registry.register(listRemindersTool);
registry.register(cancelReminderTool);

/**
 * Poll SQLite forever, calling `onFire(id, text)` for each due reminder.
 *
 * Intended to run as a long-lived background task started at server
 * startup. Never throws out of the loop — a single tick's failure is
 * logged and the ticker keeps going so one bad reminder can't kill
 * scheduling entirely. Returns a stop function.
 */
export function runScheduler(
  onFire: (id: string, text: string) => Promise<void>,
  intervalMs: number = TICK_INTERVAL_MS,
): () => void {
  let stopped = false;

  const tick = async (): Promise<void> => {
    if (stopped) return;
    try {
      await ensureDbDir();
      const due = popDueReminders();
      for (const { id, text } of due) {
        try {
          await onFire(id, text);
        } catch (exc) {
          console.error(`[scheduler] onFire callback failed for reminder ${id}:`, exc);
        }
      }
    } catch (exc) {
      console.error('[scheduler] tick failed:', exc);
    }
  };

  const timer = setInterval(() => {
    void tick();
  }, intervalMs);
  // Don't hold the process open just for the scheduler in short-lived contexts.
  timer.unref?.();

  return () => {
    stopped = true;
    clearInterval(timer);
  };
}

function popDueReminders(): Array<{ id: string; text: string }> {
  const now = Date.now() / 1000;
  const database = getDb();
  const rows = database
    .prepare('SELECT id, text FROM reminders WHERE fired = 0 AND cancelled = 0 AND due_at <= ?')
    .all(now) as Array<{ id: string; text: string }>;

  if (rows.length > 0) {
    const markFired = database.prepare('UPDATE reminders SET fired = 1 WHERE id = ?');
    const markAll = database.transaction((ids: string[]) => {
      for (const id of ids) markFired.run(id);
    });
    markAll(rows.map((r) => r.id));
  }
  return rows;
}
