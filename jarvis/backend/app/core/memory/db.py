"""SQLite schema and connection management for the memory subsystem.

Uses `aiosqlite` for fully-async access. `init_db()` is idempotent — it is
safe (and expected) to call on every process start; it only ever creates
tables/indexes that do not already exist, it never drops or rewrites data.

Persistence correctness matters more than raw speed here, so every write
goes through WAL journaling and an explicit commit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

SCHEMA_VERSION = 1

_CREATE_SCHEMA_META = """
CREATE TABLE IF NOT EXISTS _schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CREATE_MEMORIES = """
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
);
"""

_CREATE_CONVERSATIONS = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    turn_id     TEXT,
    created_at  REAL NOT NULL
);
"""

_CREATE_SUMMARIES = """
CREATE TABLE IF NOT EXISTS summaries (
    id          TEXT PRIMARY KEY,
    period      TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""

_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);",
    "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_conversations_turn_id ON conversations(turn_id);",
    "CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_summaries_period ON summaries(period);",
)


@asynccontextmanager
async def connect(db_path: str | Path) -> AsyncIterator[aiosqlite.Connection]:
    """Open a short-lived connection with sane pragmas, always closed on exit.

    Each call opens and closes its own connection rather than sharing one
    long-lived handle across the process — simpler correctness story for an
    assistant that must survive restarts and doesn't need extreme throughput.
    """
    db = await aiosqlite.connect(str(db_path))
    try:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA foreign_keys=ON;")
        await db.execute("PRAGMA busy_timeout=5000;")
        yield db
    finally:
        await db.close()


async def init_db(db_path: str | Path) -> None:
    """Create the database file, tables, and indexes if they don't exist yet.

    Idempotent: calling this on an already-initialized database is a no-op
    beyond a handful of `CREATE ... IF NOT EXISTS` statements.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    async with connect(path) as db:
        await db.execute(_CREATE_SCHEMA_META)
        await db.execute(_CREATE_MEMORIES)
        await db.execute(_CREATE_CONVERSATIONS)
        await db.execute(_CREATE_SUMMARIES)
        for statement in _INDEXES:
            await db.execute(statement)
        await db.execute(
            "INSERT OR IGNORE INTO _schema_meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        await db.commit()
