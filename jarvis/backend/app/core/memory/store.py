"""Persistent long-term memory store.

`SQLiteMemoryStore` implements the `MemoryStore` protocol from
`docs/CONTRACTS.md` section 4 (`remember`, `recall`, `history`), plus the
extra primitives the rest of the memory subsystem (owned by `memory-profile`)
needs to build on: `forget`, `consolidate`, and `append_turn`.

Recall blends three signals into one ranking score:

    final_score = 0.7 * cosine_similarity + 0.2 * recency + 0.1 * importance

- cosine_similarity: semantic closeness between the query and stored text,
  computed in numpy over the stored float32 embedding.
- recency: 1.0 for the newest memory in the table, decaying linearly toward
  0.0 for the oldest, relative to the current result set.
- importance: the caller-assigned weight (0..1) stored with the memory.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from app.models.schemas import Message, MemoryHit, MemoryKind, Role

from .db import connect, init_db
from .embeddings import Embedder

_RECALL_WEIGHT_COSINE = 0.7
_RECALL_WEIGHT_RECENCY = 0.2
_RECALL_WEIGHT_IMPORTANCE = 0.1


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SQLiteMemoryStore:
    """SQLite-backed implementation of `MemoryStore` (CONTRACTS.md section 4).

    Owns three tables (`memories`, `conversations`, `summaries`) created by
    `db.init_db()`, which this class calls lazily on first use so a bare
    `SQLiteMemoryStore(path)` is cheap and side-effect-free to construct.
    """

    def __init__(self, db_path: str | Path, embedder: Embedder | None = None) -> None:
        self._db_path = str(db_path)
        self._embedder = embedder or Embedder()
        self._ready = False

    async def _ensure_ready(self) -> None:
        if not self._ready:
            await init_db(self._db_path)
            self._ready = True

    # ------------------------------------------------------------------
    # MemoryStore protocol
    # ------------------------------------------------------------------

    async def remember(self, text: str, kind: str, meta: dict[str, Any] | None = None) -> str:
        """Store a new memory and return its generated id.

        `kind` must be a valid `MemoryKind` value (raises `ValueError`
        otherwise — input validation at the boundary). `meta` may optionally
        carry an `"importance"` float in [0, 1]; it defaults to 0.5 and is
        also persisted verbatim inside the `meta` JSON blob.
        """
        await self._ensure_ready()
        parsed_kind = MemoryKind(kind)
        meta = dict(meta or {})
        importance = min(max(float(meta.get("importance", 0.5)), 0.0), 1.0)

        mem_id = str(uuid.uuid4())
        now = time.time()
        vector = await self._embedder.encode_one(text)
        blob = Embedder.to_bytes(vector)

        async with connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO memories
                    (id, text, kind, embedding, created_at, accessed_at,
                     access_count, importance, meta)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    mem_id,
                    text,
                    parsed_kind.value,
                    blob,
                    now,
                    now,
                    importance,
                    json.dumps(meta),
                ),
            )
            await db.commit()
        return mem_id

    async def recall(self, query: str, k: int = 6) -> list[MemoryHit]:
        """Return the top-k memories ranked by the blended recall score.

        Bumps `accessed_at` / `access_count` on every memory returned.
        """
        await self._ensure_ready()

        async with connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, text, kind, embedding, created_at, importance, meta FROM memories"
            )
            rows = await cursor.fetchall()

        if not rows:
            return []

        query_vec = await self._embedder.encode_one(query)
        now = time.time()
        ages = [max(now - row["created_at"], 0.0) for row in rows]
        max_age = max(ages) if ages else 0.0

        scored: list[tuple[float, aiosqlite.Row]] = []
        for row, age in zip(rows, ages, strict=True):
            vec = Embedder.from_bytes(row["embedding"])
            cosine = _cosine_similarity(query_vec, vec)
            recency = 1.0 - (age / max_age) if max_age > 0 else 1.0
            importance = float(row["importance"])
            score = (
                _RECALL_WEIGHT_COSINE * cosine
                + _RECALL_WEIGHT_RECENCY * recency
                + _RECALL_WEIGHT_IMPORTANCE * importance
            )
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: max(k, 0)]

        if top:
            async with connect(self._db_path) as db:
                await db.executemany(
                    "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 "
                    "WHERE id = ?",
                    [(now, row["id"]) for _, row in top],
                )
                await db.commit()

        return [
            MemoryHit(
                id=row["id"],
                text=row["text"],
                kind=MemoryKind(row["kind"]),
                score=score,
                created_at=row["created_at"],
                meta=json.loads(row["meta"] or "{}"),
            )
            for score, row in top
        ]

    async def history(self, limit: int = 20) -> list[Message]:
        """Return the most recent `limit` conversation turns, oldest first."""
        await self._ensure_ready()
        async with connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT role, content FROM conversations ORDER BY created_at DESC LIMIT ?",
                (max(limit, 0),),
            )
            rows = await cursor.fetchall()
        return [Message(role=Role(row["role"]), content=row["content"]) for row in reversed(rows)]

    # ------------------------------------------------------------------
    # Extra primitives used by the rest of the memory subsystem
    # ------------------------------------------------------------------

    async def append_turn(self, role: str, content: str, turn_id: str | None = None) -> str:
        """Append one raw conversation turn and return its generated id."""
        await self._ensure_ready()
        parsed_role = Role(role)
        row_id = str(uuid.uuid4())
        now = time.time()
        async with connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO conversations (id, role, content, turn_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (row_id, parsed_role.value, content, turn_id, now),
            )
            await db.commit()
        return row_id

    async def forget(self, memory_id: str) -> bool:
        """Delete a memory by id. Returns True if a row was actually removed."""
        await self._ensure_ready()
        async with connect(self._db_path) as db:
            cursor = await db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await db.commit()
            return cursor.rowcount > 0

    async def consolidate(self, threshold: float = 0.95) -> int:
        """Dedupe near-identical memories (cosine similarity >= threshold).

        For each pair above the threshold, keeps the one with higher
        `importance` (ties keep the earlier-created row) and deletes the
        other. Returns the number of memories removed.
        """
        await self._ensure_ready()
        async with connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, embedding, importance FROM memories ORDER BY created_at ASC"
            )
            rows = await cursor.fetchall()

        if len(rows) < 2:
            return 0

        vectors = [Embedder.from_bytes(row["embedding"]) for row in rows]
        removed: set[str] = set()

        for i in range(len(rows)):
            if rows[i]["id"] in removed:
                continue
            for j in range(i + 1, len(rows)):
                if rows[j]["id"] in removed:
                    continue
                similarity = _cosine_similarity(vectors[i], vectors[j])
                if similarity < threshold:
                    continue
                imp_i = float(rows[i]["importance"])
                imp_j = float(rows[j]["importance"])
                if imp_j > imp_i:
                    removed.add(rows[i]["id"])
                    break  # rows[i] is gone; nothing left to compare it against
                removed.add(rows[j]["id"])

        if removed:
            async with connect(self._db_path) as db:
                await db.executemany(
                    "DELETE FROM memories WHERE id = ?", [(mid,) for mid in removed]
                )
                await db.commit()

        return len(removed)
