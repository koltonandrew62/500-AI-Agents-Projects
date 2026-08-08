"""Reminder scheduling: create_reminder, list_reminders, cancel_reminder,
plus an async `run_scheduler(on_fire)` ticker that fires due reminders.

Reminders persist to SQLite at `settings.tasks_db_path` so they survive a
process restart. All DB access goes through a tiny helper module-level
connection guarded by a lock since sqlite3 connections aren't safely shared
across concurrent awaits without one.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from app.config import settings
from app.core.tools.base import register
from app.models.schemas import ToolResult

logger = logging.getLogger("jarvis.tools.scheduler")

_TICK_INTERVAL_S = 5.0
_db_lock = asyncio.Lock()


def _connect() -> sqlite3.Connection:
    settings.tasks_db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.tasks_db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reminders (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            due_at REAL NOT NULL,
            created_at REAL NOT NULL,
            fired INTEGER NOT NULL DEFAULT 0,
            cancelled INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


async def _run_db(fn: Callable[[sqlite3.Connection], Any]) -> Any:
    """Run a synchronous sqlite operation off the event loop, serialized."""
    async with _db_lock:
        conn = _connect()
        try:
            return await asyncio.to_thread(_with_conn, conn, fn)
        finally:
            conn.close()


def _with_conn(conn: sqlite3.Connection, fn: Callable[[sqlite3.Connection], Any]) -> Any:
    result = fn(conn)
    conn.commit()
    return result


@register
class CreateReminderTool:
    name = "create_reminder"
    description = "Schedule a reminder to fire after a given number of seconds from now."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1, "description": "Reminder text."},
            "delay_seconds": {
                "type": "number",
                "minimum": 1,
                "description": "How many seconds from now the reminder should fire.",
            },
        },
        "required": ["text", "delay_seconds"],
        "additionalProperties": False,
    }

    async def run(self, text: str, delay_seconds: float) -> ToolResult:
        reminder_id = str(uuid.uuid4())
        now = time.time()
        due_at = now + float(delay_seconds)

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO reminders (id, text, due_at, created_at, fired, cancelled) "
                "VALUES (?, ?, ?, ?, 0, 0)",
                (reminder_id, text, due_at, now),
            )

        await _run_db(_insert)
        return ToolResult(
            ok=True,
            output=reminder_id,
            summary=f"Reminder {reminder_id} scheduled in {delay_seconds:.0f}s",
            meta={"id": reminder_id, "due_at": due_at},
        )


@register
class ListRemindersTool:
    name = "list_reminders"
    description = "List pending (not yet fired or cancelled) reminders."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    async def run(self) -> ToolResult:
        def _select(conn: sqlite3.Connection) -> list[tuple[str, str, float]]:
            cur = conn.execute(
                "SELECT id, text, due_at FROM reminders WHERE fired = 0 AND cancelled = 0 ORDER BY due_at ASC"
            )
            return cur.fetchall()

        rows = await _run_db(_select)
        lines = [f"{rid}: {text!r} due at {due_at:.0f}" for rid, text, due_at in rows]
        return ToolResult(
            ok=True,
            output="\n".join(lines),
            summary=f"{len(rows)} pending reminder(s)",
            meta={"count": len(rows)},
        )


@register
class CancelReminderTool:
    name = "cancel_reminder"
    description = "Cancel a pending reminder by id."
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"id": {"type": "string", "minLength": 1}},
        "required": ["id"],
        "additionalProperties": False,
    }

    async def run(self, id: str) -> ToolResult:
        def _cancel(conn: sqlite3.Connection) -> int:
            cur = conn.execute(
                "UPDATE reminders SET cancelled = 1 WHERE id = ? AND fired = 0 AND cancelled = 0", (id,)
            )
            return cur.rowcount

        rowcount = await _run_db(_cancel)
        if rowcount == 0:
            return ToolResult(ok=False, output="", summary=f"No pending reminder with id {id!r}")
        return ToolResult(ok=True, output="", summary=f"Cancelled reminder {id!r}")


async def run_scheduler(
    on_fire: Callable[[str, str], Awaitable[None]],
    *,
    interval_s: float = _TICK_INTERVAL_S,
) -> None:
    """Poll SQLite forever, calling `on_fire(id, text)` for each due reminder.

    Intended to run as a long-lived background task started at app startup.
    Never raises out of the loop — a single tick's failure is logged and the
    ticker keeps going so one bad reminder can't kill scheduling entirely.
    """
    while True:
        try:
            due = await _pop_due_reminders()
            for reminder_id, text in due:
                try:
                    await on_fire(reminder_id, text)
                except Exception:  # noqa: BLE001 - one bad callback must not kill the ticker
                    logger.exception("on_fire callback failed for reminder %s", reminder_id)
        except Exception:  # noqa: BLE001 - scheduler loop must never die
            logger.exception("scheduler tick failed")
        await asyncio.sleep(interval_s)


async def _pop_due_reminders() -> list[tuple[str, str]]:
    now = time.time()

    def _select_and_mark(conn: sqlite3.Connection) -> list[tuple[str, str]]:
        cur = conn.execute(
            "SELECT id, text FROM reminders WHERE fired = 0 AND cancelled = 0 AND due_at <= ?", (now,)
        )
        rows = cur.fetchall()
        if rows:
            ids = [r[0] for r in rows]
            conn.executemany("UPDATE reminders SET fired = 1 WHERE id = ?", [(i,) for i in ids])
        return rows

    return await _run_db(_select_and_mark)
