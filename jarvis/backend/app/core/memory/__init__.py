"""Persistent long-term memory subsystem for J.A.R.V.I.S.

Public surface: `SQLiteMemoryStore` (implements the `MemoryStore` protocol
from `docs/CONTRACTS.md` section 4), `Embedder`, and `init_db`.

`profile.py` and `recall.py` (owned by the `memory-profile` agent) build on
top of `SQLiteMemoryStore` — they are not exported from here.
"""

from __future__ import annotations

from .db import init_db
from .embeddings import Embedder
from .store import SQLiteMemoryStore

__all__ = ["SQLiteMemoryStore", "Embedder", "init_db"]
