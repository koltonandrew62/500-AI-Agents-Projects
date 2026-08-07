"""Text embedding backend for the memory subsystem.

`Embedder` wraps sentence-transformers' `all-MiniLM-L6-v2` (384-dim) but
never imports it at module load time — `sentence-transformers` pulls in
torch (~2GB) and can take real time to import, so the model is loaded lazily
on first use. If `sentence-transformers` isn't installed (or fails to load
for any reason), encoding transparently falls back to a deterministic
hash-based embedder of the same dimensionality: recall still works, just
without real semantic understanding, and the system never crashes for lack
of the optional dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

import numpy as np

EMBED_DIM = 384

_WORD_RE = re.compile(r"[a-z0-9]+")


def _hash_embed(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic, dependency-free bag-of-words embedding.

    Every distinct word hashes to the same fixed pseudo-random unit-ish
    vector (seeded from a SHA-256 digest of the word), and a text's
    embedding is the L2-normalized sum of its words' vectors. Same input
    text always produces the same output vector, across processes and
    restarts, without needing any ML model.
    """
    vec = np.zeros(dim, dtype=np.float64)
    words = _WORD_RE.findall(text.lower()) or [""]
    for word in words:
        digest = hashlib.sha256(word.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vec += rng.normal(size=dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


class Embedder:
    """Lazily-loaded text embedder with a deterministic offline fallback.

    Safe to instantiate at import time / app startup — no model is loaded
    and no heavyweight import happens until `encode()` is first awaited.
    """

    DIM = EMBED_DIM

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._backend: str | None = None
        self._lock = asyncio.Lock()

    @property
    def backend(self) -> str | None:
        """Which backend is active: 'sentence-transformers', 'hash', or None if not yet loaded."""
        return self._backend

    async def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        async with self._lock:
            if self._backend is not None:
                return
            await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._backend = "sentence-transformers"
        except Exception:
            # Missing dependency, no internet to fetch weights, out of
            # memory, etc. — degrade gracefully rather than crash the
            # assistant's memory subsystem.
            self._model = None
            self._backend = "hash"

    def _encode_sync(self, texts: list[str]) -> list[np.ndarray]:
        if self._backend == "sentence-transformers" and self._model is not None:
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [np.asarray(v, dtype=np.float32) for v in vectors]
        return [_hash_embed(t) for t in texts]

    async def encode(self, texts: list[str]) -> list[np.ndarray]:
        """Batch-encode texts into 384-dim float32 vectors, in input order."""
        if not texts:
            return []
        await self._ensure_loaded()
        return await asyncio.to_thread(self._encode_sync, texts)

    async def encode_one(self, text: str) -> np.ndarray:
        """Convenience wrapper for encoding a single string."""
        vectors = await self.encode([text])
        return vectors[0]

    @staticmethod
    def to_bytes(vec: np.ndarray) -> bytes:
        """Serialize an embedding vector to raw float32 bytes for BLOB storage."""
        return np.asarray(vec, dtype=np.float32).tobytes()

    @staticmethod
    def from_bytes(data: bytes) -> np.ndarray:
        """Deserialize a BLOB column back into a float32 numpy vector."""
        return np.frombuffer(data, dtype=np.float32).copy()
