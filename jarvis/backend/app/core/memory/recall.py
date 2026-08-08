"""Smarter retrieval than raw vector search.

A single embedding of the user's literal message is a weak query: people ask
"what was that thing I mentioned about the camera" when the stored memory says
"prefers the webcam feed muted on startup". `RecallEngine` widens the net by
rewriting one message into several retrieval queries, running them all, and
fusing the rankings.

Every LLM call here is optional. If the provider is down, rate-limited, or has
no key, each stage degrades to a deterministic heuristic rather than failing
the turn — recall gets fuzzier, the assistant keeps working.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Sequence

from app.models.schemas import MemoryHit, Message, Role

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.core.memory.store import SQLiteMemoryStore

logger = logging.getLogger(__name__)

# Reciprocal-rank-fusion damping. 60 is the value from the original RRF paper;
# it keeps a rank-1 hit from a single query from dominating a result that
# placed respectably across all of them.
RRF_K = 60

MAX_EXPANSIONS = 3

# Turns that never justify a retrieval round-trip.
_TRIVIAL = frozenset(
    {
        "hi", "hey", "hello", "yo", "sup", "thanks", "thank you", "ty", "ok",
        "okay", "k", "cool", "nice", "got it", "sure", "yes", "no", "yep",
        "nope", "stop", "cancel", "nevermind", "never mind", "quiet", "shut up",
        "wake up", "jarvis", "hey jarvis", "good morning", "good night", "bye",
    }
)

_EXPANSION_PROMPT = """Rewrite the user's message into up to {n} short search \
queries for a personal memory database. Each query should capture a different \
angle: literal keywords, the underlying topic, and any implied subject.

Return ONLY the queries, one per line, no numbering, no commentary.

User message: {text}"""


class RecallEngine:
    """Multi-query retrieval with reciprocal-rank fusion over a memory store."""

    def __init__(
        self,
        store: "SQLiteMemoryStore",
        router: Any | None = None,
        *,
        max_expansions: int = MAX_EXPANSIONS,
    ) -> None:
        self._store = store
        self._router = router
        self._max_expansions = max(1, max_expansions)

    # -- gating ----------------------------------------------------------

    @staticmethod
    def should_recall(text: str) -> bool:
        """Fast heuristic: is this turn worth a retrieval round-trip?

        Deliberately cheap and conservative — when unsure, recall. The cost of
        a needless lookup is milliseconds; the cost of missing context is the
        assistant forgetting who it is talking to.
        """
        cleaned = re.sub(r"[^\w\s]", "", (text or "").strip().lower())
        if not cleaned:
            return False
        if cleaned in _TRIVIAL:
            return False
        # Very short utterances carry too little signal to retrieve against,
        # unless they are a question ("why?", "when?") that leans on context.
        if len(cleaned.split()) <= 2 and not (text or "").strip().endswith("?"):
            return False
        return True

    # -- query expansion -------------------------------------------------

    async def expand(self, query: str) -> list[str]:
        """Rewrite one message into several retrieval queries."""
        queries = [query]
        if self._router is None:
            return queries + _heuristic_expansions(query, self._max_expansions - 1)

        prompt = _EXPANSION_PROMPT.format(n=self._max_expansions, text=query)
        try:
            routed = await self._router.complete_chat(
                [Message(role=Role.USER, content=prompt)]
            )
            raw = getattr(routed, "text", None) or str(routed)
        except Exception as exc:  # provider down / rate limited / no key
            logger.debug("query expansion unavailable, using heuristics: %s", exc)
            return queries + _heuristic_expansions(query, self._max_expansions - 1)

        for line in (raw or "").splitlines():
            candidate = line.strip().lstrip("-•*0123456789. ").strip()
            if not candidate or len(candidate) < 3:
                continue
            if candidate.lower() in {q.lower() for q in queries}:
                continue
            queries.append(candidate)
            if len(queries) >= self._max_expansions + 1:
                break

        if len(queries) == 1:
            queries += _heuristic_expansions(query, self._max_expansions - 1)
        return queries

    # -- retrieval -------------------------------------------------------

    async def recall(self, query: str, k: int = 6) -> list[MemoryHit]:
        """Multi-query recall fused by reciprocal rank."""
        if not self.should_recall(query):
            return []

        queries = await self.expand(query)

        # Over-fetch per query so fusion has material to work with.
        per_query = max(k, 8)
        rankings: list[list[MemoryHit]] = []
        for q in queries:
            try:
                rankings.append(await self._store.recall(q, k=per_query))
            except Exception as exc:
                logger.warning("recall failed for %r: %s", q, exc)

        return _fuse(rankings, k)

    async def build_context(self, query: str, k: int = 6) -> str:
        """Render the winning memories as a compact prompt block."""
        return render_memories(await self.recall(query, k))


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _fuse(rankings: Sequence[Sequence[MemoryHit]], k: int) -> list[MemoryHit]:
    """Reciprocal-rank fusion, deduplicated by memory id.

    Scoring by rank rather than raw similarity is what makes this robust: the
    stores' cosine scores are not comparable across differently-phrased
    queries, but positions are.
    """
    scores: dict[str, float] = {}
    best: dict[str, MemoryHit] = {}

    for ranking in rankings:
        for position, hit in enumerate(ranking):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + position + 1)
            # Keep the representation that scored highest on its own query.
            if hit.id not in best or hit.score > best[hit.id].score:
                best[hit.id] = hit

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    fused: list[MemoryHit] = []
    seen_text: set[str] = set()
    for memory_id, _ in ordered:
        hit = best[memory_id]
        # Near-duplicate text can survive as distinct rows; collapse it here so
        # the prompt block does not repeat itself.
        fingerprint = re.sub(r"\W+", "", hit.text.lower())[:120]
        if fingerprint in seen_text:
            continue
        seen_text.add(fingerprint)
        fused.append(hit)
        if len(fused) >= k:
            break
    return fused


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_memories(hits: Sequence[MemoryHit], *, now: float | None = None) -> str:
    """Render memories as a compact prompt block with relative timestamps."""
    if not hits:
        return ""
    current = now if now is not None else time.time()
    lines = [
        f"- [{relative_time(hit.created_at, current)}] {hit.text.strip()}"
        for hit in hits
        if hit.text and hit.text.strip()
    ]
    if not lines:
        return ""
    return "RELEVANT MEMORY:\n" + "\n".join(lines)


def relative_time(then: float, now: float | None = None) -> str:
    """Human-readable age: 'just now', 'yesterday', '3 weeks ago'."""
    current = now if now is not None else time.time()
    delta = max(0.0, current - then)

    minutes = delta / 60
    if minutes < 2:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)} minutes ago"

    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} hour{'s' if int(hours) != 1 else ''} ago"

    days = hours / 24
    if days < 2:
        return "yesterday"
    if days < 7:
        return f"{int(days)} days ago"

    weeks = days / 7
    if weeks < 5:
        return f"{int(weeks)} week{'s' if int(weeks) != 1 else ''} ago"

    months = days / 30
    if months < 12:
        return f"{int(months)} month{'s' if int(months) != 1 else ''} ago"

    return f"{int(days / 365)} year{'s' if int(days / 365) != 1 else ''} ago"


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were",
        "do", "does", "did", "have", "has", "had", "i", "you", "me", "my",
        "your", "it", "that", "this", "what", "when", "where", "who", "how",
        "to", "of", "in", "on", "for", "with", "about", "can", "could", "would",
        "should", "please", "jarvis", "tell", "again",
    }
)


def _heuristic_expansions(query: str, limit: int) -> list[str]:
    """Keyword-only expansion for when the LLM is unavailable."""
    if limit <= 0:
        return []

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", query.lower())
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    if not keywords:
        return []

    out: list[str] = []
    # The content words alone — drops question scaffolding that dilutes the vector.
    joined = " ".join(dict.fromkeys(keywords))
    if joined and joined != query.lower():
        out.append(joined)
    # The longest single term, as a narrow high-precision probe.
    if len(keywords) > 1:
        out.append(max(keywords, key=len))

    return out[:limit]
