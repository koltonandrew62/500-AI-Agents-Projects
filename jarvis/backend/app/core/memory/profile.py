"""The durable model of who the user is.

Owned by agent `memory-profile`. `UserProfile` maintains a small set of
structured fields (name, location, timezone, occupation, interests, people
the user mentions, devices, stated preferences, communication style) backed
by individual FACT / PREFERENCE memories in the shared memory store (owned by
agent `memory-store`, see `core/memory/store.py`).

This module never imports `store.py` directly -- it depends only on the
structural `_ProfileStore` protocol below (matching the `MemoryStore`
protocol in CONTRACTS.md section 4 plus the `SQLiteMemoryStore.remember`
signature), so it can be built and unit tested independently of the concrete
store implementation landing in parallel.

Persistence convention
-----------------------
Every fact this module writes carries structured `meta`:

    {"profile_field": "<field name>", "value": "<value>", "key": "<optional>"}

`key` is used for the two dict-shaped fields (`people`, `preferences`) to
identify which person / topic the value is about. `load()` trusts this meta
as the source of truth (rather than parsing the human-readable sentence
text) and reconstructs state via targeted `recall()` queries per field --
the store exposes semantic recall, not a list-all-of-kind primitive, so this
is the best reconstruction available under the current contract.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from app.models.schemas import Message, MemoryHit, MemoryKind, Role

logger = logging.getLogger(__name__)

# Soft import: the concrete router class per CONTRACTS.md section 4. Imported
# for documentation / typing convenience only -- `UserProfile` never depends
# on it at runtime, it depends on the structural `_LLMClient` protocol below,
# so this file keeps working even if `core/llm` hasn't landed its package
# exports yet while agents build in parallel.
try:  # pragma: no cover - best-effort convenience import
    from app.core.llm import ModelRouter  # noqa: F401
except Exception:  # pragma: no cover
    ModelRouter = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Structural dependencies
# ---------------------------------------------------------------------------


class _ProfileStore(Protocol):
    """The slice of `SQLiteMemoryStore` this module needs."""

    async def remember(self, text: str, kind: str, meta: dict[str, Any]) -> str: ...

    async def recall(self, query: str, k: int = 6) -> list[MemoryHit]: ...


class _LLMClient(Protocol):
    """The slice of `ModelRouter` / `LLMProvider` this module needs."""

    async def complete(self, messages: list[Message]) -> str: ...


# ---------------------------------------------------------------------------
# Field taxonomy
# ---------------------------------------------------------------------------

_SCALAR_FIELDS = ("name", "location", "timezone", "occupation")
_LIST_FIELDS = ("interests", "devices", "communication_style")
_DICT_FIELDS = ("people", "preferences")

_FIELD_KIND: dict[str, MemoryKind] = {
    "name": MemoryKind.FACT,
    "location": MemoryKind.FACT,
    "timezone": MemoryKind.FACT,
    "occupation": MemoryKind.FACT,
    "interests": MemoryKind.FACT,
    "people": MemoryKind.FACT,
    "devices": MemoryKind.FACT,
    "preferences": MemoryKind.PREFERENCE,
    "communication_style": MemoryKind.PREFERENCE,
}

# Targeted recall query per field, used by load() to reconstruct state.
_FIELD_QUERIES: dict[str, str] = {
    "name": "the user's name",
    "location": "where the user lives, their location",
    "timezone": "the user's timezone",
    "occupation": "the user's job, occupation, profession",
    "interests": "things the user is interested in, hobbies",
    "people": "people the user has mentioned: family, friends, colleagues, pets",
    "devices": "devices the user owns or uses",
    "preferences": "the user's stated preferences, likes and dislikes",
    "communication_style": "the user's preferred communication style or tone",
}

_SENTENCE_TEMPLATES: dict[str, str] = {
    "name": "The user's name is {value}.",
    "location": "The user lives in {value}.",
    "timezone": "The user's timezone is {value}.",
    "occupation": "The user works as {value}.",
    "interests": "The user is interested in {value}.",
    "devices": "The user uses a device: {value}.",
    "people": "The user mentioned {key}, described as: {value}.",
    "preferences": "Regarding {key}, the user prefers: {value}.",
    "communication_style": "The user's communication style: {value}.",
}

_EXTRACTION_SYSTEM_PROMPT = """\
You extract durable, self-reported facts about a user from one turn of a \
conversation with their personal AI assistant.

Rules (follow strictly):
1. Only extract facts the USER explicitly and unambiguously stated about \
THEMSELVES in their own message. Never extract facts about the assistant, \
and never extract something about another person unless the user is \
directly stating their relationship to that person.
2. Never infer or guess. A single ambiguous, hypothetical, or third-hand \
mention ("what if I lived in Texas", "my friend loves hiking") is NOT a \
fact to extract. Only clear, direct, first-person statements are \
("I live in Texas", "I love hiking").
3. Do not repeat anything already present in "Known profile" below -- only \
report facts that are genuinely NEW or that CHANGE a known value.
4. If nothing new and durable was stated, return an empty JSON object: {}
5. Respond with ONLY a single JSON object. No prose, no markdown fences, \
no commentary.

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
}
"""


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _render_fact_sentence(field_name: str, value: str, key: str | None) -> str:
    template = _SENTENCE_TEMPLATES[field_name]
    if key is not None:
        return template.format(value=value, key=key)
    return template.format(value=value)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of the first top-level JSON object in `text`.

    LLMs sometimes wrap JSON in prose or markdown fences despite explicit
    instructions not to; this scans for a balanced `{...}` span (respecting
    string quoting) instead of assuming the whole response is valid JSON.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


# ---------------------------------------------------------------------------
# UserProfile
# ---------------------------------------------------------------------------


class UserProfile:
    """The durable, structured model of who the user is.

    Construct once per session with a store (and, optionally, an LLM client
    for fact extraction) and call `load()` before first use.
    """

    def __init__(self, store: _ProfileStore, llm: _LLMClient | None = None) -> None:
        self._store = store
        self._llm = llm
        self._loaded = False

        self.name: str | None = None
        self.location: str | None = None
        self.timezone: str | None = None
        self.occupation: str | None = None
        self.interests: list[str] = []
        self.people: dict[str, str] = {}
        self.devices: list[str] = []
        self.preferences: dict[str, str] = {}
        self.communication_style: list[str] = []

    # -- loading ------------------------------------------------------

    def _reset_fields(self) -> None:
        self.name = None
        self.location = None
        self.timezone = None
        self.occupation = None
        self.interests = []
        self.people = {}
        self.devices = []
        self.preferences = {}
        self.communication_style = []

    def _apply(self, field_name: str, value: str, key: str | None = None) -> None:
        if field_name in _SCALAR_FIELDS:
            setattr(self, field_name, value)
        elif field_name in _LIST_FIELDS:
            items: list[str] = getattr(self, field_name)
            if _normalize(value) not in {_normalize(v) for v in items}:
                items.append(value)
        elif field_name in _DICT_FIELDS:
            if key:
                getattr(self, field_name)[key] = value

    async def load(self) -> None:
        """(Re)populate profile fields from the memory store.

        The store only exposes semantic `recall()`, not a list-all-of-kind
        primitive, so this issues one targeted query per field and trusts
        the structured `meta` this module writes (`profile_field`, `value`,
        `key`) as ground truth rather than parsing sentence text.
        """
        self._reset_fields()
        # slot -> (created_at, value); keeps the most recent write per slot.
        best: dict[tuple[str, str | None], tuple[float, str]] = {}

        for field_name, query in _FIELD_QUERIES.items():
            try:
                hits = await self._store.recall(query, k=10)
            except Exception as exc:  # provider/store failure -> skip field
                logger.warning(
                    "UserProfile.load(): recall failed for field %r: %s",
                    field_name,
                    exc,
                )
                continue
            for hit in hits:
                meta = hit.meta or {}
                if meta.get("profile_field") != field_name:
                    continue
                value = meta.get("value")
                if not value:
                    continue
                key = meta.get("key")
                slot = (field_name, key)
                prev = best.get(slot)
                if prev is None or hit.created_at > prev[0]:
                    best[slot] = (hit.created_at, str(value))

        for (field_name, key), (_, value) in best.items():
            self._apply(field_name, value, key)
        self._loaded = True

    # -- rendering ------------------------------------------------------

    def render_for_prompt(self) -> str:
        """Compact block for injection into the persona/system prompt."""
        lines: list[str] = []
        if self.name:
            lines.append(f"Name: {self.name}")
        if self.location:
            lines.append(f"Location: {self.location}")
        if self.timezone:
            lines.append(f"Timezone: {self.timezone}")
        if self.occupation:
            lines.append(f"Occupation: {self.occupation}")
        if self.interests:
            lines.append(f"Interests: {', '.join(self.interests)}")
        if self.people:
            people = "; ".join(
                f"{name} ({rel})" if rel else name for name, rel in self.people.items()
            )
            lines.append(f"People: {people}")
        if self.devices:
            lines.append(f"Devices: {', '.join(self.devices)}")
        if self.preferences:
            prefs = "; ".join(f"{topic}: {val}" for topic, val in self.preferences.items())
            lines.append(f"Preferences: {prefs}")
        if self.communication_style:
            lines.append(f"Communication style: {', '.join(self.communication_style)}")
        if not lines:
            return ""
        return "USER PROFILE:\n" + "\n".join(f"- {line}" for line in lines)

    # -- direct fact access ------------------------------------------------------

    async def set_fact(self, field_name: str, value: str, *, key: str | None = None) -> None:
        """Directly persist a single known fact, bypassing LLM extraction.

        Used both internally by `update_from_turn()` and by any caller that
        already knows a fact for certain (e.g. an explicit "remember that
        ..." command handled elsewhere).
        """
        if field_name not in _FIELD_KIND:
            raise ValueError(f"Unknown profile field: {field_name!r}")
        value = value.strip()
        if not value:
            return
        if field_name in _DICT_FIELDS and not key:
            raise ValueError(f"Field {field_name!r} requires a key")

        kind = _FIELD_KIND[field_name]
        text = _render_fact_sentence(field_name, value, key)
        meta: dict[str, Any] = {"profile_field": field_name, "value": value}
        if key:
            meta["key"] = key

        try:
            await self._store.remember(text, kind.value, meta)
        except Exception as exc:
            logger.warning(
                "UserProfile.set_fact(): failed to persist %s=%s: %s",
                field_name,
                value,
                exc,
            )
            return
        self._apply(field_name, value, key)

    def get_fact(self, field_name: str, key: str | None = None) -> Any:
        """Read a fact from the in-memory profile (no store round-trip)."""
        if field_name in _SCALAR_FIELDS:
            return getattr(self, field_name)
        if field_name in _LIST_FIELDS:
            return list(getattr(self, field_name))
        if field_name in _DICT_FIELDS:
            values: dict[str, str] = getattr(self, field_name)
            return values.get(key) if key is not None else dict(values)
        raise ValueError(f"Unknown profile field: {field_name!r}")

    # -- extraction ------------------------------------------------------

    def _extraction_messages(self, user_text: str, assistant_text: str) -> list[Message]:
        known = self.render_for_prompt() or "(nothing known yet)"
        user_prompt = (
            f"Known profile:\n{known}\n\n"
            f"User said: {user_text!r}\n"
            f"Assistant replied: {assistant_text!r}\n\n"
            "Extract only NEW durable facts the user stated about themselves, "
            "per the rules. Respond with the JSON object only."
        )
        return [
            Message(role=Role.SYSTEM, content=_EXTRACTION_SYSTEM_PROMPT),
            Message(role=Role.USER, content=user_prompt),
        ]

    async def update_from_turn(self, user_text: str, assistant_text: str) -> list[str]:
        """Extract and persist NEW durable facts from one conversation turn.

        Conservative by design: only saves things the user actually stated
        about themselves, deduped against the currently loaded profile.
        Never re-saves something already known. If the LLM is unavailable
        or fails, this degrades to a no-op for the turn rather than raising
        -- extraction requires judgement that has no safe non-LLM fallback,
        so "degrade gracefully" here means "skip, don't guess".

        Returns a list of short human-readable descriptions of what was
        saved (for logging/telemetry), e.g. `["location=Austin, TX"]`.
        """
        if not self._loaded:
            await self.load()
        if self._llm is None or not user_text.strip():
            return []

        try:
            raw = await self._llm.complete(self._extraction_messages(user_text, assistant_text))
        except Exception as exc:
            logger.warning(
                "UserProfile.update_from_turn(): LLM extraction failed, skipping: %s",
                exc,
            )
            return []

        extracted = _extract_json_object(raw)
        if not extracted:
            return []
        return await self._save_new_facts(extracted)

    async def _save_new_facts(self, data: dict[str, Any]) -> list[str]:
        saved: list[str] = []

        for field_name in _SCALAR_FIELDS:
            value = data.get(field_name)
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            current = getattr(self, field_name)
            if current is not None and _normalize(current) == _normalize(value):
                continue
            await self.set_fact(field_name, value)
            saved.append(f"{field_name}={value}")

        for field_name in _LIST_FIELDS:
            items = data.get(field_name)
            if not isinstance(items, list):
                continue
            existing_norm = {_normalize(v) for v in getattr(self, field_name)}
            for item in items:
                if not isinstance(item, str) or not item.strip():
                    continue
                item = item.strip()
                if _normalize(item) in existing_norm:
                    continue
                await self.set_fact(field_name, item)
                existing_norm.add(_normalize(item))
                saved.append(f"{field_name}+={item}")

        people = data.get("people")
        if isinstance(people, list):
            for entry in people:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", "")).strip()
                relation = str(entry.get("relation", "")).strip()
                if not name or not relation:
                    continue
                current = self.people.get(name)
                if current is not None and _normalize(current) == _normalize(relation):
                    continue
                await self.set_fact("people", relation, key=name)
                saved.append(f"people[{name}]={relation}")

        preferences = data.get("preferences")
        if isinstance(preferences, list):
            for entry in preferences:
                if not isinstance(entry, dict):
                    continue
                topic = str(entry.get("topic", "")).strip()
                value = str(entry.get("value", "")).strip()
                if not topic or not value:
                    continue
                current = self.preferences.get(topic)
                if current is not None and _normalize(current) == _normalize(value):
                    continue
                await self.set_fact("preferences", value, key=topic)
                saved.append(f"preferences[{topic}]={value}")

        return saved


__all__ = ["UserProfile"]
