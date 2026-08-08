"""Turn classification: does this request need the planner?

Planning costs a full round-trip to a free-tier reasoning model, which is the
slowest thing in the turn. The overwhelming majority of turns are conversation
and must never pay for it. So: a pure-heuristic fast path decides confidently in
microseconds at both ends of the spectrum, and only the genuinely ambiguous
middle band escalates to a one-token LLM judgement.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.models.schemas import Message, Role, TurnContext

from app.core.agent.planner import complete_with_model

__all__ = [
    "Classification",
    "TurnClassifier",
    "classify_heuristic",
    "needs_vision",
]

log = logging.getLogger("jarvis.agent.classifier")

Source = Literal["heuristic", "llm", "default"]

_CLASSIFIER_TIMEOUT_S = 12.0
_AMBIGUOUS_LOW = 1
_AMBIGUOUS_HIGH = 3


@dataclass(frozen=True, slots=True)
class Classification:
    """Verdict for one turn."""

    multi_step: bool
    score: int
    reason: str
    source: Source = "heuristic"

    @property
    def needs_planner(self) -> bool:
        """Alias reading naturally at the call site in the agent loop."""
        return self.multi_step


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Turns that are unambiguously conversation, whatever else they contain.
_CHITCHAT = {
    "hi", "hey", "hello", "yo", "sup", "morning", "good morning", "good evening",
    "good night", "thanks", "thank you", "ta", "cheers", "ok", "okay", "k", "cool",
    "nice", "great", "lol", "hmm", "yes", "no", "yep", "nope", "sure", "stop",
    "cancel", "never mind", "nevermind", "wake up", "you there", "you awake",
    "jarvis", "hey jarvis", "hello jarvis", "shut up", "quiet", "bye", "goodbye",
}

# Explicit sequencing language — the strongest multi-step signal there is.
_SEQUENCE_PATTERNS = (
    r"\bstep[- ]by[- ]step\b",
    r"\bfirst\b[^.?!]{0,80}\bthen\b",
    r"\band then\b",
    r"\bafter (?:that|which|you)\b",
    r"\bonce (?:you|that|it)(?:'ve| have| is| are)?\b[^.?!]{0,40}\b(?:then|next)\b",
    r"\bfollowed by\b",
    r"\bfinally,\b",
    r"\bnext,?\s+(?:you|please)?\s*\w+",
    r"\bfor each\b",
    r"\bone by one\b",
    r"\bmake (?:me )?a plan\b",
    r"\bplan (?:out|this|a)\b",
    r"\bwork (?:out|through)\b",
    r"\bbreak (?:it|this) down\b",
    r"\bin order to\b[^.?!]{0,60}\bthen\b",
)

# Verbs that imply reaching outside the model — each one is roughly one tool call.
_ACTION_VERBS = {
    "analyse", "analyze", "audit", "backup", "benchmark", "build", "check",
    "clean", "compare", "compile", "configure", "convert", "copy", "create",
    "debug", "delete", "deploy", "diagnose", "download", "edit", "execute",
    "export", "fetch", "find", "fix", "generate", "grep", "implement", "index",
    "inspect", "install", "kill", "launch", "list", "measure", "migrate",
    "monitor", "move", "open", "optimise", "optimize", "parse", "patch", "ping",
    "profile", "pull", "read", "refactor", "rename", "render", "research",
    "restart", "run", "save", "scan", "search", "set", "setup", "sort", "start",
    "summarise", "summarize", "sync", "test", "trace", "update", "upgrade",
    "validate", "verify", "write",
}

# Objects that only exist behind a tool.
_TOOL_NOUNS = {
    "file", "files", "folder", "folders", "directory", "directories", "repo",
    "repository", "script", "scripts", "log", "logs", "process", "processes",
    "port", "ports", "package", "packages", "dependency", "dependencies",
    "database", "table", "endpoint", "api", "url", "website", "codebase",
    "project", "test", "tests", "commit", "branch", "disk", "memory", "cpu",
}

_NUMBERED_LIST_RE = re.compile(r"(?m)^\s*(?:\d+[.)]\s+|[-*•]\s+)")
_QUESTION_OPENERS = (
    "what", "who", "when", "where", "why", "how", "which", "whose", "is", "are",
    "was", "were", "do", "does", "did", "can", "could", "will", "would", "should",
    "am", "have", "has", "tell me", "explain", "define", "remind me",
)

_VISION_PATTERNS = (
    r"\b(?:web)?cam(?:era)?\b",
    r"\bwhat (?:do|can) you see\b",
    r"\bcan you see\b",
    r"\blook at (?:this|me|that)\b",
    r"\btake a look\b",
    r"\bwhat am i (?:holding|wearing|doing|pointing)\b",
    r"\bhow many (?:fingers|people|faces)\b",
    r"\bin front of (?:me|the camera)\b",
    r"\bwho (?:am i|is (?:this|that))\b",
    r"\bread (?:this|that|my screen|the screen)\b",
    r"\bwhat(?:'s| is) (?:this|that|on (?:my|the) screen)\b",
    r"\bdescribe (?:this|the scene|what you see)\b",
    r"\bmy face\b",
    r"\bhow do i look\b",
)
_VISION_RE = re.compile("|".join(_VISION_PATTERNS), re.IGNORECASE)
_SEQUENCE_RE = re.compile("|".join(_SEQUENCE_PATTERNS), re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z][a-z'-]*")

_CLASSIFIER_PROMPT = """\
You route requests inside an AI assistant. Decide whether a request needs a
multi-step plan (several dependent actions or tool calls) or can be answered in a
single response or single tool call.

Reply with exactly one word: SIMPLE or COMPLEX. No punctuation, no explanation."""


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


def needs_vision(text: str) -> bool:
    """True when the turn plainly refers to what the camera can see."""
    return bool(_VISION_RE.search(text or ""))


def _normalise(text: str) -> str:
    return " ".join((text or "").lower().split())


def classify_heuristic(text: str) -> Classification:
    """Score a turn without touching the network.

    A score at or below :data:`_AMBIGUOUS_LOW` is confidently simple; at or above
    :data:`_AMBIGUOUS_HIGH` is confidently multi-step; the band between is
    ambiguous and reported as ``source="heuristic"`` with ``multi_step=False`` so
    callers may choose to escalate.
    """
    norm = _normalise(text)
    if not norm:
        return Classification(False, 0, "empty input", "heuristic")

    stripped = norm.rstrip("!?. ")
    if stripped in _CHITCHAT:
        return Classification(False, -5, "conversational filler", "heuristic")

    words = _WORD_RE.findall(norm)
    n_words = len(words)
    reasons: list[str] = []
    score = 0

    if _SEQUENCE_RE.search(norm):
        score += 3
        reasons.append("explicit sequencing language")

    list_items = len(_NUMBERED_LIST_RE.findall(text or ""))
    if list_items >= 2:
        score += 3
        reasons.append(f"{list_items} enumerated items")

    verbs = {w for w in words if w in _ACTION_VERBS}
    if len(verbs) >= 2:
        score += 2
        reasons.append(f"{len(verbs)} distinct action verbs")
    elif verbs:
        score += 1
        reasons.append("one action verb")

    nouns = {w for w in words if w in _TOOL_NOUNS}
    if verbs and nouns:
        score += 1
        reasons.append("action applied to a tool-backed object")

    # " ... and <verb> ..." conjoins two separate jobs.
    if verbs and re.search(r"\b(?:and|also|plus|as well as)\b\s+\w+", norm):
        for match in re.finditer(r"\b(?:and|also|plus|as well as)\b\s+([a-z']+)", norm):
            if match.group(1) in _ACTION_VERBS:
                score += 2
                reasons.append("conjoined second action")
                break

    sentences = [s for s in re.split(r"[.!?\n]+", norm) if s.strip()]
    if len(sentences) >= 3 and verbs:
        score += 1
        reasons.append("multiple imperative sentences")

    if n_words >= 45 and verbs:
        score += 1
        reasons.append("long directive")

    # Pull back toward "simple" for plain questions and short turns.
    is_question = norm.endswith("?") or norm.startswith(_QUESTION_OPENERS)
    if is_question and score < _AMBIGUOUS_HIGH:
        score -= 1
        reasons.append("phrased as a question")
    if n_words <= 6:
        score -= 2
        reasons.append("very short")
    if needs_vision(norm) and n_words <= 14:
        score -= 2
        reasons.append("single vision lookup")

    multi = score >= _AMBIGUOUS_HIGH
    reason = "; ".join(reasons) if reasons else "no strong signals"
    return Classification(multi, score, reason, "heuristic")


def _is_ambiguous(result: Classification) -> bool:
    """True when the heuristic score sits in the undecided band."""
    return _AMBIGUOUS_LOW < result.score < _AMBIGUOUS_HIGH


def _read_verdict(raw: str) -> bool | None:
    """Extract SIMPLE/COMPLEX from a model reply, tolerating stray prose."""
    text = (raw or "").strip().lower()
    if not text:
        return None
    text = re.sub(r"<(think|thinking|reasoning)>.*?</\1>", " ", text, flags=re.DOTALL)
    if re.search(r"\b(complex|multi[- ]?step|plan)\b", text):
        return True
    if re.search(r"\b(simple|single[- ]?step|direct|chat)\b", text):
        return False
    return None


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TurnClassifier:
    """Decides whether a turn is worth planning.

    The heuristic answers alone whenever it is confident. Ambiguous turns fall
    through to a cheap single-word LLM judgement, and if that path is unavailable
    or fails, the turn is treated as simple — a missed plan costs a slightly
    worse answer, a spurious plan costs the user several seconds.
    """

    def __init__(
        self,
        provider: Any = None,
        *,
        use_llm_fallback: bool = True,
        model: str | None = None,
        timeout_s: float = _CLASSIFIER_TIMEOUT_S,
    ) -> None:
        self._provider = provider
        self._use_llm = use_llm_fallback and provider is not None
        self._model = model
        self._timeout_s = timeout_s

    async def classify(
        self,
        user_text: str,
        ctx: TurnContext | None = None,
    ) -> Classification:
        """Classify one turn. Never raises; degrades to the heuristic verdict."""
        result = classify_heuristic(user_text)
        if not self._use_llm or not _is_ambiguous(result):
            return result

        verdict = await self._ask_llm(user_text, ctx)
        if verdict is None:
            return Classification(
                result.multi_step,
                result.score,
                f"{result.reason}; llm arbitration unavailable",
                "default",
            )
        return Classification(
            verdict,
            result.score,
            f"{result.reason}; llm says {'complex' if verdict else 'simple'}",
            "llm",
        )

    async def _ask_llm(self, user_text: str, ctx: TurnContext | None) -> bool | None:
        """One-word arbitration for ambiguous turns. Returns None on any failure."""
        messages = [
            Message(role=Role.SYSTEM, content=_CLASSIFIER_PROMPT),
            Message(role=Role.USER, content=self._context_line(user_text, ctx)),
        ]
        try:
            raw = await complete_with_model(
                self._provider, messages, self._model, timeout_s=self._timeout_s
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("classifier llm fallback failed: %s", exc)
            return None
        return _read_verdict(raw or "")

    @staticmethod
    def _context_line(user_text: str, ctx: TurnContext | None) -> str:
        """Give the arbiter the request plus one line of conversational context."""
        request = " ".join((user_text or "").split())[:1000]
        if ctx is None or not ctx.history:
            return f"REQUEST: {request}"
        previous: Sequence[Message] = ctx.history[-2:]
        lines = [
            f"{m.role.value}: {' '.join(m.content.split())[:200]}"
            for m in previous
            if m.content
        ]
        prefix = "RECENT:\n" + "\n".join(lines) + "\n\n" if lines else ""
        return f"{prefix}REQUEST: {request}"
