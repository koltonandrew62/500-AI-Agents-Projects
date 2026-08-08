"""LLM-grounded scene description -- the "what am I holding / what's wrong
with this / how do I fix this" feature.

Calls the vision model chain (CONTRACTS.md section 5, ``settings.vision_models``)
through :class:`app.core.llm.ModelRouter`. The prompt is deliberately tuned
for *practical real-world assistance* rather than a generic image caption:
identify concrete objects, read any visible text, and end with actionable
next steps.

``ModelRouter`` is imported lazily so this module (and therefore the whole
vision pipeline) can be imported even before/without ``app.core.llm`` being
available, and so a missing OpenRouter key degrades to a clear spoken
fallback instead of crashing a webcam turn.
"""

from __future__ import annotations

import logging

from app.models.schemas import Message, Role

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are J.A.R.V.I.S., a real-world visual assistant looking through the "
    "user's webcam. You are not writing a photo caption -- you are helping "
    "someone in the moment. For every frame:\n"
    "1. Identify the concrete object(s), scene, or situation in view. Be "
    "specific (brand, model, part names) whenever you can tell.\n"
    "2. If there is any readable text, labels, error messages, or displays, "
    "read them out and use them.\n"
    "3. If something looks broken, unfinished, or ambiguous, say what's "
    "wrong with it.\n"
    "4. ALWAYS end with concrete, actionable next steps -- what the user "
    "should do next, in order. Prefer numbered steps for anything with more "
    "than one action.\n"
    "Keep it tight: a few sentences of identification, then the steps. No "
    "generic filler like 'this image shows'. If the user asked a specific "
    "question, answer it directly first."
)

DEFAULT_QUESTION = "What am I looking at, and what should I do with it?"

FALLBACK_TEXT = (
    "I can't reach my vision model right now, so I can't describe what's in "
    "frame. Try again in a moment."
)


def _build_messages(question: str, ocr_text: str = "", detected_labels: list[str] | None = None) -> list[Message]:
    hints: list[str] = []
    if detected_labels:
        hints.append(f"On-device detectors also spotted: {', '.join(detected_labels)}.")
    if ocr_text:
        hints.append(f"On-device OCR also read this text from the frame: {ocr_text!r}.")

    user_text = question.strip() or DEFAULT_QUESTION
    if hints:
        user_text = f"{user_text}\n\n(" + " ".join(hints) + ")"

    return [
        Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        Message(role=Role.USER, content=user_text),
    ]


async def describe_scene(
    jpeg_b64: str,
    question: str = "",
    *,
    ocr_text: str = "",
    detected_labels: list[str] | None = None,
) -> str:
    """Answer a question grounded in a webcam frame using the vision LLM chain.

    ``ocr_text`` and ``detected_labels`` are optional hints from the local
    (non-LLM) detectors in this package -- passing them lets the model skip
    re-deriving what on-device detection already found and focus its budget
    on interpretation and next steps.

    Returns :data:`FALLBACK_TEXT` (never raises) if no vision-capable model
    in the chain is reachable, so a webcam turn always gets *some* reply.
    """
    try:
        from app.core.llm import ModelRouter  # lazy: sibling module, may not exist yet
    except ImportError:
        logger.warning("app.core.llm.ModelRouter not available; scene description degraded")
        return FALLBACK_TEXT

    try:
        from app.config import settings

        models = list(settings.vision_models)
    except Exception:  # pragma: no cover - defensive, config layer optional at import time
        logger.warning("Could not load settings.vision_models; using built-in default chain")
        from app.config import DEFAULT_VISION_MODELS

        models = list(DEFAULT_VISION_MODELS)

    if not models:
        logger.warning("No vision models configured; scene description degraded")
        return FALLBACK_TEXT

    messages = _build_messages(question, ocr_text=ocr_text, detected_labels=detected_labels)

    try:
        router = ModelRouter(models)
        reply = await router.vision(messages, jpeg_b64)
        return reply.strip() or FALLBACK_TEXT
    except Exception:  # noqa: BLE001 - any provider/router failure must degrade, not crash
        logger.warning("Vision model chain failed to describe the scene", exc_info=True)
        return FALLBACK_TEXT
