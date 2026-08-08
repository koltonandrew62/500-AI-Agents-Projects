"""Text-to-speech synthesis, server-side, best-effort.

``synthesize`` tries ``pyttsx3`` (which itself shells out to a platform
engine -- ``espeak``/``espeak-ng`` on Linux, SAPI5 on Windows, NSSpeech on
macOS) and returns raw audio bytes on success. When no TTS engine is
available -- ``pyttsx3`` not installed, or installed but no backing engine
found on the host -- it returns ``None`` so the caller (the WebSocket layer,
via the ``speak`` event in CONTRACTS.md section 3) can fall back to the
browser's ``SpeechSynthesis`` API instead. This module must never raise and
must never hard-require a TTS engine to be present.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_engine: Any = None
_engine_load_attempted = False
_engine_unavailable_logged = False


def _get_engine() -> Any:
    """Lazily construct and cache a ``pyttsx3`` engine instance.

    Returns ``None`` (without raising) if ``pyttsx3`` is not installed or if
    it fails to bind to a platform speech engine (e.g. no ``espeak`` on a
    minimal Linux container).
    """
    global _engine, _engine_load_attempted
    if _engine is None and not _engine_load_attempted:
        _engine_load_attempted = True
        try:
            import pyttsx3  # type: ignore[import-untyped]

            _engine = pyttsx3.init()
        except ImportError:
            logger.warning("pyttsx3 not installed; server-side TTS disabled, browser SpeechSynthesis will be used")
        except Exception:  # pragma: no cover - defensive, platform engine variance
            logger.warning("pyttsx3 failed to bind a platform speech engine; falling back to browser TTS", exc_info=True)
    return _engine


def synthesize(text: str) -> bytes | None:
    """Synthesize ``text`` to audio bytes (WAV), or ``None`` if unavailable.

    Never raises. A ``None`` return is a normal, expected degradation path --
    it signals the caller to let the browser speak the text instead via the
    ``speak`` WebSocket event.
    """
    global _engine_unavailable_logged
    text = text.strip()
    if not text:
        return None

    engine = _get_engine()
    if engine is None:
        if not _engine_unavailable_logged:
            logger.info("TTS synthesis skipped (no engine available); delegating to browser SpeechSynthesis")
            _engine_unavailable_logged = True
        return None

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "speech.wav"
            engine.save_to_file(text, str(out_path))
            engine.runAndWait()
            if not out_path.exists():
                logger.warning("pyttsx3 reported success but produced no output file")
                return None
            return out_path.read_bytes()
    except Exception:  # pragma: no cover - defensive, engine/runtime failures
        logger.warning("pyttsx3 synthesis failed for this utterance", exc_info=True)
        return None
