"""Speech-to-text transcription, server-side, optional.

Per CONTRACTS.md section 3, the primary voice path is browser-side STT (the
client sends a ``voice`` message with an already-transcribed ``text``). This
module exists as an optional server-side fallback -- e.g. for raw audio
uploaded some other way -- via ``faster-whisper`` when it happens to be
installed. It is never a hard dependency: with no engine available,
``transcribe`` returns ``""`` rather than raising, and the import is lazy so
application startup never pays the (large) faster-whisper/ctranslate2 import
cost unless this path is actually used.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_model: Any = None
_model_load_attempted = False
_model_unavailable_logged = False

# Small, CPU-friendly default. Overridable via settings in a future pass;
# kept as a local constant so this module has zero import-time coupling
# beyond what's already used elsewhere in this package.
_DEFAULT_MODEL_SIZE = "base"


def _get_model() -> Any:
    """Lazily construct and cache a ``faster-whisper`` model.

    Returns ``None`` (without raising) if ``faster-whisper`` is not
    installed, or if model construction fails for any reason (missing
    weights, unsupported hardware, etc.).
    """
    global _model, _model_load_attempted
    if _model is None and not _model_load_attempted:
        _model_load_attempted = True
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            _model = WhisperModel(_DEFAULT_MODEL_SIZE, device="cpu", compute_type="int8")
        except ImportError:
            logger.warning(
                "faster-whisper not installed; server-side STT disabled "
                "(browser STT via the `voice` message remains the primary path)"
            )
        except Exception:  # pragma: no cover - defensive, model/hardware variance
            logger.warning("faster-whisper failed to load a model; server-side STT disabled", exc_info=True)
    return _model


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe raw audio bytes to text, or ``""`` if unavailable.

    Never raises. Accepts whatever container/codec ``faster-whisper``'s
    bundled ffmpeg-based decoder can read (wav, mp3, webm, ...). The bytes
    are written to a temporary file only because the underlying engine reads
    from a filesystem path -- the file lives in a process-local temp
    directory and is removed immediately after transcription.
    """
    global _model_unavailable_logged
    if not audio_bytes:
        return ""

    model = _get_model()
    if model is None:
        if not _model_unavailable_logged:
            logger.info("STT transcription skipped (no engine available)")
            _model_unavailable_logged = True
        return ""

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "audio.bin"
            audio_path.write_bytes(audio_bytes)
            segments, _info = model.transcribe(str(audio_path))
            text = " ".join(segment.text.strip() for segment in segments)
            return text.strip()
    except Exception:  # pragma: no cover - defensive, decode/runtime failures
        logger.warning("faster-whisper failed to transcribe this audio", exc_info=True)
        return ""
