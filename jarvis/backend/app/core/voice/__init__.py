"""Voice subsystem public surface: server-side TTS/STT, both optional.

Both ``synthesize`` and ``transcribe`` degrade gracefully with no optional
dependency installed (``pyttsx3``, ``faster-whisper``) -- see their
respective modules. This package must import cleanly regardless.
"""

from __future__ import annotations

from app.core.voice.stt import transcribe
from app.core.voice.tts import synthesize

__all__ = ["synthesize", "transcribe"]
