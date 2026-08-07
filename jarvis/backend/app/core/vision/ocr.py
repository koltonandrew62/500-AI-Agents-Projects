"""Text extraction (OCR) for webcam frames via ``rapidocr-onnxruntime``.

The RapidOCR engine is lazily imported and cached on first use -- constructing
it is relatively expensive (it loads ONNX models into memory) and it is not a
hard dependency, so a missing install must degrade to an empty string rather
than crashing the vision pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_engine: Any = None
_engine_load_attempted = False
_engine_unavailable_logged = False


def _get_engine() -> Any:
    global _engine, _engine_load_attempted
    if _engine is None and not _engine_load_attempted:
        _engine_load_attempted = True
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]

            _engine = RapidOCR()
        except ImportError:
            logger.warning("rapidocr-onnxruntime not installed; OCR degrades to empty text")
        except Exception:  # pragma: no cover - defensive, model asset issues
            logger.warning("Failed to initialize RapidOCR engine", exc_info=True)
    return _engine


def _clean_and_dedupe(lines: list[str]) -> str:
    """Normalize whitespace and drop consecutive/near-duplicate OCR lines.

    Scanned text from a live webcam frequently repeats the same line across
    a handful of near-identical detections (e.g. a label read twice at
    slightly different box splits). We collapse whitespace, drop empties,
    and dedupe while preserving first-seen order.
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in lines:
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return "\n".join(cleaned)


def extract_text(image: np.ndarray) -> str:
    """Run OCR on a BGR image and return cleaned, de-duplicated text.

    Returns ``""`` if RapidOCR is unavailable or extraction fails for any
    reason -- OCR is best-effort and must never take down the vision
    pipeline.
    """
    global _engine_unavailable_logged
    engine = _get_engine()
    if engine is None:
        if not _engine_unavailable_logged:
            logger.info("OCR skipped (no engine available)")
            _engine_unavailable_logged = True
        return ""

    try:
        result, _elapsed = engine(image)
    except Exception:  # pragma: no cover - defensive, runtime OCR failures
        logger.warning("RapidOCR failed on this frame", exc_info=True)
        return ""

    if not result:
        return ""

    # RapidOCR returns list[[box, text, confidence], ...]
    lines = [entry[1] for entry in result if len(entry) >= 2 and entry[1]]
    return _clean_and_dedupe(lines)
