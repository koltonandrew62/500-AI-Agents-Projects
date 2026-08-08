"""Top-level vision orchestrator: base64 JPEG in, :class:`VisionResult` out.

``VisionPipeline.analyze`` is the single entry point the rest of the backend
calls (from the `frame` WebSocket handler for lightweight per-tick
annotation, and from the agent loop -- with a ``question`` -- when a chat
turn needs grounded scene understanding). It:

1. Decodes the incoming base64 JPEG to an in-memory BGR frame. The frame is
   NEVER written to disk (CONTRACTS.md section 7).
2. Short-circuits on a near-identical frame (compared against the last
   *analyzed* frame) by returning the cached result -- this is what keeps a
   webcam ticking at several FPS from re-running detectors and, more
   importantly, from burning free-tier LLM vision calls on a static scene.
3. Otherwise runs every enabled analyzer concurrently and merges their
   output into one :class:`VisionResult`.

Every analyzer degrades independently (see ``detect.py``, ``ocr.py``,
``scene.py`` docstrings) -- a missing optional dependency shrinks the result,
it never raises out of ``analyze()``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.vision import detect, ocr, scene
from app.models.schemas import VisionResult

logger = logging.getLogger(__name__)

# Downsampled signature size used for the frame-similarity short-circuit.
# Small and grayscale on purpose -- this only needs to answer "did the scene
# change at all", not produce a perceptual hash suitable for search.
_SIG_SIZE = (32, 32)

# Mean absolute pixel difference (0-255 scale) below which two frames are
# considered "near-identical" for caching purposes.
_DEFAULT_SIMILARITY_THRESHOLD = 2.0

_cv2: Any = None
_cv2_load_attempted = False


def _get_cv2() -> Any:
    global _cv2, _cv2_load_attempted
    if _cv2 is None and not _cv2_load_attempted:
        _cv2_load_attempted = True
        try:
            import cv2  # type: ignore[import-untyped]

            _cv2 = cv2
        except ImportError:
            logger.warning("opencv-python not installed; VisionPipeline cannot decode frames")
    return _cv2


@dataclass
class _Cache:
    signature: np.ndarray
    question: str
    result: VisionResult
    timestamp: float


def _decode_b64_jpeg(jpeg_b64: str) -> bytes | None:
    """Strip an optional data-URI prefix and base64-decode to raw JPEG bytes."""
    payload = jpeg_b64.split(",", 1)[-1] if jpeg_b64.startswith("data:") else jpeg_b64
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        logger.warning("Received malformed base64 frame payload")
        return None


def _frame_signature(image: np.ndarray) -> np.ndarray | None:
    """Downsample a BGR frame to a small grayscale signature for diffing."""
    cv2 = _get_cv2()
    if cv2 is None:
        return None
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, _SIG_SIZE, interpolation=cv2.INTER_AREA)
        return small.astype(np.float32)
    except Exception:  # pragma: no cover - defensive, OpenCV build variance
        logger.warning("Failed to compute frame signature", exc_info=True)
        return None


def _signatures_similar(
    a: np.ndarray | None, b: np.ndarray | None, threshold: float
) -> bool:
    if a is None or b is None or a.shape != b.shape:
        return False
    return float(np.mean(np.abs(a - b))) < threshold


class VisionPipeline:
    """Decodes webcam frames and merges concurrent analyzer output.

    Not thread-safe across event loops, but safe for the normal single
    asyncio-loop, possibly-concurrent-tasks usage of one instance per
    WebSocket connection (an internal lock serializes the cache
    check-and-update around each ``analyze`` call).
    """

    def __init__(
        self,
        *,
        enable_detection: bool = True,
        enable_ocr: bool = True,
        enable_scene: bool = True,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.enable_detection = enable_detection
        self.enable_ocr = enable_ocr
        self.enable_scene = enable_scene
        self.similarity_threshold = similarity_threshold
        self._cache: _Cache | None = None
        self._lock = asyncio.Lock()

    async def analyze(self, jpeg_b64: str, question: str = "") -> VisionResult:
        """Decode ``jpeg_b64`` and return a merged :class:`VisionResult`.

        ``question`` grounds the LLM scene description (skipped entirely,
        cheaply, when empty -- see module docstring). Never raises: any
        failure at any stage degrades to an emptier-than-ideal result.
        """
        if not self._feature_enabled():
            return VisionResult()

        raw = _decode_b64_jpeg(jpeg_b64)
        if raw is None:
            return VisionResult()

        image = detect.decode_jpeg(raw)
        if image is None:
            return VisionResult()

        height, width = image.shape[:2]
        signature = _frame_signature(image)

        async with self._lock:
            cached = self._cache
            if (
                cached is not None
                and cached.question == question
                and _signatures_similar(signature, cached.signature, self.similarity_threshold)
            ):
                logger.debug("Frame near-identical to last analyzed frame; returning cached result")
                return cached.result

        result = await self._run_analyzers(image, question, width=width, height=height)

        if signature is not None:
            async with self._lock:
                self._cache = _Cache(
                    signature=signature, question=question, result=result, timestamp=time.monotonic()
                )

        return result

    def _feature_enabled(self) -> bool:
        try:
            from app.config import settings

            return bool(getattr(settings, "vision_enabled", True))
        except Exception:  # pragma: no cover - defensive, config layer optional at import time
            return True

    async def _run_analyzers(
        self, image: np.ndarray, question: str, *, width: int, height: int
    ) -> VisionResult:
        tasks: dict[str, asyncio.Task[Any]] = {}

        if self.enable_detection:
            tasks["detect"] = asyncio.create_task(asyncio.to_thread(detect.detect_all, image))
        if self.enable_ocr:
            tasks["ocr"] = asyncio.create_task(asyncio.to_thread(ocr.extract_text, image))
        if self.enable_scene and question:
            tasks["scene"] = asyncio.create_task(scene.describe_scene(_reencode_b64(image), question))

        if not tasks:
            return VisionResult(width=width, height=height)

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        by_name = dict(zip(tasks.keys(), results))

        objects: list[Any] = []
        faces = 0
        hands = 0
        text = ""
        scene_text = ""

        detect_outcome = by_name.get("detect")
        if isinstance(detect_outcome, BaseException):
            logger.warning("Detection analyzer failed", exc_info=detect_outcome)
        elif detect_outcome is not None:
            objects, faces, hands = detect_outcome

        ocr_outcome = by_name.get("ocr")
        if isinstance(ocr_outcome, BaseException):
            logger.warning("OCR analyzer failed", exc_info=ocr_outcome)
        elif ocr_outcome is not None:
            text = ocr_outcome

        scene_outcome = by_name.get("scene")
        if isinstance(scene_outcome, BaseException):
            logger.warning("Scene description analyzer failed", exc_info=scene_outcome)
        elif scene_outcome is not None:
            scene_text = scene_outcome

        return VisionResult(
            objects=objects,
            faces=faces,
            hands=hands,
            text=text,
            scene=scene_text,
            width=width,
            height=height,
        )


def _reencode_b64(image: np.ndarray) -> str:
    """Re-encode an in-memory frame back to base64 JPEG for the vision LLM call.

    We decode once up front for local detectors, so the LLM call re-encodes
    from the decoded array rather than needing the caller's original base64
    string threaded through every analyzer -- still fully in-memory, nothing
    touches disk.
    """
    cv2 = _get_cv2()
    if cv2 is None:
        return ""
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        logger.warning("Failed to re-encode frame to JPEG for scene description")
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")
