"""Vision subsystem public surface.

Re-exports :class:`VisionPipeline` (the single entry point the rest of the
backend calls) along with the lower-level analyzer helpers, so callers can do
``from app.core.vision import VisionPipeline`` without reaching into
``app.core.vision.pipeline`` directly.

This module must import cleanly with zero optional vision dependencies
installed (opencv-python, mediapipe, rapidocr-onnxruntime) -- every analyzer
in ``pipeline.py`` / ``detect.py`` / ``ocr.py`` / ``scene.py`` lazily imports
its own optional dependency and degrades independently.
"""

from __future__ import annotations

from app.core.vision.detect import decode_jpeg, detect_all, detect_faces, detect_hands, detect_objects
from app.core.vision.ocr import extract_text
from app.core.vision.pipeline import VisionPipeline
from app.core.vision.scene import describe_scene

__all__ = [
    "VisionPipeline",
    "decode_jpeg",
    "detect_all",
    "detect_faces",
    "detect_hands",
    "detect_objects",
    "extract_text",
    "describe_scene",
]
