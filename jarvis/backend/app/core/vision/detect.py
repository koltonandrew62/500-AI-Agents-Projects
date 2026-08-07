"""Object / face / hand detection for webcam frames.

Primary backend is MediaPipe (``face_detection``, ``hands``, and the newer
task-based ``ObjectDetector``). Every MediaPipe import is lazy and wrapped in
``try/except`` so this module -- and therefore the whole vision pipeline --
imports cleanly even when ``mediapipe`` is not installed. Face detection
additionally degrades to an OpenCV Haar cascade when MediaPipe is missing but
``opencv-python`` is present.

All detectors work on an in-memory ``numpy.ndarray`` (BGR, as decoded by
OpenCV) and return :class:`~app.models.schemas.DetectedObject` with pixel
boxes ``(x, y, w, h)`` -- never anything written to disk.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.models.schemas import DetectedObject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy, cached optional dependencies
# ---------------------------------------------------------------------------

_cv2: Any = None
_cv2_load_attempted = False

_mp: Any = None
_mp_load_attempted = False

_haar_face_cascade: Any = None
_haar_load_attempted = False

# MediaPipe task objects, created once and reused across frames.
_mp_face_detector: Any = None
_mp_hands_detector: Any = None
_mp_object_detector: Any = None
_mp_face_init_attempted = False
_mp_hands_init_attempted = False
_mp_object_init_attempted = False

# Bundled MediaPipe object-detection model (EfficientDet-Lite0). If it is not
# present on disk we simply skip general object detection -- faces/hands are
# unaffected since they use their own bundled solutions.
_MP_OBJECT_MODEL_CANDIDATES = (
    "efficientdet_lite0.tflite",
    "/usr/local/share/mediapipe/efficientdet_lite0.tflite",
)


def _get_cv2() -> Any:
    global _cv2, _cv2_load_attempted
    if _cv2 is None and not _cv2_load_attempted:
        _cv2_load_attempted = True
        try:
            import cv2  # type: ignore[import-untyped]

            _cv2 = cv2
        except ImportError:
            logger.warning("opencv-python not installed; frame decoding/detection unavailable")
    return _cv2


def _get_mediapipe() -> Any:
    global _mp, _mp_load_attempted
    if _mp is None and not _mp_load_attempted:
        _mp_load_attempted = True
        try:
            import mediapipe as mp  # type: ignore[import-untyped]

            _mp = mp
        except ImportError:
            logger.warning(
                "mediapipe not installed; using OpenCV Haar cascade fallback for faces, "
                "no hand or general object detection available"
            )
    return _mp


def _get_haar_face_cascade() -> Any:
    """Load the bundled OpenCV Haar cascade for frontal faces (fallback path)."""
    global _haar_face_cascade, _haar_load_attempted
    if _haar_face_cascade is None and not _haar_load_attempted:
        _haar_load_attempted = True
        cv2 = _get_cv2()
        if cv2 is None:
            return None
        try:
            cascade_path = f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                logger.warning("Haar cascade file failed to load: %s", cascade_path)
            else:
                _haar_face_cascade = cascade
        except Exception:  # pragma: no cover - defensive, OpenCV build variance
            logger.warning("Could not initialize Haar cascade face detector", exc_info=True)
    return _haar_face_cascade


def _get_mp_face_detector() -> Any:
    global _mp_face_detector, _mp_face_init_attempted
    if _mp_face_detector is None and not _mp_face_init_attempted:
        _mp_face_init_attempted = True
        mp = _get_mediapipe()
        if mp is None:
            return None
        try:
            _mp_face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("Failed to initialize MediaPipe FaceDetection", exc_info=True)
    return _mp_face_detector


def _get_mp_hands_detector() -> Any:
    global _mp_hands_detector, _mp_hands_init_attempted
    if _mp_hands_detector is None and not _mp_hands_init_attempted:
        _mp_hands_init_attempted = True
        mp = _get_mediapipe()
        if mp is None:
            return None
        try:
            _mp_hands_detector = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=4,
                min_detection_confidence=0.5,
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("Failed to initialize MediaPipe Hands", exc_info=True)
    return _mp_hands_detector


def _get_mp_object_detector() -> Any:
    """Build the task-based MediaPipe ObjectDetector, if a model file is findable."""
    global _mp_object_detector, _mp_object_init_attempted
    if _mp_object_detector is None and not _mp_object_init_attempted:
        _mp_object_init_attempted = True
        mp = _get_mediapipe()
        if mp is None:
            return None
        try:
            import os

            from mediapipe.tasks import python as mp_tasks  # type: ignore[import-untyped]
            from mediapipe.tasks.python import vision as mp_vision  # type: ignore[import-untyped]

            model_path = next(
                (p for p in _MP_OBJECT_MODEL_CANDIDATES if os.path.exists(p)), None
            )
            if model_path is None:
                logger.warning(
                    "MediaPipe object-detection model not found on disk (checked %s); "
                    "general object detection disabled, face/hand detection unaffected",
                    _MP_OBJECT_MODEL_CANDIDATES,
                )
                return None

            base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
            options = mp_vision.ObjectDetectorOptions(
                base_options=base_options,
                max_results=20,
                score_threshold=0.4,
            )
            _mp_object_detector = mp_vision.ObjectDetector.create_from_options(options)
        except Exception:  # pragma: no cover - defensive, optional model asset
            logger.warning("Failed to initialize MediaPipe ObjectDetector", exc_info=True)
    return _mp_object_detector


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray | None:
    """Decode raw JPEG bytes to a BGR ``numpy.ndarray`` using OpenCV.

    Returns ``None`` (with a logged warning) if OpenCV is unavailable or the
    bytes fail to decode. Never touches disk.
    """
    cv2 = _get_cv2()
    if cv2 is None:
        return None
    buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        logger.warning("Failed to decode JPEG frame (corrupt or unsupported data)")
    return image


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def detect_faces(image: np.ndarray) -> list[DetectedObject]:
    """Detect faces, preferring MediaPipe and falling back to Haar cascades."""
    height, width = image.shape[:2]
    detector = _get_mp_face_detector()
    if detector is not None:
        try:
            rgb = _bgr_to_rgb(image)
            result = detector.process(rgb)
            faces: list[DetectedObject] = []
            for detection in result.detections or []:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * width))
                y = max(0, int(bbox.ymin * height))
                w = max(0, int(bbox.width * width))
                h = max(0, int(bbox.height * height))
                score = detection.score[0] if detection.score else 0.5
                faces.append(
                    DetectedObject(label="face", confidence=float(score), box=(x, y, w, h))
                )
            return faces
        except Exception:  # pragma: no cover - defensive
            logger.warning("MediaPipe face detection failed on this frame", exc_info=True)
            return []

    cascade = _get_haar_face_cascade()
    cv2 = _get_cv2()
    if cascade is None or cv2 is None:
        return []
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [
            DetectedObject(label="face", confidence=0.5, box=(int(x), int(y), int(w), int(h)))
            for (x, y, w, h) in boxes
        ]
    except Exception:  # pragma: no cover - defensive
        logger.warning("Haar cascade face detection failed on this frame", exc_info=True)
        return []


def detect_hands(image: np.ndarray) -> list[DetectedObject]:
    """Detect hands via MediaPipe. Returns an empty list if unavailable."""
    detector = _get_mp_hands_detector()
    if detector is None:
        return []
    height, width = image.shape[:2]
    try:
        rgb = _bgr_to_rgb(image)
        result = detector.process(rgb)
        hands: list[DetectedObject] = []
        landmark_lists = result.multi_hand_landmarks or []
        handedness_list = result.multi_handedness or []
        for i, landmarks in enumerate(landmark_lists):
            xs = [lm.x * width for lm in landmarks.landmark]
            ys = [lm.y * height for lm in landmarks.landmark]
            x_min, x_max = max(0, int(min(xs))), min(width, int(max(xs)))
            y_min, y_max = max(0, int(min(ys))), min(height, int(max(ys)))
            confidence = 0.5
            if i < len(handedness_list) and handedness_list[i].classification:
                confidence = float(handedness_list[i].classification[0].score)
            hands.append(
                DetectedObject(
                    label="hand",
                    confidence=confidence,
                    box=(x_min, y_min, x_max - x_min, y_max - y_min),
                )
            )
        return hands
    except Exception:  # pragma: no cover - defensive
        logger.warning("MediaPipe hand detection failed on this frame", exc_info=True)
        return []


def detect_objects(image: np.ndarray) -> list[DetectedObject]:
    """Detect general objects via the MediaPipe task-based ObjectDetector.

    Returns an empty list if MediaPipe or the bundled model file is
    unavailable -- this is expected on installs that skip the model asset.
    """
    detector = _get_mp_object_detector()
    if detector is None:
        return []
    mp = _get_mediapipe()
    try:
        rgb = _bgr_to_rgb(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        objects: list[DetectedObject] = []
        for detection in result.detections:
            category = detection.categories[0] if detection.categories else None
            label = category.category_name if category else "object"
            confidence = float(category.score) if category else 0.0
            bbox = detection.bounding_box
            objects.append(
                DetectedObject(
                    label=label,
                    confidence=confidence,
                    box=(bbox.origin_x, bbox.origin_y, bbox.width, bbox.height),
                )
            )
        return objects
    except Exception:  # pragma: no cover - defensive
        logger.warning("MediaPipe object detection failed on this frame", exc_info=True)
        return []


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    cv2 = _get_cv2()
    if cv2 is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[:, :, ::-1]  # manual channel-swap fallback


def detect_all(image: np.ndarray) -> tuple[list[DetectedObject], int, int]:
    """Run every detector and return ``(objects_incl_faces_and_hands, face_count, hand_count)``.

    Convenience aggregate used by :class:`~app.core.vision.pipeline.VisionPipeline`.
    """
    faces = detect_faces(image)
    hands = detect_hands(image)
    objects = detect_objects(image)
    return [*objects, *faces, *hands], len(faces), len(hands)
