"""POST /api/vision/analyze -- multipart image upload -> VisionResult.

Mirrors the WebSocket `frame` handler's pipeline call but for one-shot HTTP
callers that can't hold a socket open (a curl test, a server-side batch
job). The upload is read fully into memory, base64-encoded, and handed to
the same `VisionPipeline.analyze` the `frame` message uses -- never written
to disk (CONTRACTS.md section 7).
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_vision_pipeline
from app.models.schemas import VisionResult

log = logging.getLogger("jarvis.api.vision")

router = APIRouter(prefix="/vision", tags=["vision"])

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/analyze", response_model=VisionResult)
async def analyze_frame(
    file: UploadFile = File(...),
    question: str = Form(""),
    pipeline: Any = Depends(get_vision_pipeline),
) -> VisionResult:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="vision subsystem unavailable")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")

    jpeg_b64 = base64.b64encode(raw).decode("ascii")
    try:
        return await pipeline.analyze(jpeg_b64, question)
    except Exception as exc:
        log.exception("vision analyze failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
