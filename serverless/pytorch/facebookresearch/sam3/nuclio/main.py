# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 Nuclio Function Handler

Multi-endpoint handler supporting:
- /segment-text: Text-to-Segment (text prompt → mask)
- /segment-box: Box-to-Segment (bbox → mask + polygon)
- /segment-points: Refine with Clicks (points → mask)
- /segment-combined: Text + Points refinement
- /refine-box: Adjust rough bbox to fit object precisely
- /detect: Text-to-Detect (text prompt → all instances)
- /track/init: Initialize video tracking
- /track/frame: Propagate to next frame
- /track/status: Get tracking session status
- /track/clear: Clear tracking session
- /cache/status: Get embeddings cache status
- /cache/clear: Clear embeddings cache
- / (default): Legacy embeddings endpoint
"""

import base64
import io
import json
import logging

import numpy as np
from model_handler import ModelHandler
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_context(context):
    context.logger.info("Init context...  0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")


def decode_image(image_base64: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    buf = io.BytesIO(base64.b64decode(image_base64))
    image = Image.open(buf)
    return image.convert("RGB")


def encode_response(data: dict) -> str:
    """Encode response data to JSON."""
    return json.dumps(data)


def handler(context, event):
    """
    Main handler routing requests to appropriate methods.
    """
    context.logger.info("SAM3 handler called")

    data = event.body
    path = event.path or data.get("endpoint", "/")

    # Remove leading slash for comparison
    path = path.lstrip("/")

    model = context.user_data.model

    try:
        # Route to appropriate handler
        if path == "segment-text":
            return handle_segment_text(context, model, data)

        elif path == "segment-box":
            return handle_segment_box(context, model, data)

        elif path == "segment-points":
            return handle_segment_points(context, model, data)

        elif path == "segment-combined":
            return handle_segment_combined(context, model, data)

        elif path == "refine-box":
            return handle_refine_box(context, model, data)

        elif path == "detect":
            return handle_detect(context, model, data)

        elif path == "track/init":
            return handle_track_init(context, model, data)

        elif path == "track/frame":
            return handle_track_frame(context, model, data)

        elif path == "track/status":
            return handle_track_status(context, model, data)

        elif path == "track/clear":
            return handle_track_clear(context, model, data)

        elif path == "cache/status":
            return handle_cache_status(context, model, data)

        elif path == "cache/clear":
            return handle_cache_clear(context, model, data)

        else:
            # Default: Legacy embeddings endpoint
            return handle_embeddings(context, model, data)

    except Exception as e:
        context.logger.error(f"Handler error: {e}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type="application/json",
            status_code=500,
        )


def handle_embeddings(context, model, data):
    """
    Legacy endpoint: Return model info or perform text-to-segment if text provided.

    For backward compatibility, if 'text' is provided, performs text-to-segment.
    Otherwise, returns model information.
    """
    # If text is provided, treat as segment-text request
    if "text" in data or "text_prompt" in data:
        return handle_segment_text(context, model, data)

    # Otherwise return model info
    model_info = model.get_model_info()

    return context.Response(
        body=json.dumps(model_info),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_segment_text(context, model, data):
    """
    Text-to-Segment: Segment using text prompt.

    Request:
        image: base64 encoded image
        text: Text prompt (e.g., "person in red shirt")
        threshold: Detection confidence threshold (default 0.1)
        job_id: Optional job ID for caching
        frame_idx: Optional frame index for caching

    Response:
        masks: List of masks (as nested lists)
        boxes: List of bounding boxes
        scores: Confidence scores
        labels: Labels (repeated text prompt)
    """
    image = decode_image(data["image"])
    text_prompt = data.get("text", data.get("text_prompt", ""))
    threshold = float(data.get("threshold", 0.1))  # Lower default for Grounding DINO

    if not text_prompt:
        return context.Response(
            body=json.dumps({"error": "text prompt is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.segment_with_text(
        image=image,
        text_prompt=text_prompt,
        threshold=threshold,
        job_id=data.get("job_id"),
        frame_idx=data.get("frame_idx"),
    )

    # Convert numpy arrays to lists for JSON serialization
    if isinstance(result.get("masks"), np.ndarray):
        result["masks"] = result["masks"].tolist()

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_segment_box(context, model, data):
    """
    Box-to-Segment: Segment object inside a bounding box.

    User draws a rough bbox around an object, SAM3 segments it precisely.
    Returns mask and polygon for "Convert masks to polygons" functionality.

    Request:
        image: base64 encoded image
        box: Bounding box [x1, y1, x2, y2] or 4-point polygon [x1,y1,x2,y2,x3,y3,x4,y4]
        job_id: Optional job ID for caching
        frame_idx: Optional frame index for caching

    Response:
        mask: Binary mask (nested list)
        polygon: Contour points for polygon conversion [[x,y], ...]
        bounds: Tight bounding box [x1, y1, x2, y2]
        score: Confidence score
    """
    image = decode_image(data["image"])
    box = data.get("box")

    if not box:
        return context.Response(
            body=json.dumps({"error": "box is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.segment_with_box(
        image=image,
        box=box,
        job_id=data.get("job_id"),
        frame_idx=data.get("frame_idx"),
    )

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_refine_box(context, model, data):
    """
    Refine Box: Adjust a rough bounding box to fit the object precisely.

    User draws a quick/rough bbox, this returns a tight-fitting bbox.
    Also returns polygon for more precise annotations.

    Request:
        image: base64 encoded image
        box: Rough bounding box [x1, y1, x2, y2]
        job_id: Optional job ID for caching
        frame_idx: Optional frame index for caching

    Response:
        refined_box: Tight bounding box [x1, y1, x2, y2]
        polygon: Object contour for polygon annotations [[x,y], ...]
        score: Confidence score
    """
    image = decode_image(data["image"])
    box = data.get("box")

    if not box:
        return context.Response(
            body=json.dumps({"error": "box is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.refine_box(
        image=image,
        box=box,
        job_id=data.get("job_id"),
        frame_idx=data.get("frame_idx"),
    )

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_segment_points(context, model, data):
    """
    Refine with Clicks: Segment using point prompts.

    Request:
        image: base64 encoded image
        pos_points: List of positive points [[x, y], ...]
        neg_points: List of negative points [[x, y], ...]
        box: Optional bounding box [x1, y1, x2, y2]
        job_id: Optional job ID for caching
        frame_idx: Optional frame index for caching

    Response:
        mask: Binary mask (nested list)
        bounds: Bounding box [x1, y1, x2, y2]
        points: Contour points
    """
    image = decode_image(data["image"])
    pos_points = data.get("pos_points", [])
    neg_points = data.get("neg_points", [])
    box = data.get("box")

    if not pos_points and not box:
        return context.Response(
            body=json.dumps({"error": "pos_points or box is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.segment_with_points(
        image=image,
        pos_points=pos_points,
        neg_points=neg_points,
        box=box,
        job_id=data.get("job_id"),
        frame_idx=data.get("frame_idx"),
    )

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_segment_combined(context, model, data):
    """
    Combined: Segment using text + points refinement.

    Request:
        image: base64 encoded image
        text: Text prompt
        pos_points: Optional positive points
        neg_points: Optional negative points
        box: Optional bounding box
        job_id: Optional job ID
        frame_idx: Optional frame index

    Response:
        mask: Binary mask
        bounds: Bounding box
        points: Contour points
    """
    image = decode_image(data["image"])
    text_prompt = data.get("text", data.get("text_prompt", ""))

    if not text_prompt:
        return context.Response(
            body=json.dumps({"error": "text prompt is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.segment_combined(
        image=image,
        text_prompt=text_prompt,
        pos_points=data.get("pos_points"),
        neg_points=data.get("neg_points"),
        box=data.get("box"),
        job_id=data.get("job_id"),
        frame_idx=data.get("frame_idx"),
    )

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_detect(context, model, data):
    """
    Text-to-Detect: Detect all instances matching text.

    Request:
        image: base64 encoded image
        text: Text prompt (e.g., "car")
        threshold: Confidence threshold (default 0.1)
        job_id: Optional job ID
        frame_idx: Optional frame index

    Response:
        detections: List of {mask, bbox, polygon, score, label}
        count: Number of detections
        prompt: Original text prompt
    """
    image = decode_image(data["image"])
    text_prompt = data.get("text", data.get("text_prompt", ""))
    threshold = float(data.get("threshold", 0.1))  # Lower default for Grounding DINO

    if not text_prompt:
        return context.Response(
            body=json.dumps({"error": "text prompt is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.detect_all(
        image=image,
        text_prompt=text_prompt,
        threshold=threshold,
        job_id=data.get("job_id"),
        frame_idx=data.get("frame_idx"),
    )

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_track_init(context, model, data):
    """
    Initialize video tracking session.

    Request:
        image: base64 encoded first frame
        job_id: Job ID for session
        init_type: "mask", "text", "points", or "box"
        mask: Initial mask (for init_type="mask")
        text: Text prompt (for init_type="text")
        pos_points: Points (for init_type="points")
        box: Bounding box (for init_type="box")

    Response:
        session_id: Tracking session ID
        frame_idx: Current frame (0)
        mask: Initial mask
        polygon: Contour points
        object_ids: List of tracked object IDs
    """
    image = decode_image(data["image"])
    job_id = data.get("job_id")

    if not job_id:
        return context.Response(
            body=json.dumps({"error": "job_id is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    init_type = data.get("init_type", "text")
    mask = data.get("mask")
    text_prompt = data.get("text", data.get("text_prompt"))
    pos_points = data.get("pos_points")
    box = data.get("box")

    result = model.init_tracking(
        image=image,
        job_id=job_id,
        init_type=init_type,
        mask=mask,
        text_prompt=text_prompt,
        pos_points=pos_points,
        box=box,
    )

    if result.get("error"):
        return context.Response(
            body=json.dumps(result),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_track_frame(context, model, data):
    """
    Track object to next frame.

    Request:
        image: base64 encoded next frame
        session_id: Tracking session ID

    Response:
        session_id: Session ID
        frame_idx: Current frame index
        mask: Predicted mask
        polygon: Contour points
    """
    image = decode_image(data["image"])
    session_id = data.get("session_id")

    if not session_id:
        return context.Response(
            body=json.dumps({"error": "session_id is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.track_frame(image=image, session_id=session_id)

    if result.get("error"):
        return context.Response(
            body=json.dumps(result),
            headers={},
            content_type="application/json",
            status_code=404,
        )

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_track_status(context, model, data):
    """
    Get tracking session status.

    Request:
        session_id: Tracking session ID

    Response:
        exists: Whether session exists
        session_id: Session ID
        frame_idx: Current frame index
        init_type: Initialization type
        text_prompt: Text prompt if used
        ttl: Remaining TTL in seconds
    """
    session_id = data.get("session_id")

    if not session_id:
        return context.Response(
            body=json.dumps({"error": "session_id is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.get_tracking_status(session_id)

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_track_clear(context, model, data):
    """
    Clear tracking session.

    Request:
        session_id: Tracking session ID

    Response:
        cleared: True if successful
        session_id: Session ID
    """
    session_id = data.get("session_id")

    if not session_id:
        return context.Response(
            body=json.dumps({"error": "session_id is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.clear_tracking(session_id)

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_cache_status(context, model, data):
    """
    Get embeddings cache status.

    Request:
        job_id: Job ID
        frame_idx: Optional frame index

    Response:
        job_id: Job ID
        frame_idx: Frame index
        cached: Whether embeddings are cached
        ttl: Remaining TTL in seconds
    """
    job_id = data.get("job_id")

    if not job_id:
        return context.Response(
            body=json.dumps({"error": "job_id is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.get_cache_status(
        job_id=job_id,
        frame_idx=data.get("frame_idx"),
    )

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def handle_cache_clear(context, model, data):
    """
    Clear embeddings cache.

    Request:
        job_id: Job ID
        frame_idx: Optional frame index

    Response:
        cleared: True if successful
        job_id: Job ID
        frame_idx: Frame index
    """
    job_id = data.get("job_id")

    if not job_id:
        return context.Response(
            body=json.dumps({"error": "job_id is required"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    result = model.clear_cache(
        job_id=job_id,
        frame_idx=data.get("frame_idx"),
    )

    return context.Response(
        body=json.dumps(result),
        headers={},
        content_type="application/json",
        status_code=200,
    )
