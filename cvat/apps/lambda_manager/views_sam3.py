# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 Views

Extends lambda_manager with endpoints for SAM3 capabilities:
- Text-to-Segment: Segment objects using text prompts
- Refine with Clicks: Interactive refinement
- Text-to-Detect: Detect all instances matching text
- Text-to-Track: Video tracking with text/mask initialization
- Cache management
"""

from __future__ import annotations

import base64
import logging

import requests
from django.conf import settings
from drf_spectacular.utils import (
    OpenApiParameter,
    OpenApiResponse,
    extend_schema,
)
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from cvat.apps.engine.frame_provider import FrameQuality, TaskFrameProvider
from cvat.apps.engine.models import Job
from cvat.apps.engine.types import ExtendedRequest
from cvat.utils.http import make_requests_session

logger = logging.getLogger(__name__)


def invoke_sam3_function(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    """
    Invoke SAM3 Nuclio function.

    Args:
        endpoint: Function endpoint (e.g., '/segment-text', '/detect')
        payload: JSON payload to send
        timeout: Request timeout in seconds

    Returns:
        Response JSON

    Raises:
        requests.HTTPError: If request fails
    """
    nuclio_url = f"{settings.NUCLIO['SCHEME']}://{settings.NUCLIO['HOST']}:{settings.NUCLIO['PORT']}"

    headers = {
        "Content-Type": "application/json",
        "x-nuclio-function-name": "pth-facebookresearch-sam3-gpu",
        "x-nuclio-project-name": "cvat",
        "x-nuclio-function-namespace": settings.NUCLIO.get(
            "FUNCTION_NAMESPACE", "nuclio"
        ),
        "x-nuclio-path": endpoint,
    }

    with make_requests_session() as session:
        response = session.post(
            f"{nuclio_url}/api/function_invocations",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


class SAM3ViewSet(viewsets.ViewSet):
    """
    ViewSet for SAM3 operations.

    Endpoints:
    - POST /segment-text      - Segment using text prompt
    - POST /segment-points    - Segment using click points
    - POST /segment-combined  - Segment using text + points
    - POST /detect            - Detect all instances
    - POST /track/init        - Initialize video tracking
    - POST /track/frame       - Track to next frame
    - GET  /track/status      - Get tracking session status
    - POST /track/clear       - Clear tracking session
    - GET  /cache/status      - Get cache status
    - POST /cache/clear       - Clear cache
    """

    permission_classes = [IsAuthenticated]

    # ========== Text-to-Segment ==========

    @extend_schema(
        summary="Segment with text prompt",
        description="Segment objects in an image using natural language description (Text-to-Segment).",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Job ID"},
                    "frame": {"type": "integer", "description": "Frame index"},
                    "text": {
                        "type": "string",
                        "description": "Text prompt (e.g., 'person in red shirt')",
                    },
                },
                "required": ["job_id", "frame", "text"],
            }
        },
        responses={
            200: OpenApiResponse(
                description="Segmentation results with masks, boxes, scores"
            ),
            400: OpenApiResponse(description="Invalid request"),
            404: OpenApiResponse(description="Job not found"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="segment-text")
    def segment_text(self, request: ExtendedRequest) -> Response:
        """
        Segment using text prompt.
        """
        job_id = request.data.get("job_id")
        frame_idx = request.data.get("frame")
        text_prompt = request.data.get("text")

        if not all([job_id, frame_idx is not None, text_prompt]):
            return Response(
                {"error": "job_id, frame, and text are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            job = Job.objects.select_related("segment", "segment__task").get(
                pk=int(job_id)
            )
        except Job.DoesNotExist:
            return Response(
                {"error": f"Job {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Get frame image
        task = job.segment.task
        frame_provider = TaskFrameProvider(task)

        try:
            frame_data = frame_provider.get_frame(
                frame_idx, quality=FrameQuality.ORIGINAL
            )
            image_base64 = base64.b64encode(frame_data.data.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to get frame {frame_idx}: {e}")
            return Response(
                {"error": f"Failed to get frame: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Call SAM3 function
        try:
            result = invoke_sam3_function(
                "/segment-text",
                {
                    "image": image_base64,
                    "text": text_prompt,
                    "job_id": int(job_id),
                    "frame_idx": frame_idx,
                },
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 segment-text failed: {e}")
            return Response(
                {"error": f"Segmentation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Refine with Clicks ==========

    @extend_schema(
        summary="Segment with point prompts",
        description="Segment objects using positive/negative click points (Refine with Clicks).",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Job ID"},
                    "frame": {"type": "integer", "description": "Frame index"},
                    "pos_points": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                        "description": "Positive points [[x, y], ...]",
                    },
                    "neg_points": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                        "description": "Negative points [[x, y], ...]",
                    },
                    "box": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Bounding box [x1, y1, x2, y2]",
                    },
                },
                "required": ["job_id", "frame"],
            }
        },
        responses={
            200: OpenApiResponse(
                description="Segmentation result with mask, bounds, points"
            ),
            400: OpenApiResponse(description="Invalid request"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="segment-points")
    def segment_points(self, request: ExtendedRequest) -> Response:
        """
        Segment using point prompts.
        """
        job_id = request.data.get("job_id")
        frame_idx = request.data.get("frame")
        pos_points = request.data.get("pos_points", [])
        neg_points = request.data.get("neg_points", [])
        box = request.data.get("box")

        if not job_id or frame_idx is None:
            return Response(
                {"error": "job_id and frame are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not pos_points and not box:
            return Response(
                {"error": "pos_points or box is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            job = Job.objects.select_related("segment", "segment__task").get(
                pk=int(job_id)
            )
        except Job.DoesNotExist:
            return Response(
                {"error": f"Job {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Get frame image
        task = job.segment.task
        frame_provider = TaskFrameProvider(task)

        try:
            frame_data = frame_provider.get_frame(
                frame_idx, quality=FrameQuality.ORIGINAL
            )
            image_base64 = base64.b64encode(frame_data.data.getvalue()).decode("utf-8")
        except Exception as e:
            return Response(
                {"error": f"Failed to get frame: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Call SAM3 function
        try:
            result = invoke_sam3_function(
                "/segment-points",
                {
                    "image": image_base64,
                    "pos_points": pos_points,
                    "neg_points": neg_points,
                    "box": box,
                    "job_id": int(job_id),
                    "frame_idx": frame_idx,
                },
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 segment-points failed: {e}")
            return Response(
                {"error": f"Segmentation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Combined (Text + Points) ==========

    @extend_schema(
        summary="Segment with text and points",
        description="Segment using text prompt refined with click points.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer"},
                    "frame": {"type": "integer"},
                    "text": {"type": "string", "description": "Text prompt"},
                    "pos_points": {"type": "array"},
                    "neg_points": {"type": "array"},
                    "box": {"type": "array"},
                },
                "required": ["job_id", "frame", "text"],
            }
        },
        responses={200: OpenApiResponse(description="Refined segmentation result")},
    )
    @action(detail=False, methods=["POST"], url_path="segment-combined")
    def segment_combined(self, request: ExtendedRequest) -> Response:
        """
        Segment using text + points refinement.
        """
        job_id = request.data.get("job_id")
        frame_idx = request.data.get("frame")
        text_prompt = request.data.get("text")

        if not all([job_id, frame_idx is not None, text_prompt]):
            return Response(
                {"error": "job_id, frame, and text are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            job = Job.objects.select_related("segment", "segment__task").get(
                pk=int(job_id)
            )
        except Job.DoesNotExist:
            return Response(
                {"error": f"Job {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        task = job.segment.task
        frame_provider = TaskFrameProvider(task)

        try:
            frame_data = frame_provider.get_frame(
                frame_idx, quality=FrameQuality.ORIGINAL
            )
            image_base64 = base64.b64encode(frame_data.data.getvalue()).decode("utf-8")
        except Exception as e:
            return Response(
                {"error": f"Failed to get frame: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_sam3_function(
                "/segment-combined",
                {
                    "image": image_base64,
                    "text": text_prompt,
                    "pos_points": request.data.get("pos_points"),
                    "neg_points": request.data.get("neg_points"),
                    "box": request.data.get("box"),
                    "job_id": int(job_id),
                    "frame_idx": frame_idx,
                },
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 segment-combined failed: {e}")
            return Response(
                {"error": f"Segmentation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Text-to-Detect ==========

    @extend_schema(
        summary="Detect all instances",
        description="Detect ALL instances matching text prompt in an image (Text-to-Detect).",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer"},
                    "frame": {"type": "integer"},
                    "text": {
                        "type": "string",
                        "description": "What to detect (e.g., 'car')",
                    },
                    "threshold": {"type": "number", "default": 0.25},
                },
                "required": ["job_id", "frame", "text"],
            }
        },
        responses={
            200: OpenApiResponse(description="Detection results with all instances"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="detect")
    def detect(self, request: ExtendedRequest) -> Response:
        """
        Detect all instances matching text prompt.
        """
        job_id = request.data.get("job_id")
        frame_idx = request.data.get("frame")
        text_prompt = request.data.get("text")
        threshold = float(request.data.get("threshold", 0.25))

        if not all([job_id, frame_idx is not None, text_prompt]):
            return Response(
                {"error": "job_id, frame, and text are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            job = Job.objects.select_related("segment", "segment__task").get(
                pk=int(job_id)
            )
        except Job.DoesNotExist:
            return Response(
                {"error": f"Job {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        task = job.segment.task
        frame_provider = TaskFrameProvider(task)

        try:
            frame_data = frame_provider.get_frame(
                frame_idx, quality=FrameQuality.ORIGINAL
            )
            image_base64 = base64.b64encode(frame_data.data.getvalue()).decode("utf-8")
        except Exception as e:
            return Response(
                {"error": f"Failed to get frame: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_sam3_function(
                "/detect",
                {
                    "image": image_base64,
                    "text": text_prompt,
                    "threshold": threshold,
                    "job_id": int(job_id),
                    "frame_idx": frame_idx,
                },
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 detect failed: {e}")
            return Response(
                {"error": f"Detection failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Text-to-Track: Init ==========

    @extend_schema(
        summary="Initialize video tracking",
        description="Initialize a video tracking session with text, mask, points, or box.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer"},
                    "frame": {"type": "integer", "description": "Initial frame index"},
                    "init_type": {
                        "type": "string",
                        "enum": ["text", "mask", "points", "box"],
                        "description": "Initialization type",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text prompt (for init_type=text)",
                    },
                    "mask": {
                        "type": "array",
                        "description": "Initial mask (for init_type=mask)",
                    },
                    "pos_points": {
                        "type": "array",
                        "description": "Points (for init_type=points)",
                    },
                    "box": {
                        "type": "array",
                        "description": "Box [x1,y1,x2,y2] (for init_type=box)",
                    },
                },
                "required": ["job_id", "frame", "init_type"],
            }
        },
        responses={
            200: OpenApiResponse(description="Tracking session initialized"),
            400: OpenApiResponse(description="Invalid request"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="track/init")
    def track_init(self, request: ExtendedRequest) -> Response:
        """
        Initialize video tracking session.
        """
        job_id = request.data.get("job_id")
        frame_idx = request.data.get("frame")
        init_type = request.data.get("init_type", "text")

        if not job_id or frame_idx is None:
            return Response(
                {"error": "job_id and frame are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            job = Job.objects.select_related("segment", "segment__task").get(
                pk=int(job_id)
            )
        except Job.DoesNotExist:
            return Response(
                {"error": f"Job {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        task = job.segment.task
        frame_provider = TaskFrameProvider(task)

        try:
            frame_data = frame_provider.get_frame(
                frame_idx, quality=FrameQuality.ORIGINAL
            )
            image_base64 = base64.b64encode(frame_data.data.getvalue()).decode("utf-8")
        except Exception as e:
            return Response(
                {"error": f"Failed to get frame: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_sam3_function(
                "/track/init",
                {
                    "image": image_base64,
                    "job_id": int(job_id),
                    "init_type": init_type,
                    "text": request.data.get("text"),
                    "mask": request.data.get("mask"),
                    "pos_points": request.data.get("pos_points"),
                    "box": request.data.get("box"),
                },
                timeout=180,  # Tracking init may take longer
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 track/init failed: {e}")
            return Response(
                {"error": f"Tracking initialization failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Text-to-Track: Frame ==========

    @extend_schema(
        summary="Track to next frame",
        description="Propagate tracking to the next frame in the video.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer"},
                    "frame": {"type": "integer", "description": "Target frame index"},
                    "session_id": {
                        "type": "string",
                        "description": "Tracking session ID",
                    },
                },
                "required": ["job_id", "frame", "session_id"],
            }
        },
        responses={
            200: OpenApiResponse(description="Tracking result for frame"),
            404: OpenApiResponse(description="Session not found"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="track/frame")
    def track_frame(self, request: ExtendedRequest) -> Response:
        """
        Track object to next frame.
        """
        job_id = request.data.get("job_id")
        frame_idx = request.data.get("frame")
        session_id = request.data.get("session_id")

        if not all([job_id, frame_idx is not None, session_id]):
            return Response(
                {"error": "job_id, frame, and session_id are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            job = Job.objects.select_related("segment", "segment__task").get(
                pk=int(job_id)
            )
        except Job.DoesNotExist:
            return Response(
                {"error": f"Job {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        task = job.segment.task
        frame_provider = TaskFrameProvider(task)

        try:
            frame_data = frame_provider.get_frame(
                frame_idx, quality=FrameQuality.ORIGINAL
            )
            image_base64 = base64.b64encode(frame_data.data.getvalue()).decode("utf-8")
        except Exception as e:
            return Response(
                {"error": f"Failed to get frame: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_sam3_function(
                "/track/frame",
                {
                    "image": image_base64,
                    "session_id": session_id,
                },
            )

            if result.get("error"):
                return Response(result, status=status.HTTP_404_NOT_FOUND)

            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 track/frame failed: {e}")
            return Response(
                {"error": f"Tracking failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Track Status ==========

    @extend_schema(
        summary="Get tracking session status",
        description="Check status of a video tracking session.",
        parameters=[
            OpenApiParameter(name="session_id", type=str, required=True),
        ],
        responses={200: OpenApiResponse(description="Tracking session status")},
    )
    @action(detail=False, methods=["GET"], url_path="track/status")
    def track_status(self, request: ExtendedRequest) -> Response:
        """
        Get tracking session status.
        """
        session_id = request.query_params.get("session_id")

        if not session_id:
            return Response(
                {"error": "session_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_sam3_function(
                "/track/status",
                {"session_id": session_id},
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 track/status failed: {e}")
            return Response(
                {"error": f"Failed to get status: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Track Clear ==========

    @extend_schema(
        summary="Clear tracking session",
        description="Clear a video tracking session and free resources.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                },
                "required": ["session_id"],
            }
        },
        responses={200: OpenApiResponse(description="Tracking session cleared")},
    )
    @action(detail=False, methods=["POST"], url_path="track/clear")
    def track_clear(self, request: ExtendedRequest) -> Response:
        """
        Clear tracking session.
        """
        session_id = request.data.get("session_id")

        if not session_id:
            return Response(
                {"error": "session_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_sam3_function(
                "/track/clear",
                {"session_id": session_id},
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 track/clear failed: {e}")
            return Response(
                {"error": f"Failed to clear session: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Cache Status ==========

    @extend_schema(
        summary="Get cache status",
        description="Check embeddings cache status for a job/frame.",
        parameters=[
            OpenApiParameter(name="job_id", type=int, required=True),
            OpenApiParameter(name="frame", type=int, required=False),
        ],
        responses={200: OpenApiResponse(description="Cache status")},
    )
    @action(detail=False, methods=["GET"], url_path="cache/status")
    def cache_status(self, request: ExtendedRequest) -> Response:
        """
        Get embeddings cache status.
        """
        job_id = request.query_params.get("job_id")
        frame_idx = request.query_params.get("frame")

        if not job_id:
            return Response(
                {"error": "job_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_sam3_function(
                "/cache/status",
                {
                    "job_id": int(job_id),
                    "frame_idx": int(frame_idx) if frame_idx else None,
                },
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 cache/status failed: {e}")
            return Response(
                {"error": f"Failed to get cache status: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Cache Clear ==========

    @extend_schema(
        summary="Clear cache",
        description="Clear embeddings cache for a job/frame.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer"},
                    "frame": {"type": "integer"},
                },
                "required": ["job_id"],
            }
        },
        responses={200: OpenApiResponse(description="Cache cleared")},
    )
    @action(detail=False, methods=["POST"], url_path="cache/clear")
    def cache_clear(self, request: ExtendedRequest) -> Response:
        """
        Clear embeddings cache.
        """
        job_id = request.data.get("job_id")
        frame_idx = request.data.get("frame")

        if not job_id:
            return Response(
                {"error": "job_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_sam3_function(
                "/cache/clear",
                {
                    "job_id": int(job_id),
                    "frame_idx": frame_idx,
                },
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"SAM3 cache/clear failed: {e}")
            return Response(
                {"error": f"Failed to clear cache: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # ========== Apply Detections ==========

    @extend_schema(
        summary="Apply detections as annotations",
        description="Convert SAM3 detection results to CVAT annotations.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer"},
                    "frame": {"type": "integer"},
                    "detections": {
                        "type": "array",
                        "description": "Detection results from detect endpoint",
                    },
                    "output_type": {
                        "type": "string",
                        "enum": ["polygon", "rectangle", "mask"],
                        "default": "polygon",
                    },
                },
                "required": ["job_id", "frame", "detections"],
            }
        },
        responses={
            200: OpenApiResponse(description="Annotations applied"),
            400: OpenApiResponse(description="Invalid request"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="apply")
    def apply_detections(self, request: ExtendedRequest) -> Response:
        """
        Apply detection results as annotations to the job.
        """
        job_id = request.data.get("job_id")
        frame_idx = request.data.get("frame")
        detections = request.data.get("detections", [])
        output_type = request.data.get("output_type", "polygon")

        if not job_id or frame_idx is None:
            return Response(
                {"error": "job_id and frame are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not detections:
            return Response(
                {"error": "detections is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            job = Job.objects.get(pk=int(job_id))
        except Job.DoesNotExist:
            return Response(
                {"error": f"Job {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Get label mapping
        label_mapping = {}
        for label in job.get_labels():
            label_mapping[label.name.lower()] = label.id

        # Convert detections to CVAT format
        shapes = []
        for det in detections:
            label_name = det.get("label", "").lower()
            label_id = label_mapping.get(label_name)

            if label_id is None:
                # Try case-insensitive match
                for name, lid in label_mapping.items():
                    if name.lower() == label_name:
                        label_id = lid
                        break

            if label_id is None:
                logger.warning(f"Label '{det.get('label')}' not found in job labels")
                continue

            if output_type == "polygon" and det.get("polygon"):
                points = det["polygon"]
                shape_type = "polygon"
            elif output_type == "rectangle" and det.get("bbox"):
                points = det["bbox"]
                shape_type = "rectangle"
            elif det.get("bbox"):
                points = det["bbox"]
                shape_type = "rectangle"
            else:
                continue

            shapes.append(
                {
                    "type": shape_type,
                    "frame": frame_idx,
                    "label_id": label_id,
                    "points": points,
                    "occluded": False,
                    "z_order": 0,
                    "source": "auto",
                    "attributes": [],
                }
            )

        if not shapes:
            return Response(
                {
                    "applied": 0,
                    "message": "No valid annotations. Check that labels match job labels.",
                }
            )

        # Apply annotations
        import cvat.apps.dataset_manager as dm
        from cvat.apps.dataset_manager.task import PatchAction

        data = {"shapes": shapes, "tracks": [], "tags": []}
        dm.patch_job_data(job.id, data, PatchAction.CREATE)

        return Response(
            {
                "applied": len(shapes),
                "message": f"Applied {len(shapes)} annotations",
            }
        )
