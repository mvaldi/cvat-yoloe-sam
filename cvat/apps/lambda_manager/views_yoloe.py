# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
YOLOE Visual Prompt Views

Extends lambda_manager with endpoints for YOLOE Visual Prompting:
- Get annotated frames from a job
- Generate VPE from reference frames
- Run detection on target frames
- Manage VPE cache
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


# Maximum number of reference frames
MAX_REFERENCES = 50


def get_nuclio_url(endpoint: str = "") -> str:
    """Get Nuclio function URL for YOLOE Visual Prompt."""
    scheme = settings.NUCLIO.get("SCHEME", "http")
    host = settings.NUCLIO.get("HOST", "localhost")

    base_url = f"{scheme}://{host}"

    if endpoint:
        return f"{base_url}/{endpoint.lstrip('/')}"
    return base_url


def invoke_yoloe_function(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    """
    Invoke YOLOE Visual Prompt Nuclio function.

    Args:
        endpoint: Function endpoint (e.g., '/generate-vpe', '/predict')
        payload: JSON payload to send
        timeout: Request timeout in seconds

    Returns:
        Response JSON

    Raises:
        requests.HTTPError: If request fails
    """
    # Get function URL from Nuclio dashboard
    nuclio_url = f"{settings.NUCLIO['SCHEME']}://{settings.NUCLIO['HOST']}:{settings.NUCLIO['PORT']}"

    headers = {
        "Content-Type": "application/json",
        "x-nuclio-function-name": "pth-ultralytics-yoloe-visual-prompt",
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


class YOLOEVisualPromptViewSet(viewsets.ViewSet):
    """
    ViewSet for YOLOE Visual Prompting operations.

    Endpoints:
    - GET  /annotated-frames?job_id=N     - List frames with annotations
    - POST /generate-vpe                   - Generate VPE from references
    - POST /predict                        - Run detection on target frames
    - GET  /status?job_id=N               - Get VPE cache status
    - POST /clear                          - Clear VPE cache
    """

    # Use standard DRF authentication
    permission_classes = [IsAuthenticated]

    @extend_schema(
        summary="Get annotated frames",
        description="List frames that have annotations in a job. Only these can be used as references.",
        parameters=[
            OpenApiParameter(
                name="job_id",
                type=int,
                required=True,
                description="Job ID",
            ),
        ],
        responses={
            200: OpenApiResponse(
                description="List of annotated frames with their annotations"
            ),
            404: OpenApiResponse(description="Job not found"),
        },
    )
    @action(detail=False, methods=["GET"], url_path="annotated-frames")
    def annotated_frames(self, request: ExtendedRequest) -> Response:
        """
        Get list of frames with annotations in a job.

        Returns frames that have at least one annotation, along with:
        - Frame index
        - Number of annotations
        - Labels used
        """
        job_id = request.query_params.get("job_id")
        if not job_id:
            return Response(
                {"error": "job_id is required"},
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

        # Get annotations for the job using dataset_manager
        import cvat.apps.dataset_manager as dm

        # get_job_data returns a dict with shapes, tracks, tags
        job_annotations = dm.task.get_job_data(job.id)

        # Group annotations by frame
        annotated_frames = {}

        # Process shapes (rectangles, polygons, etc.)
        # job_annotations is a dict with 'shapes', 'tracks', 'tags' keys
        shapes = job_annotations.get("shapes", [])
        tracks = job_annotations.get("tracks", [])

        logger.info(
            f"YOLOE annotated_frames: Found {len(shapes)} shapes and {len(tracks)} tracks"
        )

        # Process standalone shapes
        for shape in shapes:
            frame = shape.get("frame", 0)
            if frame not in annotated_frames:
                annotated_frames[frame] = {
                    "frame": frame,
                    "annotations": [],
                    "labels": set(),
                }

            label_id = shape.get("label_id")
            label_name = self._get_label_name(job, label_id)

            annotated_frames[frame]["annotations"].append(
                {
                    "type": shape.get("type"),
                    "label": label_name,
                    "points": shape.get("points", []),
                }
            )
            annotated_frames[frame]["labels"].add(label_name)

        # Process track shapes
        for track in tracks:
            track_label_id = track.get("label_id")
            track_label_name = self._get_label_name(job, track_label_id)
            track_shapes = track.get("shapes", [])

            for shape in track_shapes:
                frame = shape.get("frame", 0)
                if shape.get("outside", False):
                    continue  # Skip shapes marked as outside

                if frame not in annotated_frames:
                    annotated_frames[frame] = {
                        "frame": frame,
                        "annotations": [],
                        "labels": set(),
                    }

                annotated_frames[frame]["annotations"].append(
                    {
                        "type": shape.get("type"),
                        "label": track_label_name,
                        "points": shape.get("points", []),
                    }
                )
                annotated_frames[frame]["labels"].add(track_label_name)

        # Convert sets to lists and sort by frame
        result = []
        for frame_data in sorted(annotated_frames.values(), key=lambda x: x["frame"]):
            frame_data["labels"] = sorted(list(frame_data["labels"]))
            frame_data["annotation_count"] = len(frame_data["annotations"])
            result.append(frame_data)

        return Response(
            {
                "job_id": int(job_id),
                "total_frames": job.segment.stop_frame - job.segment.start_frame + 1,
                "annotated_count": len(result),
                "max_references": MAX_REFERENCES,
                "frames": result,
            }
        )

    def _get_label_name(self, job: Job, label_id: int) -> str:
        """Get label name from ID."""
        try:
            label = job.get_labels().get(pk=label_id)
            return label.name
        except Exception:
            return f"label_{label_id}"

    @extend_schema(
        summary="Generate Visual Prompt Embeddings",
        description="Generate VPE from selected reference frames with annotations. Max 50 references.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Job ID"},
                    "reference_frames": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of frame indices to use as references",
                        "maxItems": MAX_REFERENCES,
                    },
                },
                "required": ["job_id", "reference_frames"],
            }
        },
        responses={
            200: OpenApiResponse(description="VPE generated and cached successfully"),
            400: OpenApiResponse(description="Invalid request"),
            404: OpenApiResponse(description="Job not found"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="generate-vpe")
    def generate_vpe(self, request: ExtendedRequest) -> Response:
        """
        Generate Visual Prompt Embeddings from reference frames.

        Collects images and annotations from specified frames,
        sends to YOLOE function, and caches the VPE.
        """
        job_id = request.data.get("job_id")
        reference_frames = request.data.get("reference_frames", [])

        if not job_id:
            return Response(
                {"error": "job_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not reference_frames:
            return Response(
                {"error": "reference_frames is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if len(reference_frames) > MAX_REFERENCES:
            return Response(
                {"error": f"Maximum {MAX_REFERENCES} reference frames allowed"},
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

        # Get frame provider for images
        task = job.segment.task
        frame_provider = TaskFrameProvider(task)

        # Get annotations using dataset_manager
        import cvat.apps.dataset_manager as dm

        # get_job_data returns a dict with shapes, tracks, tags
        job_annotations = dm.task.get_job_data(job.id)
        shapes = job_annotations.get("shapes", [])
        tracks = job_annotations.get("tracks", [])

        logger.info(
            f"YOLOE generate_vpe: Found {len(shapes)} shapes and {len(tracks)} tracks"
        )
        logger.info(f"YOLOE generate_vpe: reference_frames = {reference_frames}")

        # Build references payload
        references = []

        for frame_idx in reference_frames:
            # Get frame image
            try:
                frame_data = frame_provider.get_frame(
                    frame_idx,
                    quality=FrameQuality.ORIGINAL,
                )
                image_base64 = base64.b64encode(frame_data.data.getvalue()).decode(
                    "utf-8"
                )
            except Exception as e:
                logger.warning(f"Failed to get frame {frame_idx}: {e}")
                continue

            # Get annotations for this frame - check both shapes and track shapes
            frame_annotations = []

            # Collect all shapes for this frame (from shapes and tracks)
            all_shapes = list(shapes)
            for track in tracks:
                track_shapes = track.get("shapes", [])
                track_label_id = track.get("label_id")
                for ts in track_shapes:
                    # Add label_id from track to shape
                    shape_with_label = dict(ts)
                    shape_with_label["label_id"] = track_label_id
                    all_shapes.append(shape_with_label)

            for shape in all_shapes:
                if shape.get("frame") != frame_idx:
                    continue

                shape_type = shape.get("type")
                points = shape.get("points", [])
                label_id = shape.get("label_id")
                label_name = self._get_label_name(job, label_id)

                logger.info(
                    f"YOLOE: Frame {frame_idx}, shape_type={shape_type}, type={type(shape_type)}, points_len={len(points)}, label={label_name}"
                )

                # Convert to bbox format [x1, y1, x2, y2]
                # shape_type is a string: "rectangle", "polygon", "polyline", etc.
                if shape_type == "rectangle":
                    # Rectangle is already in [x1, y1, x2, y2] format
                    bbox = points[:4] if len(points) >= 4 else None
                elif shape_type == "polygon" and len(points) >= 6:
                    # Get bounding box from polygon points
                    xs = points[0::2]
                    ys = points[1::2]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    # Other shapes - get bounding box
                    if len(points) >= 4:
                        xs = points[0::2]
                        ys = points[1::2]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                    else:
                        continue

                if bbox:
                    frame_annotations.append(
                        {
                            "bbox": bbox,
                            "label": label_name,
                        }
                    )

            if frame_annotations:
                references.append(
                    {
                        "frame": frame_idx,
                        "image": image_base64,
                        "annotations": frame_annotations,
                    }
                )

        if not references:
            return Response(
                {
                    "error": "No valid references found. Ensure selected frames have annotations."
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Call YOLOE function to generate VPE
        try:
            result = invoke_yoloe_function(
                "/generate-vpe",
                {
                    "job_id": int(job_id),
                    "references": references,
                },
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"YOLOE generate-vpe failed: {e}")
            return Response(
                {"error": f"Failed to generate VPE: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Predict on frames",
        description="Run YOLOE detection on target frames using cached VPE.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Job ID"},
                    "frames": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Frame indices to run detection on. If empty, uses current frame.",
                    },
                    "threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.25,
                        "description": "Confidence threshold",
                    },
                    "output_type": {
                        "type": "string",
                        "enum": ["rectangle", "polygon", "obb"],
                        "default": "rectangle",
                        "description": "Output annotation type",
                    },
                },
                "required": ["job_id"],
            }
        },
        responses={
            200: OpenApiResponse(description="Detection results"),
            400: OpenApiResponse(description="Invalid request"),
            404: OpenApiResponse(description="Job or VPE not found"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="predict")
    def predict(self, request: ExtendedRequest) -> Response:
        """
        Run YOLOE detection on target frames using cached VPE.

        Returns detection results that can be applied as annotations.
        """
        job_id = request.data.get("job_id")
        frames = request.data.get("frames", [])
        threshold = float(request.data.get("threshold", 0.25))
        output_type = request.data.get("output_type", "rectangle")

        if not job_id:
            return Response(
                {"error": "job_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            job = Job.objects.get(pk=int(job_id))
        except Job.DoesNotExist:
            return Response(
                {"error": f"Job {job_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # If no frames specified, return error
        if not frames:
            return Response(
                {"error": "frames list is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Get frame provider for images
        task = job.segment.task
        frame_provider = TaskFrameProvider(task)

        # Prepare images for batch prediction
        images_data = []
        for frame_idx in frames:
            try:
                frame_data = frame_provider.get_frame(
                    frame_idx,
                    quality=FrameQuality.ORIGINAL,
                )
                image_base64 = base64.b64encode(frame_data.data.getvalue()).decode(
                    "utf-8"
                )
                images_data.append(
                    {
                        "frame": frame_idx,
                        "image": image_base64,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to get frame {frame_idx}: {e}")
                continue

        if not images_data:
            return Response(
                {"error": "No valid frames to process"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Call YOLOE function
        try:
            result = invoke_yoloe_function(
                "/predict-batch",
                {
                    "job_id": int(job_id),
                    "images": images_data,
                    "threshold": threshold,
                    "output_type": output_type,
                },
            )
            return Response(result)
        except requests.HTTPError as e:
            error_msg = str(e)
            if "404" in error_msg:
                return Response(
                    {
                        "error": "No VPE found. Generate VPE first by selecting reference frames."
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )
            logger.error(f"YOLOE predict failed: {e}")
            return Response(
                {"error": f"Failed to run prediction: {error_msg}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Get VPE cache status",
        description="Check if VPE is cached for a job and get metadata.",
        parameters=[
            OpenApiParameter(
                name="job_id",
                type=int,
                required=True,
                description="Job ID",
            ),
        ],
        responses={
            200: OpenApiResponse(description="VPE cache status"),
        },
    )
    @action(detail=False, methods=["GET"], url_path="status")
    def vpe_status(self, request: ExtendedRequest) -> Response:
        """Get VPE cache status for a job."""
        job_id = request.query_params.get("job_id")
        if not job_id:
            return Response(
                {"error": "job_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_yoloe_function(
                "/status",
                {"job_id": int(job_id)},
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"YOLOE status failed: {e}")
            return Response(
                {"error": f"Failed to get status: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Clear VPE cache",
        description="Clear cached VPE for a job.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Job ID"},
                },
                "required": ["job_id"],
            }
        },
        responses={
            200: OpenApiResponse(description="VPE cleared"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="clear")
    def clear_vpe(self, request: ExtendedRequest) -> Response:
        """Clear VPE cache for a job."""
        job_id = request.data.get("job_id")
        if not job_id:
            return Response(
                {"error": "job_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = invoke_yoloe_function(
                "/clear",
                {"job_id": int(job_id)},
            )
            return Response(result)
        except requests.HTTPError as e:
            logger.error(f"YOLOE clear failed: {e}")
            return Response(
                {"error": f"Failed to clear VPE: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Apply predictions as annotations",
        description="Apply detection results to job as annotations.",
        request={
            "application/json": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "integer", "description": "Job ID"},
                    "results": {
                        "type": "array",
                        "description": "Detection results from predict endpoint",
                    },
                },
                "required": ["job_id", "results"],
            }
        },
        responses={
            200: OpenApiResponse(description="Annotations applied"),
            400: OpenApiResponse(description="Invalid request"),
        },
    )
    @action(detail=False, methods=["POST"], url_path="apply")
    def apply_predictions(self, request: ExtendedRequest) -> Response:
        """
        Apply detection results as annotations to the job.

        Converts YOLOE results to CVAT annotation format and saves.
        """
        job_id = request.data.get("job_id")
        results = request.data.get("results", [])

        if not job_id:
            return Response(
                {"error": "job_id is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not results:
            return Response(
                {"error": "results is required"},
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

        # Convert results to CVAT format
        shapes = []
        for frame_result in results:
            frame = frame_result.get("frame", 0)
            detections = frame_result.get("detections", [])

            for det in detections:
                label_name = det.get("label", "").lower()
                label_id = label_mapping.get(label_name)

                if label_id is None:
                    # Try to find by case-insensitive match
                    for name, lid in label_mapping.items():
                        if name.lower() == label_name:
                            label_id = lid
                            break

                if label_id is None:
                    logger.warning(
                        f"Label '{det.get('label')}' not found in job labels"
                    )
                    continue

                shape_type = det.get("type", "rectangle")
                points = det.get("points", [])

                shapes.append(
                    {
                        "type": shape_type,
                        "frame": frame,
                        "label_id": label_id,
                        "points": points,
                        "occluded": False,
                        "z_order": 0,
                        "source": "auto",  # Mark as auto-generated
                        "attributes": [],
                    }
                )

        if not shapes:
            return Response(
                {
                    "applied": 0,
                    "message": "No valid annotations to apply. Check that labels match.",
                }
            )

        # Apply annotations using dataset manager
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
