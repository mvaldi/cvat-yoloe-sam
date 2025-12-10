# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

"""
YOLOE Visual Prompt - Nuclio HTTP Handler

Endpoints:
- POST /generate-vpe: Generate VPE from reference images with annotations
- POST /predict: Detect objects using cached VPE
- POST /predict-batch: Detect objects in multiple images
- GET /status: Get VPE cache status for a job
- DELETE /clear: Clear VPE cache for a job
- Standard detector endpoint for CVAT compatibility
"""

import base64
import io
import json
import os

import numpy as np
from model_handler import YOLOEVisualPromptHandler
from PIL import Image
from redis_cache import VPECacheHandler


def init_context(context):
    """Initialize handlers on container start."""
    context.logger.info("Init YOLOE Visual Prompt context... 0%")

    # Initialize model handler
    model_path = os.environ.get("MODEL_PATH", "/opt/nuclio/yoloe-11l-seg.pt")
    context.logger.info(f"Loading YOLOE model from: {model_path}")
    context.user_data.model = YOLOEVisualPromptHandler(model_path)

    # Initialize Redis cache handler
    context.user_data.cache = VPECacheHandler()

    context.logger.info("Init YOLOE Visual Prompt context... 100%")


def handler(context, event):
    """Main HTTP handler routing requests to appropriate endpoints."""

    # Get request path and method
    path = event.path if hasattr(event, "path") else ""
    method = event.method if hasattr(event, "method") else "POST"

    context.logger.info(f"Request: {method} {path}")

    # Route to appropriate handler
    if path == "/generate-vpe" or path.endswith("/generate-vpe"):
        return handle_generate_vpe(context, event)
    elif path == "/predict" or path.endswith("/predict"):
        return handle_predict(context, event)
    elif path == "/predict-batch" or path.endswith("/predict-batch"):
        return handle_predict_batch(context, event)
    elif path == "/status" or path.endswith("/status"):
        return handle_status(context, event)
    elif path == "/clear" or path.endswith("/clear"):
        return handle_clear(context, event)
    else:
        # Default: standard CVAT detector endpoint
        # This allows the function to work as a regular detector too
        return handle_standard_detect(context, event)


def handle_generate_vpe(context, event):
    """
    Generate Visual Prompt Embeddings from reference images.

    Expected body:
    {
        "job_id": 123,
        "references": [
            {
                "frame": 0,
                "image": "<base64>",
                "annotations": [
                    {"bbox": [x1,y1,x2,y2], "label": "class_name"},
                    ...
                ]
            },
            ...
        ]
    }

    Response:
    {
        "success": true,
        "job_id": 123,
        "num_references": 5,
        "total_annotations": 100,
        "class_names": ["class1", "class2", ...],
        "ttl_days": 30
    }
    """
    try:
        data = event.body
        if isinstance(data, bytes):
            data = json.loads(data.decode("utf-8"))

        job_id = data.get("job_id")
        references = data.get("references", [])

        if not job_id:
            return error_response(context, "job_id is required", 400)

        if not references:
            return error_response(context, "At least one reference is required", 400)

        if len(references) > YOLOEVisualPromptHandler.MAX_REFERENCES:
            return error_response(
                context,
                f"Maximum {YOLOEVisualPromptHandler.MAX_REFERENCES} references allowed",
                400,
            )

        # Parse references - build per-image data structure
        parsed_references = []
        class_name_set = set()
        reference_frames = []

        for ref in references:
            # Decode image
            img_data = base64.b64decode(ref["image"])
            img = Image.open(io.BytesIO(img_data)).convert("RGB")

            # Parse annotations for this image
            annotations = ref.get("annotations", [])
            frame_bboxes = []
            frame_labels = []

            for ann in annotations:
                bbox = ann["bbox"]  # [x1, y1, x2, y2]
                label = ann["label"]
                class_name_set.add(label)
                frame_bboxes.append(bbox)
                frame_labels.append(label)

            parsed_references.append(
                {
                    "image": img,
                    "bboxes": frame_bboxes,
                    "labels": frame_labels,  # Will convert to indices below
                }
            )
            reference_frames.append(ref.get("frame", len(reference_frames)))

        # Convert class names to indices
        class_names = sorted(list(class_name_set))
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        # Convert labels to class indices in each reference
        for ref in parsed_references:
            ref["cls"] = [class_to_idx[label] for label in ref["labels"]]

        # Generate VPE
        context.logger.info(
            f"Generating VPE for job {job_id}: {len(parsed_references)} images, {len(class_names)} classes"
        )

        vpe_data = context.user_data.model.generate_vpe(
            references=parsed_references,
            class_names=class_names,
        )

        # Store in Redis
        success = context.user_data.cache.store_vpe(
            job_id=job_id,
            vpe_data=vpe_data,
            reference_frames=reference_frames,
            class_names=class_names,
        )

        if not success:
            return error_response(context, "Failed to store VPE in cache", 500)

        return context.Response(
            body=json.dumps(
                {
                    "success": True,
                    "job_id": job_id,
                    "num_references": vpe_data["num_references"],
                    "total_annotations": vpe_data["total_annotations"],
                    "class_names": class_names,
                    "ttl_days": context.user_data.cache.ttl_days,
                }
            ),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"Error in generate_vpe: {e}")
        return error_response(context, str(e), 500)


def handle_predict(context, event):
    """
    Predict objects in a single image using cached VPE.

    Expected body:
    {
        "job_id": 123,
        "image": "<base64>",
        "threshold": 0.25,
        "output_type": "rectangle"  // or "polygon", "obb"
    }

    Response: List of detections in CVAT format
    """
    try:
        data = event.body
        if isinstance(data, bytes):
            data = json.loads(data.decode("utf-8"))

        job_id = data.get("job_id")
        threshold = float(data.get("threshold", 0.25))
        output_type = data.get("output_type", "rectangle")

        if not job_id:
            return error_response(context, "job_id is required", 400)

        # Get VPE from cache (this also renews TTL)
        vpe_data = context.user_data.cache.get_vpe(job_id)
        if vpe_data is None:
            return error_response(
                context,
                f"No VPE found for job {job_id}. Generate VPE first using /generate-vpe",
                404,
            )

        # Decode image
        img_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Run prediction
        results = context.user_data.model.predict(
            image=image,
            vpe_data=vpe_data,
            threshold=threshold,
            output_type=output_type,
        )

        context.logger.info(f"Prediction for job {job_id}: {len(results)} detections")

        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"Error in predict: {e}")
        return error_response(context, str(e), 500)


def handle_predict_batch(context, event):
    """
    Predict objects in multiple images using cached VPE.

    Expected body:
    {
        "job_id": 123,
        "images": [
            {"frame": 10, "image": "<base64>"},
            {"frame": 11, "image": "<base64>"},
            ...
        ],
        "threshold": 0.25,
        "output_type": "rectangle"
    }

    Response:
    {
        "results": [
            {"frame": 10, "detections": [...]},
            {"frame": 11, "detections": [...]},
            ...
        ]
    }
    """
    try:
        data = event.body
        if isinstance(data, bytes):
            data = json.loads(data.decode("utf-8"))

        job_id = data.get("job_id")
        images_data = data.get("images", [])
        threshold = float(data.get("threshold", 0.25))
        output_type = data.get("output_type", "rectangle")

        if not job_id:
            return error_response(context, "job_id is required", 400)

        if not images_data:
            return error_response(context, "At least one image is required", 400)

        # Get VPE from cache
        vpe_data = context.user_data.cache.get_vpe(job_id)
        if vpe_data is None:
            return error_response(
                context, f"No VPE found for job {job_id}. Generate VPE first.", 404
            )

        # Process each image
        results = []
        for img_info in images_data:
            frame = img_info.get("frame", len(results))
            img_data = base64.b64decode(img_info["image"])
            image = Image.open(io.BytesIO(img_data)).convert("RGB")

            detections = context.user_data.model.predict(
                image=image,
                vpe_data=vpe_data,
                threshold=threshold,
                output_type=output_type,
            )

            results.append(
                {
                    "frame": frame,
                    "detections": detections,
                }
            )

        context.logger.info(f"Batch prediction for job {job_id}: {len(results)} frames")

        return context.Response(
            body=json.dumps({"results": results}),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"Error in predict_batch: {e}")
        return error_response(context, str(e), 500)


def handle_status(context, event):
    """
    Get VPE cache status for a job.

    Expected query params: ?job_id=123

    Response:
    {
        "exists": true,
        "job_id": 123,
        "reference_frames": [0, 1, 2],
        "class_names": ["class1", "class2"],
        "num_references": 3,
        "total_annotations": 60,
        "ttl_remaining_days": 29.5,
        "created_at": "2025-12-09T...",
        "updated_at": "2025-12-09T..."
    }
    """
    try:
        # Parse job_id from query params or body
        data = event.body
        if isinstance(data, bytes):
            try:
                data = json.loads(data.decode("utf-8"))
            except Exception:
                data = {}

        job_id = data.get("job_id")

        if not job_id:
            return error_response(context, "job_id is required", 400)

        # Check if VPE exists
        if not context.user_data.cache.exists(job_id):
            return context.Response(
                body=json.dumps(
                    {
                        "exists": False,
                        "job_id": job_id,
                    }
                ),
                headers={},
                content_type="application/json",
                status_code=200,
            )

        # Get metadata
        metadata = context.user_data.cache.get_metadata(job_id)
        ttl_seconds = context.user_data.cache.get_ttl(job_id)
        ttl_days = ttl_seconds / (24 * 60 * 60) if ttl_seconds > 0 else 0

        response_data = {
            "exists": True,
            "job_id": job_id,
            "ttl_remaining_days": round(ttl_days, 2),
        }

        if metadata:
            response_data.update(
                {
                    "reference_frames": metadata.get("reference_frames", []),
                    "class_names": metadata.get("class_names", []),
                    "num_references": metadata.get("num_references", 0),
                    "total_annotations": metadata.get("total_annotations", 0),
                    "created_at": metadata.get("created_at"),
                    "updated_at": metadata.get("updated_at"),
                }
            )

        return context.Response(
            body=json.dumps(response_data),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"Error in status: {e}")
        return error_response(context, str(e), 500)


def handle_clear(context, event):
    """
    Clear VPE cache for a job.

    Expected body:
    {
        "job_id": 123
    }

    Response:
    {
        "success": true,
        "job_id": 123
    }
    """
    try:
        data = event.body
        if isinstance(data, bytes):
            data = json.loads(data.decode("utf-8"))

        job_id = data.get("job_id")

        if not job_id:
            return error_response(context, "job_id is required", 400)

        success = context.user_data.cache.delete_vpe(job_id)

        return context.Response(
            body=json.dumps(
                {
                    "success": success,
                    "job_id": job_id,
                }
            ),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    except Exception as e:
        context.logger.error(f"Error in clear: {e}")
        return error_response(context, str(e), 500)


def handle_standard_detect(context, event):
    """
    Standard CVAT detector endpoint.

    YOLOE Visual Prompt funciona de la siguiente manera:
    1. Si hay VPE cacheado para el job, lo usa directamente
    2. Si no hay VPE pero se proporcionan referencias, las genera
    3. Si no hay nada, retorna mensaje de error con instrucciones

    Expected body from CVAT:
    {
        "image": "<base64>",
        "threshold": 0.5,
        "job": 123,  // CVAT envía job_id como "job"
        "frame": 10,
        // Opcionales para Visual Prompting:
        "references": [...],  // Referencias si se quieren agregar
        "output_type": "rectangle"
    }
    """
    try:
        data = event.body
        if isinstance(data, bytes):
            data = json.loads(data.decode("utf-8"))

        # CVAT envía el job_id como "job", no "job_id"
        job_id = data.get("job") or data.get("job_id")
        threshold = float(data.get("threshold", 0.25))
        output_type = data.get("output_type", "rectangle")
        frame = data.get("frame", 0)

        context.logger.info(
            f"Standard detect: job={job_id}, frame={frame}, threshold={threshold}"
        )

        # Decode image
        img_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Si hay job_id, intentar usar VPE cacheado
        if job_id:
            vpe_data = context.user_data.cache.get_vpe(job_id)
            if vpe_data:
                context.logger.info(f"Using cached VPE for job {job_id}")
                results = context.user_data.model.predict(
                    image=image,
                    vpe_data=vpe_data,
                    threshold=threshold,
                    output_type=output_type,
                )

                # Formato CVAT detector response
                return context.Response(
                    body=json.dumps(results),
                    headers={},
                    content_type="application/json",
                    status_code=200,
                )
            else:
                context.logger.info(f"No VPE cached for job {job_id}")

        # Si se proporcionan referencias inline, generar VPE primero
        if data.get("references"):
            context.logger.info("Generating VPE from inline references")
            # Procesar referencias (mismo código que handle_generate_vpe)
            references = data["references"]
            images_ref = []
            bboxes_list = []
            classes_list = []
            class_name_set = set()
            reference_frames = []

            for ref in references:
                img_ref_data = base64.b64decode(ref["image"])
                img_ref = Image.open(io.BytesIO(img_ref_data)).convert("RGB")
                images_ref.append(img_ref)

                annotations = ref.get("annotations", [])
                frame_bboxes = []
                frame_classes = []

                for ann in annotations:
                    bbox = ann["bbox"]
                    label = ann["label"]
                    class_name_set.add(label)
                    frame_bboxes.append(bbox)
                    frame_classes.append(label)

                bboxes_list.append(np.array(frame_bboxes, dtype=np.float32))
                classes_list.append(frame_classes)
                reference_frames.append(ref.get("frame", len(reference_frames)))

            # Convert class names to indices
            class_names = sorted(list(class_name_set))
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}

            classes_list_indexed = []
            for frame_classes in classes_list:
                indexed = np.array(
                    [class_to_idx[name] for name in frame_classes], dtype=np.int32
                )
                classes_list_indexed.append(indexed)

            # Generate VPE
            vpe_data = context.user_data.model.generate_vpe(
                reference_images=images_ref,
                reference_bboxes=bboxes_list,
                reference_classes=classes_list_indexed,
                class_names=class_names,
            )

            # Store in Redis if job_id available
            if job_id:
                context.user_data.cache.store_vpe(
                    job_id=job_id,
                    vpe_data=vpe_data,
                    reference_frames=reference_frames,
                    class_names=class_names,
                )

            # Now predict
            results = context.user_data.model.predict(
                image=image,
                vpe_data=vpe_data,
                threshold=threshold,
                output_type=output_type,
            )

            return context.Response(
                body=json.dumps(results),
                headers={},
                content_type="application/json",
                status_code=200,
            )

        # No VPE available y no hay referencias - retornar error informativo
        error_msg = (
            "YOLOE Visual Prompt requiere referencias visuales. "
            "Por favor primero genere VPE usando el endpoint /generate-vpe "
            "con frames anotados como referencia, o proporcione 'references' en la solicitud."
        )

        if job_id:
            error_msg += f" (Job ID: {job_id})"

        return error_response(context, error_msg, 400)

    except Exception as e:
        context.logger.error(f"Error in standard_detect: {e}")
        return error_response(context, str(e), 500)


def error_response(context, message: str, status_code: int = 500):
    """Return an error response."""
    return context.Response(
        body=json.dumps({"error": message}),
        headers={},
        content_type="application/json",
        status_code=status_code,
    )
