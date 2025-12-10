# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

"""
SAM3 Model Handler for CVAT

Uses official SAM3 (Segment Anything Model 3) from HuggingFace transformers.

SAM3 provides multiple specialized models:
- Sam3Model + Sam3Processor: Text-to-Segment (PCS - Promptable Concept Segmentation)
- Sam3TrackerModel + Sam3TrackerProcessor: Point/Box-to-Segment (PVS - Promptable Visual Segmentation)
- Sam3VideoModel + Sam3VideoProcessor: Text-to-Track (PCS for video)
- Sam3TrackerVideoModel + Sam3TrackerVideoProcessor: Visual tracking in video

Capabilities:
- Text-to-Segment: Segment objects using natural language prompts (270K+ concepts)
- Box-to-Segment: Segment using bounding box as visual reference
- Point-to-Segment: Segment using click points
- Text-to-Detect: Detect all instances matching a text prompt
- Text-to-Track: Video tracking with text initialization
- Refine-Box: Adjust bounding box to fit object precisely

Redis cache for session state (TTL: 1 day)
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import redis
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration from environment
REDIS_HOST = os.environ.get("REDIS_HOST", "cvat_redis_ondisk")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6666"))
REDIS_TTL = int(os.environ.get("REDIS_TTL", "86400"))  # 1 day

# HuggingFace token for gated models
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Try to read token from file if not in environment
if not HF_TOKEN:
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            HF_TOKEN = f.read().strip()


class RedisCache:
    """Redis cache manager for embeddings and tracking state."""

    def __init__(
        self, host: str = REDIS_HOST, port: int = REDIS_PORT, ttl: int = REDIS_TTL
    ):
        self.ttl = ttl
        try:
            self.client = redis.Redis(host=host, port=port, decode_responses=False)
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
            self.client = None
            self._memory_cache = {}

    def _make_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key."""
        return f"sam3:{prefix}:{identifier}"

    def get(self, prefix: str, identifier: str) -> Optional[Any]:
        """Get value from cache."""
        key = self._make_key(prefix, identifier)
        if self.client:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
        elif hasattr(self, "_memory_cache") and key in self._memory_cache:
            return self._memory_cache[key]
        return None

    def set(self, prefix: str, identifier: str, value: Any) -> bool:
        """Set value in cache with TTL."""
        key = self._make_key(prefix, identifier)
        try:
            data = pickle.dumps(value)
            if self.client:
                self.client.setex(key, self.ttl, data)
            else:
                if not hasattr(self, "_memory_cache"):
                    self._memory_cache = {}
                self._memory_cache[key] = value
            return True
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            return False

    def delete(self, prefix: str, identifier: str) -> bool:
        """Delete value from cache."""
        key = self._make_key(prefix, identifier)
        try:
            if self.client:
                self.client.delete(key)
            elif hasattr(self, "_memory_cache") and key in self._memory_cache:
                del self._memory_cache[key]
            return True
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False

    def exists(self, prefix: str, identifier: str) -> bool:
        """Check if key exists."""
        key = self._make_key(prefix, identifier)
        if self.client:
            return self.client.exists(key) > 0
        return hasattr(self, "_memory_cache") and key in self._memory_cache


class ModelHandler:
    """
    SAM3 Model Handler with full capabilities.

    Uses official SAM3 models from HuggingFace transformers:
    - Sam3Model: Text-to-Segment (PCS for images)
    - Sam3TrackerModel: Point/Box-to-Segment (PVS for images)
    - Sam3VideoModel: Text-to-Track (PCS for video)

    All models use "facebook/sam3" as the base.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing SAM3 on device: {self.device}")

        # Initialize Redis cache
        self.cache = RedisCache()

        # Model references (lazy loaded)
        self._sam3_model = None
        self._sam3_processor = None
        self._sam3_tracker_model = None
        self._sam3_tracker_processor = None
        self._sam3_video_model = None
        self._sam3_video_processor = None

        # Model ID
        self._model_id = "facebook/sam3"

        # Initialize primary model (Sam3Model for text-based segmentation)
        self._init_sam3_model()

        logger.info("SAM3 initialized successfully")

    def _init_sam3_model(self):
        """Initialize Sam3Model for text-based segmentation (PCS)."""
        try:
            from transformers import Sam3Model, Sam3Processor

            logger.info(f"Loading Sam3Model from {self._model_id}")

            # Use token if available (for gated models)
            load_kwargs = {}
            if HF_TOKEN:
                load_kwargs["token"] = HF_TOKEN
                logger.info("Using HuggingFace token for authentication")

            self._sam3_model = Sam3Model.from_pretrained(
                self._model_id, **load_kwargs
            ).to(self.device)
            self._sam3_model.eval()

            self._sam3_processor = Sam3Processor.from_pretrained(
                self._model_id, **load_kwargs
            )

            logger.info("Sam3Model (PCS) loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Sam3Model: {e}")
            import traceback

            traceback.print_exc()

    def _init_sam3_tracker_model(self):
        """Initialize Sam3TrackerModel for point/box-based segmentation (PVS)."""
        if self._sam3_tracker_model is not None:
            return

        try:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            logger.info(f"Loading Sam3TrackerModel from {self._model_id}")

            # Use token if available (for gated models)
            load_kwargs = {}
            if HF_TOKEN:
                load_kwargs["token"] = HF_TOKEN

            self._sam3_tracker_model = Sam3TrackerModel.from_pretrained(
                self._model_id, **load_kwargs
            ).to(self.device)
            self._sam3_tracker_model.eval()

            self._sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(
                self._model_id, **load_kwargs
            )

            logger.info("Sam3TrackerModel (PVS) loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Sam3TrackerModel: {e}")
            import traceback

            traceback.print_exc()

    def _init_sam3_video_model(self):
        """Initialize Sam3VideoModel for video tracking with text (PCS video)."""
        if self._sam3_video_model is not None:
            return

        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor

            logger.info(f"Loading Sam3VideoModel from {self._model_id}")

            # Use token if available (for gated models)
            load_kwargs = {"torch_dtype": torch.bfloat16}
            if HF_TOKEN:
                load_kwargs["token"] = HF_TOKEN

            self._sam3_video_model = Sam3VideoModel.from_pretrained(
                self._model_id, **load_kwargs
            ).to(self.device)
            self._sam3_video_model.eval()

            processor_kwargs = {}
            if HF_TOKEN:
                processor_kwargs["token"] = HF_TOKEN

            self._sam3_video_processor = Sam3VideoProcessor.from_pretrained(
                self._model_id, **processor_kwargs
            )

            logger.info("Sam3VideoModel (PCS video) loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Sam3VideoModel: {e}")
            import traceback

            traceback.print_exc()

    def _extract_mask_polygon(self, mask: np.ndarray) -> List[List[int]]:
        """Extract polygon points from binary mask."""
        if mask is None or mask.sum() == 0:
            return []

        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest, epsilon=1.0, closed=True)
        polygon = [[int(p[0][0]), int(p[0][1])] for p in approx]

        return polygon

    def _mask_to_bounds(self, mask: np.ndarray) -> Optional[List[int]]:
        """Get bounding box from mask [x1, y1, x2, y2]."""
        if mask is None or mask.sum() == 0:
            return None

        coords = np.where(mask > 0.5)
        if len(coords[0]) == 0:
            return None

        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())
        return [x_min, y_min, x_max, y_max]

    # ==================== TEXT-TO-SEGMENT ====================

    def segment_with_text(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.1,
        mask_threshold: float = 0.5,
        job_id: Optional[int] = None,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Segment objects using text prompt (Text-to-Segment / PCS).

        Uses Sam3Model with Sam3Processor for text-based segmentation.
        Supports 270K+ concepts.

        Args:
            image: PIL Image to segment
            text_prompt: Natural language description (e.g., "person", "car", "coffee cup")
            threshold: Detection confidence threshold (default 0.1)
            mask_threshold: Mask binarization threshold (default 0.5)
            job_id: Optional job ID for caching
            frame_idx: Optional frame index for caching

        Returns:
            dict with masks, boxes, scores, labels, polygons
        """
        if self._sam3_model is None or self._sam3_processor is None:
            return {
                "masks": [],
                "boxes": [],
                "scores": [],
                "labels": [],
                "polygons": [],
            }

        try:
            logger.info(
                f"Segmenting with text: '{text_prompt}' (threshold={threshold})"
            )

            # Process image with text prompt
            inputs = self._sam3_processor(
                images=image,
                text=text_prompt,
                return_tensors="pt",
            ).to(self.device)

            # Run model
            with torch.no_grad():
                outputs = self._sam3_model(**inputs)

            # Post-process results
            results = self._sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            # Extract results
            masks = []
            boxes = []
            scores = []
            labels = []
            polygons = []

            if "masks" in results and len(results["masks"]) > 0:
                for i in range(len(results["masks"])):
                    mask_tensor = results["masks"][i]
                    mask_np = (
                        mask_tensor.cpu().numpy()
                        if torch.is_tensor(mask_tensor)
                        else np.array(mask_tensor)
                    )

                    # Ensure 2D mask
                    if len(mask_np.shape) > 2:
                        mask_np = mask_np.squeeze()

                    score = float(results["scores"][i]) if "scores" in results else 1.0
                    box = (
                        results["boxes"][i]
                        if "boxes" in results
                        else self._mask_to_bounds(mask_np)
                    )

                    if torch.is_tensor(box):
                        box = box.tolist()

                    masks.append(mask_np.tolist())
                    boxes.append(box)
                    scores.append(score)
                    labels.append(text_prompt)
                    polygons.append(self._extract_mask_polygon(mask_np))

            logger.info(f"Found {len(masks)} objects matching '{text_prompt}'")

            return {
                "masks": masks,
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "polygons": polygons,
            }

        except Exception as e:
            logger.error(f"segment_with_text failed: {e}")
            import traceback

            traceback.print_exc()
            return {
                "masks": [],
                "boxes": [],
                "scores": [],
                "labels": [],
                "polygons": [],
            }

    # ==================== BOX-TO-SEGMENT ====================

    def segment_with_box(
        self,
        image: Image.Image,
        box: List[float],
        text_prompt: Optional[str] = None,
        job_id: Optional[int] = None,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Segment object inside bounding box (Box-to-Segment).

        Uses Sam3TrackerModel with a box prompt to segment the object
        within the bounding box.

        Args:
            image: PIL Image
            box: Bounding box [x1, y1, x2, y2] in pixel coordinates
            text_prompt: Optional text hint (not used with tracker model)
            job_id: Optional job ID for caching
            frame_idx: Optional frame index for caching

        Returns:
            dict with mask, bounds, polygon, score
        """
        # Initialize tracker model if needed
        self._init_sam3_tracker_model()

        if self._sam3_tracker_model is None or self._sam3_tracker_processor is None:
            return {"mask": [], "bounds": None, "polygon": [], "score": 0.0}

        try:
            logger.info(f"Segmenting with box: {box}")

            # Format box for processor: [[[x1, y1, x2, y2]]]
            # Dimensions: [batch, num_boxes, 4]
            input_boxes = [[[int(c) for c in box]]]

            # Process inputs
            inputs = self._sam3_tracker_processor(
                images=image,
                input_boxes=input_boxes,
                return_tensors="pt",
            ).to(self._sam3_tracker_model.device)

            # Run model
            with torch.no_grad():
                outputs = self._sam3_tracker_model(**inputs)

            # Post-process masks
            masks = self._sam3_tracker_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
            )[0]

            # Get the best mask (highest score)
            if masks.shape[1] > 1:
                # Multiple mask predictions - use scores to select best
                scores = (
                    outputs.iou_scores.cpu().numpy().squeeze()
                    if hasattr(outputs, "iou_scores")
                    else None
                )
                if scores is not None and len(scores) > 0:
                    best_idx = np.argmax(scores)
                else:
                    best_idx = 0
            else:
                best_idx = 0

            # Extract mask
            mask_tensor = masks[0, best_idx] if masks.shape[0] > 0 else masks[best_idx]
            mask_np = (
                mask_tensor.numpy()
                if torch.is_tensor(mask_tensor)
                else np.array(mask_tensor)
            )
            mask_np = (mask_np > 0.5).astype(np.uint8)

            bounds = self._mask_to_bounds(mask_np)
            polygon = self._extract_mask_polygon(mask_np)
            score = (
                float(outputs.iou_scores[0, best_idx])
                if hasattr(outputs, "iou_scores")
                else 0.9
            )

            return {
                "mask": mask_np.tolist(),
                "bounds": bounds if bounds else box,
                "polygon": polygon,
                "score": score,
            }

        except Exception as e:
            logger.error(f"segment_with_box failed: {e}")
            import traceback

            traceback.print_exc()
            return {"mask": [], "bounds": box, "polygon": [], "score": 0.0}

    # ==================== POINT-TO-SEGMENT ====================

    def segment_with_points(
        self,
        image: Image.Image,
        pos_points: List[List[float]],
        neg_points: Optional[List[List[float]]] = None,
        box: Optional[List[float]] = None,
        job_id: Optional[int] = None,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Segment object using click points (Point-to-Segment / Refine with Clicks).

        Uses Sam3TrackerModel with point prompts for interactive segmentation.
        Positive points indicate the object, negative points indicate background.

        Args:
            image: PIL Image
            pos_points: List of positive points [[x, y], ...]
            neg_points: Optional list of negative points [[x, y], ...]
            box: Optional bounding box to combine with points
            job_id: Optional job ID for caching
            frame_idx: Optional frame index for caching

        Returns:
            dict with mask, bounds, polygon, score
        """
        # Initialize tracker model if needed
        self._init_sam3_tracker_model()

        if self._sam3_tracker_model is None or self._sam3_tracker_processor is None:
            return {"mask": [], "bounds": None, "polygon": [], "score": 0.0}

        try:
            logger.info(
                f"Segmenting with {len(pos_points)} positive, {len(neg_points or [])} negative points"
            )

            # Format points for processor
            # Dimensions: [batch, num_objects, num_points, 2]
            all_points = []
            all_labels = []

            for pt in pos_points:
                all_points.append([float(pt[0]), float(pt[1])])
                all_labels.append(1)  # Positive

            for pt in neg_points or []:
                all_points.append([float(pt[0]), float(pt[1])])
                all_labels.append(0)  # Negative

            # Format: [[[[x, y], ...]]]
            input_points = [[[all_points]]]
            input_labels = [[[all_labels]]]

            # Flatten to correct dimensions
            input_points = [[all_points]]  # [batch, points, 2]
            input_labels = [[all_labels]]  # [batch, labels]

            # Process inputs
            processor_kwargs = {
                "images": image,
                "input_points": input_points,
                "input_labels": input_labels,
                "return_tensors": "pt",
            }

            # Add box if provided
            if box is not None:
                processor_kwargs["input_boxes"] = [[[int(c) for c in box]]]

            inputs = self._sam3_tracker_processor(**processor_kwargs).to(
                self._sam3_tracker_model.device
            )

            # Run model
            with torch.no_grad():
                outputs = self._sam3_tracker_model(**inputs)

            # Post-process masks
            masks = self._sam3_tracker_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
            )[0]

            # Select best mask
            best_idx = 0
            if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
                scores = outputs.iou_scores.cpu().numpy().squeeze()
                if len(scores.shape) > 0 and len(scores) > 0:
                    best_idx = np.argmax(scores)

            # Extract mask
            if len(masks.shape) > 2:
                mask_tensor = (
                    masks[0, best_idx] if masks.shape[0] > 0 else masks[best_idx]
                )
            else:
                mask_tensor = masks
            mask_np = (
                mask_tensor.numpy()
                if torch.is_tensor(mask_tensor)
                else np.array(mask_tensor)
            )
            mask_np = (mask_np > 0.5).astype(np.uint8)

            bounds = self._mask_to_bounds(mask_np)
            polygon = self._extract_mask_polygon(mask_np)
            score = (
                float(outputs.iou_scores[0, best_idx])
                if hasattr(outputs, "iou_scores")
                else 0.9
            )

            return {
                "mask": mask_np.tolist(),
                "bounds": bounds,
                "polygon": polygon,
                "score": score,
            }

        except Exception as e:
            logger.error(f"segment_with_points failed: {e}")
            import traceback

            traceback.print_exc()
            return {"mask": [], "bounds": None, "polygon": [], "score": 0.0}

    # ==================== REFINE BOX ====================

    def refine_box(
        self,
        image: Image.Image,
        box: List[float],
        text_prompt: Optional[str] = None,
        job_id: Optional[int] = None,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Refine a bounding box to fit the object precisely.

        Segments the object inside the box and returns a tight bounding box
        that fits the actual object boundaries.

        Args:
            image: PIL Image
            box: Initial bounding box [x1, y1, x2, y2]
            text_prompt: Optional text hint for the object
            job_id: Optional job ID for caching
            frame_idx: Optional frame index for caching

        Returns:
            dict with refined_box, mask, polygon, score
        """
        try:
            logger.info(f"Refining box: {box}")

            # Segment with the box
            result = self.segment_with_box(image, box, text_prompt, job_id, frame_idx)

            if result["bounds"] is not None:
                return {
                    "refined_box": result["bounds"],
                    "mask": result["mask"],
                    "polygon": result["polygon"],
                    "score": result["score"],
                }
            else:
                return {
                    "refined_box": box,
                    "mask": result.get("mask", []),
                    "polygon": result.get("polygon", []),
                    "score": result.get("score", 0.0),
                }

        except Exception as e:
            logger.error(f"refine_box failed: {e}")
            return {"refined_box": box, "mask": [], "polygon": [], "score": 0.0}

    # ==================== TEXT-TO-DETECT ====================

    def detect_all(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.1,
        job_id: Optional[int] = None,
        frame_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Detect all instances of objects matching text prompt (Text-to-Detect).

        Uses Sam3Model to find all instances of the specified concept.
        Returns bounding boxes for all detected objects.

        Args:
            image: PIL Image
            text_prompt: Description of objects to find
            threshold: Detection confidence threshold
            job_id: Optional job ID for caching
            frame_idx: Optional frame index for caching

        Returns:
            dict with boxes, scores, labels
        """
        try:
            logger.info(f"Detecting all '{text_prompt}' (threshold={threshold})")

            # Use segment_with_text which gives us everything
            result = self.segment_with_text(
                image, text_prompt, threshold, 0.5, job_id, frame_idx
            )

            return {
                "boxes": result["boxes"],
                "scores": result["scores"],
                "labels": result["labels"],
                "masks": result["masks"],
                "polygons": result["polygons"],
            }

        except Exception as e:
            logger.error(f"detect_all failed: {e}")
            return {
                "boxes": [],
                "scores": [],
                "labels": [],
                "masks": [],
                "polygons": [],
            }

    # ==================== VIDEO TRACKING ====================

    def init_tracking(
        self,
        frames: List[Image.Image],
        text_prompt: str,
        job_id: int,
        start_frame: int = 0,
    ) -> Dict[str, Any]:
        """
        Initialize video tracking with text prompt (Text-to-Track).

        Uses Sam3VideoModel to detect and track objects across video frames.

        Args:
            frames: List of PIL Images (video frames)
            text_prompt: Description of objects to track
            job_id: Job ID for caching session state
            start_frame: Frame index to start tracking from

        Returns:
            dict with tracking results for processed frames
        """
        # Initialize video model if needed
        self._init_sam3_video_model()

        if self._sam3_video_model is None or self._sam3_video_processor is None:
            return {
                "frames": {},
                "object_ids": [],
                "error": "Video model not available",
            }

        try:
            logger.info(
                f"Initializing tracking for '{text_prompt}' on {len(frames)} frames"
            )

            # Convert frames to numpy arrays
            frame_arrays = [np.array(f) for f in frames]

            # Initialize video inference session
            inference_session = self._sam3_video_processor.init_video_session(
                video=frame_arrays,
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )

            # Add text prompt
            inference_session = self._sam3_video_processor.add_text_prompt(
                inference_session=inference_session,
                text=text_prompt,
            )

            # Process all frames
            outputs_per_frame = {}
            object_ids = []

            for model_outputs in self._sam3_video_model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=len(frames),
            ):
                processed = self._sam3_video_processor.postprocess_outputs(
                    inference_session, model_outputs
                )

                frame_idx = model_outputs.frame_idx

                # Extract results for this frame
                frame_data = {
                    "masks": [],
                    "boxes": [],
                    "scores": [],
                    "object_ids": [],
                }

                if processed["masks"] is not None and len(processed["masks"]) > 0:
                    for i in range(len(processed["masks"])):
                        mask = processed["masks"][i]
                        mask_np = (
                            mask.cpu().numpy()
                            if torch.is_tensor(mask)
                            else np.array(mask)
                        )
                        if len(mask_np.shape) > 2:
                            mask_np = mask_np.squeeze()

                        frame_data["masks"].append(mask_np.tolist())
                        frame_data["boxes"].append(
                            processed["boxes"][i].tolist()
                            if torch.is_tensor(processed["boxes"][i])
                            else list(processed["boxes"][i])
                        )
                        frame_data["scores"].append(float(processed["scores"][i]))
                        frame_data["object_ids"].append(int(processed["object_ids"][i]))

                        # Track unique object IDs
                        obj_id = int(processed["object_ids"][i])
                        if obj_id not in object_ids:
                            object_ids.append(obj_id)

                outputs_per_frame[frame_idx] = frame_data

            # Cache the session for future track_frame calls
            session_data = {
                "text_prompt": text_prompt,
                "num_frames": len(frames),
                "object_ids": object_ids,
            }
            self.cache.set("tracking_session", str(job_id), session_data)

            logger.info(
                f"Tracking initialized: {len(outputs_per_frame)} frames, {len(object_ids)} objects"
            )

            return {
                "frames": outputs_per_frame,
                "object_ids": object_ids,
                "label": text_prompt,
            }

        except Exception as e:
            logger.error(f"init_tracking failed: {e}")
            import traceback

            traceback.print_exc()
            return {"frames": {}, "object_ids": [], "error": str(e)}

    def track_frame(
        self,
        image: Image.Image,
        job_id: int,
        frame_idx: int,
        prev_masks: Optional[List[np.ndarray]] = None,
        object_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Track objects in a single frame (streaming mode).

        For streaming video tracking where frames arrive one at a time.

        Args:
            image: PIL Image (current frame)
            job_id: Job ID for session retrieval
            frame_idx: Current frame index
            prev_masks: Previous frame masks for continuity
            object_ids: Object IDs to track

        Returns:
            dict with masks, boxes, scores for this frame
        """
        # Initialize video model if needed
        self._init_sam3_video_model()

        if self._sam3_video_model is None or self._sam3_video_processor is None:
            return {"masks": [], "boxes": [], "scores": [], "object_ids": []}

        try:
            logger.info(f"Tracking frame {frame_idx} for job {job_id}")

            # Get cached session info
            session_data = self.cache.get("tracking_session", str(job_id))

            if session_data is None:
                return {
                    "masks": [],
                    "boxes": [],
                    "scores": [],
                    "object_ids": [],
                    "error": "No tracking session",
                }

            # For streaming, we need to reinitialize with the current frame
            # This is a simplified implementation - full streaming would maintain state
            inference_session = self._sam3_video_processor.init_video_session(
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.bfloat16,
            )

            # Add text prompt from cached session
            inference_session = self._sam3_video_processor.add_text_prompt(
                inference_session=inference_session,
                text=session_data["text_prompt"],
            )

            # Process the frame
            inputs = self._sam3_video_processor(
                images=image, device=self.device, return_tensors="pt"
            )

            model_outputs = self._sam3_video_model(
                inference_session=inference_session,
                frame=inputs.pixel_values[0],
                reverse=False,
            )

            processed = self._sam3_video_processor.postprocess_outputs(
                inference_session,
                model_outputs,
                original_sizes=inputs.original_sizes,
            )

            # Extract results
            result = {
                "masks": [],
                "boxes": [],
                "scores": [],
                "object_ids": [],
            }

            if processed["masks"] is not None and len(processed["masks"]) > 0:
                for i in range(len(processed["masks"])):
                    mask = processed["masks"][i]
                    mask_np = (
                        mask.cpu().numpy() if torch.is_tensor(mask) else np.array(mask)
                    )
                    if len(mask_np.shape) > 2:
                        mask_np = mask_np.squeeze()

                    result["masks"].append(mask_np.tolist())
                    result["boxes"].append(
                        processed["boxes"][i].tolist()
                        if torch.is_tensor(processed["boxes"][i])
                        else list(processed["boxes"][i])
                    )
                    result["scores"].append(float(processed["scores"][i]))
                    result["object_ids"].append(int(processed["object_ids"][i]))

            return result

        except Exception as e:
            logger.error(f"track_frame failed: {e}")
            import traceback

            traceback.print_exc()
            return {
                "masks": [],
                "boxes": [],
                "scores": [],
                "object_ids": [],
                "error": str(e),
            }

    # ==================== UTILITY METHODS ====================

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "sam3_model": self._sam3_model is not None,
            "sam3_tracker_model": self._sam3_tracker_model is not None,
            "sam3_video_model": self._sam3_video_model is not None,
            "device": str(self.device),
            "model_id": self._model_id,
            "capabilities": [
                "text_to_segment",
                "box_to_segment",
                "point_to_segment",
                "refine_box",
                "text_to_detect",
                "text_to_track",
            ],
        }
