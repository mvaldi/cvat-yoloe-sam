# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

"""
YOLOE Visual Prompt Model Handler

This module handles YOLOE model operations for visual prompting:
- Generating Visual Prompt Embeddings (VPE) from reference images with annotations
- Detecting objects in target images using cached VPE
- Post-processing for different output types (bbox, polygon, obb)
"""

import pickle
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


class YOLOEVisualPromptHandler:
    """YOLOE Visual Prompting model handler."""

    # Maximum reference images allowed
    MAX_REFERENCES = 50

    # Output types
    OUTPUT_BBOX = "rectangle"
    OUTPUT_POLYGON = "polygon"
    OUTPUT_OBB = "obb"  # Oriented Bounding Box (4 points)

    def __init__(self, model_path: str = "/opt/nuclio/yoloe-11l-seg.pt"):
        """
        Initialize the YOLOE model.

        Args:
            model_path: Path to the YOLOE .pt model file (segmentation version)
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the YOLOE model from disk."""
        try:
            self.model = YOLOE(self.model_path)
            # Warm-up inference
            dummy_img = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
            _ = self.model.predict(dummy_img, verbose=False)
            print(f"YOLOE model loaded successfully from {self.model_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLOE model from {self.model_path}: {e}"
            )

    def generate_vpe(
        self,
        references: list[dict],
        class_names: list[str],
    ) -> dict:
        """
        Generate Visual Prompt Embeddings from reference images with annotations.

        Since YOLOE only supports one reference image at a time, we process each
        image separately and average the embeddings for each class.

        Args:
            references: List of dicts with keys:
                - 'image': PIL Image
                - 'bboxes': List of [x1,y1,x2,y2] bboxes
                - 'cls': List of class indices for each bbox
            class_names: List of unique class names

        Returns:
            Dictionary containing:
            - vpe: Serialized Visual Prompt Embeddings
            - class_names: List of class names
            - num_references: Number of reference images used
            - total_annotations: Total number of annotations
        """
        import torch

        if len(references) > self.MAX_REFERENCES:
            raise ValueError(
                f"Maximum {self.MAX_REFERENCES} reference images allowed, got {len(references)}"
            )

        num_classes = len(class_names)
        total_annotations = 0

        # Collect embeddings for each class from all reference images
        class_embeddings = {i: [] for i in range(num_classes)}

        for ref in references:
            img = ref["image"]
            bboxes = ref["bboxes"]
            cls_list = ref["cls"]

            if not bboxes:
                continue

            total_annotations += len(bboxes)

            # Prepare visual prompts for this image
            visual_prompts = {"bboxes": bboxes, "cls": cls_list}

            # Reset predictor for fresh VPE extraction
            self.model.predictor = None

            # Generate embeddings for this image
            self.model.predict(
                img,
                visual_prompts=visual_prompts,
                refer_image=[img],
                predictor=YOLOEVPSegPredictor,
                verbose=False,
            )

            # Get embeddings - shape [1, num_unique_classes_in_image, 512]
            pe = self.model.model.pe.cpu()

            # Map embeddings to class indices
            # The PE tensor has one embedding per unique class in cls_list
            unique_cls_in_image = sorted(set(cls_list))
            for i, cls_idx in enumerate(unique_cls_in_image):
                if i < pe.shape[1]:
                    class_embeddings[cls_idx].append(pe[0, i, :])

        # Average embeddings for each class
        final_pe = torch.zeros(1, num_classes, 512)
        for cls_idx in range(num_classes):
            if class_embeddings[cls_idx]:
                stacked = torch.stack(class_embeddings[cls_idx])
                final_pe[0, cls_idx, :] = stacked.mean(dim=0)

        # Serialize VPE for storage
        vpe_serialized = pickle.dumps(final_pe)

        return {
            "vpe": vpe_serialized,
            "class_names": class_names,
            "num_references": len(references),
            "total_annotations": total_annotations,
        }

    def predict(
        self,
        image: Image.Image,
        vpe_data: dict,
        threshold: float = 0.25,
        output_type: str = "rectangle",
    ) -> list:
        """
        Run detection on an image using cached VPE.

        Args:
            image: PIL Image to run inference on
            vpe_data: Dictionary with 'vpe' (serialized) and 'class_names'
            threshold: Confidence threshold for detections
            output_type: One of 'rectangle', 'polygon', 'obb'

        Returns:
            List of detections in CVAT format
        """
        # Deserialize VPE
        vpe = pickle.loads(vpe_data["vpe"])
        class_names = vpe_data["class_names"]

        # Reset predictor and set classes with VPE
        self.model.predictor = None
        self.model.set_classes(class_names, vpe)

        # Run prediction
        predictions = self.model.predict(image, conf=threshold, verbose=False)

        # Process results based on output type
        results = []
        w, h = image.size

        for pred in predictions:
            if pred.boxes is None:
                continue

            boxes = pred.boxes
            masks = (
                pred.masks
                if hasattr(pred, "masks") and pred.masks is not None
                else None
            )

            for i in range(len(boxes)):
                conf = float(boxes.conf[i].cpu().numpy())
                cls_idx = int(boxes.cls[i].cpu().numpy())
                label = (
                    class_names[cls_idx]
                    if cls_idx < len(class_names)
                    else f"class_{cls_idx}"
                )

                if output_type == self.OUTPUT_POLYGON and masks is not None:
                    # Get polygon from mask
                    result = self._mask_to_polygon(masks, i, label, conf, w, h)
                elif output_type == self.OUTPUT_OBB and masks is not None:
                    # Get OBB from mask using minAreaRect
                    result = self._mask_to_obb(masks, i, label, conf, w, h)
                else:
                    # Default: bounding box
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    result = self._bbox_to_cvat(xyxy, label, conf, w, h)

                if result:
                    results.append(result)

        return results

    def _bbox_to_cvat(
        self,
        xyxy: np.ndarray,
        label: str,
        conf: float,
        img_w: int,
        img_h: int,
    ) -> dict:
        """Convert bbox to CVAT rectangle format."""
        x1 = max(int(xyxy[0]), 0)
        y1 = max(int(xyxy[1]), 0)
        x2 = min(int(xyxy[2]), img_w)
        y2 = min(int(xyxy[3]), img_h)

        return {
            "confidence": str(round(conf, 4)),
            "label": label,
            "points": [x1, y1, x2, y2],
            "type": "rectangle",
        }

    def _mask_to_polygon(
        self,
        masks,
        idx: int,
        label: str,
        conf: float,
        img_w: int,
        img_h: int,
    ) -> Optional[dict]:
        """Convert segmentation mask to CVAT polygon format."""
        try:
            # Get mask data
            mask_data = masks.data[idx].cpu().numpy()

            # Resize mask to image size if needed
            if mask_data.shape != (img_h, img_w):
                mask_data = cv2.resize(
                    mask_data, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                )

            # Convert to uint8
            mask_uint8 = (mask_data * 255).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Simplify contour to reduce points
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Flatten points
            points = simplified.reshape(-1).tolist()

            # Need at least 3 points for a polygon
            if len(points) < 6:
                return None

            return {
                "confidence": str(round(conf, 4)),
                "label": label,
                "points": points,
                "type": "polygon",
            }
        except Exception as e:
            print(f"Error converting mask to polygon: {e}")
            return None

    def _mask_to_obb(
        self,
        masks,
        idx: int,
        label: str,
        conf: float,
        img_w: int,
        img_h: int,
    ) -> Optional[dict]:
        """
        Convert segmentation mask to Oriented Bounding Box (OBB) format.
        Uses cv2.minAreaRect to get the minimum area rotated rectangle.
        Returns 4 points (8 coordinates) in CVAT polygon format.
        """
        try:
            # Get mask data
            mask_data = masks.data[idx].cpu().numpy()

            # Resize mask to image size if needed
            if mask_data.shape != (img_h, img_w):
                mask_data = cv2.resize(
                    mask_data, (img_w, img_h), interpolation=cv2.INTER_NEAREST
                )

            # Convert to uint8
            mask_uint8 = (mask_data * 255).astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            # Get all points from all contours
            all_points = np.vstack(contours)

            # Get minimum area rotated rectangle
            rect = cv2.minAreaRect(all_points)
            box_points = cv2.boxPoints(rect)

            # Clamp points to image bounds
            box_points[:, 0] = np.clip(box_points[:, 0], 0, img_w)
            box_points[:, 1] = np.clip(box_points[:, 1], 0, img_h)

            # Convert to integer and flatten
            points = box_points.astype(int).reshape(-1).tolist()

            return {
                "confidence": str(round(conf, 4)),
                "label": label,
                "points": points,
                "type": "polygon",  # OBB is represented as a 4-point polygon in CVAT
            }
        except Exception as e:
            print(f"Error converting mask to OBB: {e}")
            return None

    def predict_batch(
        self,
        images: list[Image.Image],
        vpe_data: dict,
        threshold: float = 0.25,
        output_type: str = "rectangle",
    ) -> list[list]:
        """
        Run detection on multiple images using cached VPE.

        Args:
            images: List of PIL Images
            vpe_data: Dictionary with 'vpe' (serialized) and 'class_names'
            threshold: Confidence threshold
            output_type: Output format type

        Returns:
            List of results for each image
        """
        return [self.predict(img, vpe_data, threshold, output_type) for img in images]
