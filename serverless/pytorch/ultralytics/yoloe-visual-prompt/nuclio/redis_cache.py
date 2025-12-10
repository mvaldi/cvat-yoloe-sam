# Copyright (C) CVAT.ai Corporation
# SPDX-License-Identifier: MIT

"""
Redis Cache Handler for YOLOE Visual Prompt Embeddings

Manages storage and retrieval of VPE with TTL (Time To Live) support.
- TTL: 30 days by default
- Auto-renewal: TTL is renewed on each access
- Key format: yoloe:vpe:{job_id}
"""

import json
import os
import pickle
import time
from typing import Optional

import redis


class VPECacheHandler:
    """Redis cache handler for Visual Prompt Embeddings."""

    # Default TTL: 30 days in seconds
    DEFAULT_TTL_DAYS = 30
    DEFAULT_TTL_SECONDS = DEFAULT_TTL_DAYS * 24 * 60 * 60  # 2592000 seconds

    # Key prefix
    KEY_PREFIX = "yoloe:vpe"

    def __init__(
        self,
        host: str = None,
        port: int = None,
        ttl_days: int = None,
    ):
        """
        Initialize Redis connection.

        Args:
            host: Redis host (default from env REDIS_HOST)
            port: Redis port (default from env REDIS_PORT)
            ttl_days: TTL in days (default from env VPE_TTL_DAYS or 30)
        """
        self.host = host or os.environ.get("REDIS_HOST", "cvat_redis_ondisk")
        self.port = port or int(os.environ.get("REDIS_PORT", 6666))
        self.ttl_days = ttl_days or int(
            os.environ.get("VPE_TTL_DAYS", self.DEFAULT_TTL_DAYS)
        )
        self.ttl_seconds = self.ttl_days * 24 * 60 * 60

        self.redis_client = redis.Redis(
            host=self.host,
            port=self.port,
            decode_responses=False,  # We store binary data
        )

        # Test connection
        try:
            self.redis_client.ping()
            print(f"Redis connected: {self.host}:{self.port}")
        except redis.ConnectionError as e:
            print(f"Warning: Redis connection failed: {e}")

    def _get_key(self, job_id: int) -> str:
        """Generate Redis key for a job."""
        return f"{self.KEY_PREFIX}:{job_id}"

    def _get_metadata_key(self, job_id: int) -> str:
        """Generate Redis key for metadata."""
        return f"{self.KEY_PREFIX}:{job_id}:metadata"

    def store_vpe(
        self,
        job_id: int,
        vpe_data: dict,
        reference_frames: list[int],
        class_names: list[str],
    ) -> bool:
        """
        Store VPE data in Redis with TTL.

        Args:
            job_id: CVAT Job ID
            vpe_data: Dictionary containing serialized VPE and metadata
            reference_frames: List of frame indices used as references
            class_names: List of class names

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            key = self._get_key(job_id)
            metadata_key = self._get_metadata_key(job_id)

            # Store VPE data
            self.redis_client.setex(
                key,
                self.ttl_seconds,
                pickle.dumps(vpe_data),
            )

            # Store metadata separately (for quick access without loading VPE)
            metadata = {
                "job_id": job_id,
                "reference_frames": reference_frames,
                "class_names": class_names,
                "num_references": len(reference_frames),
                "total_annotations": vpe_data.get("total_annotations", 0),
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            self.redis_client.setex(
                metadata_key,
                self.ttl_seconds,
                json.dumps(metadata),
            )

            print(
                f"VPE stored for job {job_id}: {len(reference_frames)} refs, TTL={self.ttl_days}d"
            )
            return True

        except Exception as e:
            print(f"Error storing VPE for job {job_id}: {e}")
            return False

    def get_vpe(self, job_id: int, renew_ttl: bool = True) -> Optional[dict]:
        """
        Retrieve VPE data from Redis and optionally renew TTL.

        Args:
            job_id: CVAT Job ID
            renew_ttl: Whether to renew TTL on access (default True)

        Returns:
            VPE data dictionary or None if not found
        """
        try:
            key = self._get_key(job_id)
            metadata_key = self._get_metadata_key(job_id)

            data = self.redis_client.get(key)
            if data is None:
                return None

            # Renew TTL on access
            if renew_ttl:
                self.redis_client.expire(key, self.ttl_seconds)
                self.redis_client.expire(metadata_key, self.ttl_seconds)
                # Update metadata timestamp
                metadata_raw = self.redis_client.get(metadata_key)
                if metadata_raw:
                    metadata = json.loads(metadata_raw)
                    metadata["updated_at"] = time.time()
                    self.redis_client.setex(
                        metadata_key,
                        self.ttl_seconds,
                        json.dumps(metadata),
                    )

            return pickle.loads(data)

        except Exception as e:
            print(f"Error retrieving VPE for job {job_id}: {e}")
            return None

    def get_metadata(self, job_id: int) -> Optional[dict]:
        """
        Get VPE metadata without loading full VPE.

        Args:
            job_id: CVAT Job ID

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            metadata_key = self._get_metadata_key(job_id)
            data = self.redis_client.get(metadata_key)
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            print(f"Error retrieving metadata for job {job_id}: {e}")
            return None

    def update_references(
        self,
        job_id: int,
        vpe_data: dict,
        reference_frames: list[int],
        class_names: list[str],
    ) -> bool:
        """
        Update VPE with new/modified references.
        This replaces the existing VPE entirely.

        Args:
            job_id: CVAT Job ID
            vpe_data: New VPE data
            reference_frames: Updated list of reference frames
            class_names: Updated class names

        Returns:
            True if updated successfully
        """
        return self.store_vpe(job_id, vpe_data, reference_frames, class_names)

    def remove_reference(self, job_id: int, frame_index: int) -> bool:
        """
        Mark a reference frame for removal.
        Note: Actual VPE regeneration must be done by the model handler.

        Args:
            job_id: CVAT Job ID
            frame_index: Frame index to remove

        Returns:
            True if metadata updated, False otherwise
        """
        try:
            metadata = self.get_metadata(job_id)
            if metadata is None:
                return False

            if frame_index in metadata["reference_frames"]:
                metadata["reference_frames"].remove(frame_index)
                metadata["updated_at"] = time.time()

                metadata_key = self._get_metadata_key(job_id)
                self.redis_client.setex(
                    metadata_key,
                    self.ttl_seconds,
                    json.dumps(metadata),
                )
                return True

            return False

        except Exception as e:
            print(f"Error removing reference for job {job_id}: {e}")
            return False

    def delete_vpe(self, job_id: int) -> bool:
        """
        Delete VPE data for a job.

        Args:
            job_id: CVAT Job ID

        Returns:
            True if deleted, False otherwise
        """
        try:
            key = self._get_key(job_id)
            metadata_key = self._get_metadata_key(job_id)

            self.redis_client.delete(key)
            self.redis_client.delete(metadata_key)
            print(f"VPE deleted for job {job_id}")
            return True

        except Exception as e:
            print(f"Error deleting VPE for job {job_id}: {e}")
            return False

    def get_ttl(self, job_id: int) -> int:
        """
        Get remaining TTL in seconds for a job's VPE.

        Args:
            job_id: CVAT Job ID

        Returns:
            TTL in seconds, -1 if key doesn't expire, -2 if key doesn't exist
        """
        try:
            key = self._get_key(job_id)
            return self.redis_client.ttl(key)
        except Exception as e:
            print(f"Error getting TTL for job {job_id}: {e}")
            return -2

    def exists(self, job_id: int) -> bool:
        """
        Check if VPE exists for a job.

        Args:
            job_id: CVAT Job ID

        Returns:
            True if exists, False otherwise
        """
        try:
            key = self._get_key(job_id)
            return self.redis_client.exists(key) > 0
        except Exception as e:
            print(f"Error checking existence for job {job_id}: {e}")
            return False
