# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared resource manager for NeMo Helix controllers.

This module provides a singleton manager for resources that need to be shared
across multiple controllers running in the same process, such as the GPU pool.
"""

import logging
import threading
from typing import Optional, Self

from nhx.common.config import get_platform_config
from nhx.common.docker.gpu_detection import detect_gpu_device_ids
from nhx.common.docker.gpu_pool import DockerGPUPool

logger = logging.getLogger(__name__)


class SharedResourceManager:
    """Singleton manager for shared resources across controllers.

    This class manages resources that need to be shared between the jobs and
    models controllers when running in the same process. The primary use case
    is the GPU pool, which must be shared to prevent both controllers from
    allocating the same GPU to different workloads.

    Usage:
        manager = SharedResourceManager.get_instance()
        gpu_pool = manager.get_gpu_pool()
        if gpu_pool:
            gpu_ids = gpu_pool.allocate_gpu("my-workload", num_requested=1)

    Thread Safety:
        All methods are thread-safe. The singleton instance and GPU pool
        creation are protected by locks.
    """

    _instance: Optional[Self] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        """Private constructor. Use get_instance() to get the singleton instance."""
        if SharedResourceManager._instance is not None:
            raise RuntimeError("Use SharedResourceManager.get_instance() to get the singleton instance")
        self._gpu_pool: DockerGPUPool | None = None
        self._gpu_pool_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "SharedResourceManager":
        """Get the singleton instance of SharedResourceManager.

        Returns:
            The singleton SharedResourceManager instance.
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls.__new__(cls)
                cls._instance._gpu_pool = None
                cls._instance._gpu_pool_lock = threading.Lock()
        return cls._instance

    def get_gpu_pool(self) -> DockerGPUPool | None:
        """Get or create the shared GPU pool based on configuration.

        The GPU pool is lazily initialized on first access based on
        platform.docker.reserved_gpu_device_ids:
        - "all": Auto-detect all available GPUs
        - "none" or "": Explicitly disable GPU support
        - Comma-separated list: Use specific GPU device IDs (e.g., "0,1,2,3")

        Returns:
            The shared DockerGPUPool instance, or None if no GPUs are
            configured or detected.
        """
        with self._gpu_pool_lock:
            if self._gpu_pool is None:
                config = get_platform_config()
                reserved_ids = config.docker.get_reserved_gpu_ids()

                if reserved_ids is None:
                    # "all" - auto-detect GPUs
                    detected_gpus = detect_gpu_device_ids()

                    if detected_gpus is None or len(detected_gpus) == 0:
                        logger.info("SharedResourceManager: No GPUs detected on system")
                        return None

                    logger.info(
                        f"SharedResourceManager: Creating GPU pool with all {len(detected_gpus)} "
                        f"detected GPU(s): {detected_gpus}"
                    )
                    self._gpu_pool = DockerGPUPool(reserved_gpu_device_ids=detected_gpus)
                else:
                    # Explicit list of GPU IDs (or empty list if "none"/""
                    if not reserved_ids:
                        logger.warning(
                            "SharedResourceManager: GPU support explicitly disabled via "
                            "reserved_gpu_device_ids configuration. GPU workloads will not be available."
                        )
                        return None

                    # Validate user-specified GPU IDs against what's actually available.
                    # Filter to only use IDs that exist on the system to prevent Docker
                    # failures when trying to allocate non-existent GPUs.
                    detected_gpus = detect_gpu_device_ids()
                    if detected_gpus is not None:
                        detected_set = set(detected_gpus)
                        valid_ids = [gpu_id for gpu_id in reserved_ids if gpu_id in detected_set]
                        invalid_ids = [gpu_id for gpu_id in reserved_ids if gpu_id not in detected_set]

                        if invalid_ids:
                            logger.warning(
                                f"SharedResourceManager: Configured GPU IDs {invalid_ids} not found on system. "
                                f"Available GPUs: {detected_gpus}. Using only valid IDs: {valid_ids}"
                            )

                        if not valid_ids:
                            logger.warning(
                                "SharedResourceManager: None of the configured GPU IDs exist on this system. "
                                "GPU workloads will not be available."
                            )
                            return None

                        reserved_ids = valid_ids
                    else:
                        # GPU detection not available or no GPUs detected - trust user config
                        # and let Docker handle any errors at container creation time
                        logger.warning(
                            "SharedResourceManager: Could not detect GPUs to validate configuration. "
                            f"Trusting configured GPU IDs: {reserved_ids}"
                        )

                    logger.info(
                        f"SharedResourceManager: Creating GPU pool with {len(reserved_ids)} "
                        f"configured GPU(s): {reserved_ids}"
                    )
                    self._gpu_pool = DockerGPUPool(reserved_gpu_device_ids=reserved_ids)

            return self._gpu_pool

    def clear(self) -> None:
        """Clear all managed resources.

        This method is primarily useful for testing purposes to reset state
        between tests.
        """
        with self._gpu_pool_lock:
            if self._gpu_pool is not None:
                logger.debug("SharedResourceManager: Clearing GPU pool")
            self._gpu_pool = None

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        This method is primarily useful for testing purposes to ensure
        a fresh instance is created.
        """
        with cls._instance_lock:
            cls._instance = None
