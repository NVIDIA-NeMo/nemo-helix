# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU detection utilities for Docker-based services.

This module provides utilities to detect available NVIDIA GPUs on the system
using the NVIDIA Management Library (NVML). Used by the GPU pool to auto-detect
available devices.
"""

import logging

logger = logging.getLogger(__name__)


def detect_gpu_device_ids() -> list[int] | None:
    """Detect available NVIDIA GPU device IDs on the system.

    Uses NVML (NVIDIA Management Library) to query available GPUs and
    returns their device IDs. Device IDs are assigned sequentially starting
    from 0.

    Returns:
        List of GPU device IDs (e.g., [0, 1, 2, 3] for a 4-GPU system),
        or None if no GPUs are available or NVML is not installed.

    Note:
        This function logs warnings but does not raise exceptions for
        expected failure cases (no NVML, no GPUs). This allows CPU-only
        systems to operate normally.
    """
    try:
        import pynvml
    except ImportError:
        logger.warning("NVML not available - GPU detection unavailable")
        return None

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        # Common errors: driver not loaded, no NVIDIA hardware
        logger.warning(f"Failed to initialize NVML: {e}")
        return None

    try:
        gpu_count = pynvml.nvmlDeviceGetCount()

        if gpu_count == 0:
            logger.info("NVML reports no GPU devices available")
            return None

        device_ids = list(range(gpu_count))
        logger.info(f"Detected {gpu_count} GPU(s): device IDs {device_ids}")
        return device_ids

    except pynvml.NVMLError as e:
        logger.warning(f"Error querying GPU count: {e}")
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass  # Ignore shutdown errors


def get_gpu_count() -> int:
    """Get the number of available GPUs.

    Returns:
        Number of GPUs detected, or 0 if none available.
    """
    device_ids = detect_gpu_device_ids()
    return len(device_ids) if device_ids else 0
