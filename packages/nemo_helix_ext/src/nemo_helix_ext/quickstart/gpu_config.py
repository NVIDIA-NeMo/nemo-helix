# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU configuration helpers for quickstart: detection, CUDA_VISIBLE_DEVICES filtering, and string parsing."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

CUDA_VISIBLE_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"


def parse_comma_separated_non_negative_integers(value: str) -> list[int] | None:
    """Parse a comma-separated string of non-negative integers (shared helper).

    Used by both quickstart config parsing and CUDA_VISIBLE_DEVICES parsing.
    Returns None if the string is empty/whitespace-only, has no tokens, or
    any token is not a non-negative integer.

    Args:
        value: Comma-separated string (e.g. "0,2,1" or "0, 2, 1").

    Returns:
        List of integer device IDs in order, or None if invalid.
    """
    value = value.strip()
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    result: list[int] = []
    for p in parts:
        try:
            n = int(p)
            if n < 0:
                return None
            result.append(n)
        except ValueError:
            return None
    return result


def parse_cuda_visible_devices_integers(value: str) -> list[int] | None:
    """Parse CUDA_VISIBLE_DEVICES as comma-separated integer device IDs only.

    Only integer indices are supported. If any token is not a non-negative
    integer, returns None (caller should use full detected list and optionally log).

    Args:
        value: Raw env value (e.g. "0,2,1" or "0, 2, 1").

    Returns:
        List of integer device IDs in order, or None if any token is non-integer.
    """
    if not value or not value.strip():
        return None
    return parse_comma_separated_non_negative_integers(value)


def apply_cuda_visible_devices_filter(
    detected_ids: list[int],
    *,
    log_exclusions: bool = True,
) -> list[int]:
    """Filter detected GPU IDs by CUDA_VISIBLE_DEVICES (integer indices only).

    Uses the intersection of (parsed CUDA_VISIBLE_DEVICES) and (detected_ids),
    preserving the order of CUDA_VISIBLE_DEVICES. Logs any IDs from
    CUDA_VISIBLE_DEVICES that are not in the detected set.

    If CUDA_VISIBLE_DEVICES is unset, empty, or contains non-integers, returns
    detected_ids unchanged.

    Args:
        detected_ids: Full list of autodetected GPU device IDs.
        log_exclusions: Whether to log excluded IDs (default True).

    Returns:
        Filtered list of GPU IDs (order from CUDA_VISIBLE_DEVICES when applicable).
    """
    raw = os.environ.get(CUDA_VISIBLE_DEVICES_ENV)
    if raw is None or not raw.strip():
        return list(detected_ids)

    requested = parse_cuda_visible_devices_integers(raw)
    if requested is None:
        if log_exclusions:
            logger.info(
                "CUDA_VISIBLE_DEVICES contains non-integer values; using full autodetected GPU list. "
                "Only comma-separated integer device IDs are supported."
            )
        return list(detected_ids)

    detected_set = set(detected_ids)
    # Preserve order of CUDA_VISIBLE_DEVICES; include only those in detected_set
    result = [gpu_id for gpu_id in requested if gpu_id in detected_set]
    excluded = [gpu_id for gpu_id in requested if gpu_id not in detected_set]
    if excluded and log_exclusions:
        logger.warning(
            "CUDA_VISIBLE_DEVICES listed device(s) %s which were not in autodetected set %s; excluding them.",
            excluded,
            detected_ids,
        )
    return result


def format_gpu_ids_for_storage(gpu_ids: list[int]) -> str:
    """Format a list of GPU IDs as a comma-separated string for config storage."""
    return ",".join(str(i) for i in gpu_ids)
