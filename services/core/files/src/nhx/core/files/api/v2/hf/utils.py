# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for HuggingFace compatibility layer."""

import hashlib
from datetime import datetime


def generate_commit_hash(fileset_name: str, updated_at: datetime | None) -> str:
    """
    Generate a stable fake commit hash for a fileset.

    Uses SHA-1 to match Git's 40-character hex format.
    The hash is deterministic based on fileset ID and last update time.
    """
    timestamp = updated_at.isoformat() if updated_at else "unknown"
    data = f"{fileset_name}:{timestamp}"
    return hashlib.sha1(data.encode()).hexdigest()


def generate_etag(fileset_name: str, path: str, size: int) -> str:
    """
    Generate a stable ETag for a file.

    Returns a quoted string suitable for HTTP ETag header.
    The ETag is deterministic based on fileset ID, file path, and size.
    """
    data = f"{fileset_name}:{path}:{size}"
    hash_value = hashlib.sha256(data.encode()).hexdigest()[:16]
    return f'"{hash_value}"'
