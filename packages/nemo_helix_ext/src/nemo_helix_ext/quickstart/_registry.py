# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry-host parsing shared across quickstart modules."""

from __future__ import annotations


def image_registry_host(image: str) -> str:
    """Extract the registry host from an image reference, or "" if none.

    For 2-part names like ``ubuntu/mysql:latest`` (Docker Hub namespace/repo)
    the first segment is a namespace, not a registry, so this returns "".

    For 3+ part names (``nvcr.io/nvidia/nemo-microservices/nhx-api:tag``) or
    when the first segment looks like a host (contains ``.`` or ``:``), the
    first segment is returned, **including any explicit port**. Callers that
    need to compare against a canonical name like ``"nvcr.io"`` should strip
    the port themselves (``host.split(":", 1)[0]``).
    """
    if not image or "/" not in image:
        return ""
    parts = image.split("/")
    if len(parts) >= 3:
        return parts[0]
    if len(parts) == 2 and ("." in parts[0] or ":" in parts[0]):
        return parts[0]
    return ""
