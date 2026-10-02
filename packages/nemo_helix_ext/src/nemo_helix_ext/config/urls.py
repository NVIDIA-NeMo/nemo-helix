# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL helpers for user-facing output."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def display_url(url: str) -> str:
    """Return *url* without userinfo, query, or fragment so output never echoes URL credentials."""
    parts = urlsplit(url)
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, host, parts.path, "", "")).rstrip("/")
