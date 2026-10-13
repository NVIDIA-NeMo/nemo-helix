# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""URL helpers for user-facing output."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def display_url(url: str) -> str:
    """Return *url* without userinfo, query, or fragment so output never echoes URL credentials."""
    try:
        parts = urlsplit(url)
    except ValueError:
        # Unvalidated input such as a custom provider URL can be malformed (e.g. "http://[::1").
        # Output is display-only, so never raise and never echo the unparsed string.
        return "<invalid URL>"
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, host, parts.path, "", "")).rstrip("/")
