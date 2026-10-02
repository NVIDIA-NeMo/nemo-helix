# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for parsing git remote URLs (HTTPS and SSH forms)."""

from __future__ import annotations

from urllib.parse import urlparse


def git_remote_host(url: str) -> str:
    """Return the hostname from a git remote URL, or "" if unparseable.

    Handles schemed URLs (``https://github.com/org/repo``,
    ``ssh://git@github.com/org/repo``) and SCP-style alt form, both with
    (``git@github.com:org/repo``) and without (``github.com:org/repo``) a
    user prefix. Windows-style absolute paths (``C:\\repo``) are rejected.
    """
    if "://" in url:
        return (urlparse(url).hostname or "").lower()
    if ":" in url:
        lhs, _, rhs = url.partition(":")
        # SCP form requires a non-empty repo path after the colon and a
        # hostname-shaped lhs. Require a dot in the host (after stripping any
        # ``user@`` prefix) so Windows drive letters like ``C:\repo`` don't
        # match.
        if rhs and "/" not in lhs:
            host_only = lhs.split("@", 1)[-1]
            if "." in host_only:
                return host_only.lower()
    return ""
