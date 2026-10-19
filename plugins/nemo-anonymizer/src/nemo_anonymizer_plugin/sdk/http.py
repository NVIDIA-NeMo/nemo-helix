# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP helpers for the Anonymizer plugin SDK resources."""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from urllib.parse import quote

import httpx


@runtime_checkable
class HelixClient(Protocol):
    """Minimal structural surface the Anonymizer SDK needs from a platform client.

    Satisfied by ``NemoClient`` / ``AsyncNemoClient`` (and the legacy generated
    SDK) so resource code can read the base URL, default headers, and workspace
    without depending on any concrete client class.
    """

    @property
    def base_url(self) -> str | httpx.URL: ...
    @property
    def default_headers(self) -> dict[str, str]: ...
    @property
    def workspace(self) -> str | None: ...


_API_PREFIX = "/apis/anonymizer/v2/workspaces"


def base_url(platform: HelixClient) -> str:
    return str(platform.base_url).rstrip("/")


def headers(platform: HelixClient) -> dict[str, str]:
    return {k: v for k, v in platform.default_headers.items() if isinstance(v, str)}


def resolve_workspace(platform: HelixClient, workspace: str | None) -> str:
    resolved = workspace or platform.workspace
    if not resolved:
        raise ValueError(
            "No workspace specified. Pass `workspace=...` to the method or set `workspace` on the platform client."
        )
    return resolved


def path_segment(value: str) -> str:
    """Encode an untrusted value for use as one URL path segment."""
    if not value:
        raise ValueError("URL path segment cannot be empty.")
    return quote(value, safe="")


def url(platform: HelixClient, workspace: str | None, path: str) -> str:
    if not path.startswith("/"):
        raise ValueError("SDK URL path must start with '/'.")
    workspace_segment = path_segment(resolve_workspace(platform, workspace))
    return f"{base_url(platform)}{_API_PREFIX}/{workspace_segment}{path}"
