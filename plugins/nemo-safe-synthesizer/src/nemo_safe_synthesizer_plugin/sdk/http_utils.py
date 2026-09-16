# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP helpers for the Safe Synthesizer plugin SDK."""

from __future__ import annotations

from urllib.parse import quote, urljoin

from nemo_helix import AsyncNeMoHelix, NeMoHelix

PlatformClient = NeMoHelix | AsyncNeMoHelix

_API_PREFIX = "/apis/safe-synthesizer"


def base_url(source: str) -> str:
    """Return the normalized base URL for a raw URL string."""
    return source.rstrip("/")


def resolve_workspace(platform: PlatformClient, workspace: str | None, *, strict: bool = False) -> str:
    """Return the explicit, platform, or default workspace for Safe Synthesizer routes."""
    resolved = workspace or platform.workspace
    if resolved is None:
        if strict:
            raise ValueError("workspace must be provided when the client has no default workspace")
        return "default"
    return resolved


def url(platform: PlatformClient, path: str, workspace: str | None = None) -> str:
    """Build a full Safe Synthesizer plugin API URL for the provided route path."""
    resolved_path = path.format(workspace=resolve_workspace(platform, workspace))
    return _join_url(str(platform.base_url), f"{_API_PREFIX}/{resolved_path}")


def platform_default_headers(platform: PlatformClient) -> dict[str, str]:
    """Return string-valued default platform headers for direct plugin HTTP calls."""
    return {str(key): value for key, value in platform.default_headers.items() if isinstance(value, str)}


def job_route_base_url(*, raw_base_url: str, workspace: str, job_name: str) -> str:
    """Build the stable Safe Synthesizer plugin URL prefix for one created job."""
    encoded_workspace = quote(workspace, safe="")
    encoded_job_name = quote(job_name, safe="")
    return _join_url(raw_base_url, f"{_API_PREFIX}/v2/workspaces/{encoded_workspace}/jobs/{encoded_job_name}")


def job_route_resource_url(*, job_base_url: str, resource_path: str) -> str:
    """Build a full URL below a stable Safe Synthesizer job route."""
    return _join_url(job_base_url, resource_path)


def _join_url(root: str, relative_path: str) -> str:
    """Join a root URL and a relative path using URL parsing rules."""
    return urljoin(f"{base_url(root)}/", relative_path.lstrip("/"))
