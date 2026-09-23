# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for composing HTTP requests against the platform client."""

from nemo_helix import AsyncNeMoHelix, NeMoHelix

HelixClient = NeMoHelix | AsyncNeMoHelix

_API_PREFIX = "/apis/data-designer/v2/workspaces"


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


def url(platform: HelixClient, workspace: str | None, path: str) -> str:
    normalized_path = f"/{path.lstrip('/')}"
    return f"{base_url(platform)}{_API_PREFIX}/{resolve_workspace(platform, workspace)}{normalized_path}"
