# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Models route-reference helpers.

These helpers mirror the convenience functions historically exported from the
Stainless-backed ``models`` package. They live in the plugin client package so
typed clients can build route references without importing generated resources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedModelReference:
    """Inference route details for a workspace-qualified model reference."""

    url: str
    name: str
    host_url: str | None
    served_model_name: str | None = None


def parse_workspace_name_ref(ref: str, *, label: str, expected_format: str = "workspace/name") -> tuple[str, str]:
    """Parse a strict workspace-qualified reference."""
    workspace, separator, name = ref.partition("/")
    if separator != "/" or not workspace or not name or "/" in name:
        raise ValueError(f"{label} must be in format '{expected_format}'")
    return workspace, name


def first_provider_ref(model_providers: list[str] | None) -> tuple[str, str, str] | None:
    """Return the first valid ``(ref, workspace, name)`` provider reference, if present."""
    if not model_providers:
        return None

    provider_ref = model_providers[0]
    try:
        provider_workspace, provider_name = parse_workspace_name_ref(provider_ref, label="Provider reference")
    except ValueError:
        _logger.warning("Invalid provider reference format", extra={"provider_ref": provider_ref})
        return None
    return provider_ref, provider_workspace, provider_name


def model_entity_route_openai_url(*, base_url: str, workspace: str, name: str) -> str:
    """OpenAI SDK-compatible URL for a model-entity proxy route."""
    return f"{base_url.rstrip('/')}/apis/inference-gateway/v2/workspaces/{workspace}/model/{name}/-/v1"


def resolved_model_reference(
    *,
    base_url: str,
    name: str,
    route_workspace: str,
    route_model_name: str,
    host_url: str | None,
    served_model_name: str | None = None,
) -> ResolvedModelReference:
    """Build route details for a resolved model entity."""
    return ResolvedModelReference(
        url=model_entity_route_openai_url(base_url=base_url, workspace=route_workspace, name=route_model_name),
        name=name,
        host_url=host_url,
        served_model_name=served_model_name,
    )


def warn_provider_host_url_resolution_failure(
    provider_ref: str,
    exc: Exception,
    *,
    not_found_error_type: type[Exception],
) -> None:
    """Log a provider host-url lookup failure with the expected severity."""
    if isinstance(exc, not_found_error_type):
        _logger.warning("Provider not found during host_url resolution", extra={"provider_ref": provider_ref})
        return
    _logger.warning("Failed to resolve provider host_url", extra={"provider_ref": provider_ref}, exc_info=True)
