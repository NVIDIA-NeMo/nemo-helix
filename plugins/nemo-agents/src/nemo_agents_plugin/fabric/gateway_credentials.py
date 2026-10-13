# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transient placeholder credentials for Platform-routed Fabric models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PLATFORM_IGW_PATH_MARKER = "/apis/inference-gateway/"
PLATFORM_IGW_API_KEY_ENV = "NEMO_AGENTS_IGW_API_KEY"
PLATFORM_IGW_API_KEY_PLACEHOLDER = "not-used"


@dataclass(frozen=True, slots=True)
class HelixGatewayCredentialBinding:
    """Runtime-only credential binding required by a Fabric model adapter."""

    api_key_env: str
    value: str = PLATFORM_IGW_API_KEY_PLACEHOLDER


def resolve_platform_gateway_credential_binding(
    config: Mapping[str, Any],
) -> HelixGatewayCredentialBinding | None:
    """Return the placeholder binding for the selected IGW-routed model."""
    model = _selected_model_config(config)
    if model is None or not _is_platform_gateway_model(model):
        return None

    api_key_env = model.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env:
        api_key_env = PLATFORM_IGW_API_KEY_ENV
    return HelixGatewayCredentialBinding(api_key_env=api_key_env)


def platform_gateway_credential_env(config: Mapping[str, Any]) -> dict[str, str]:
    """Return child-process environment values for the selected Fabric model."""
    binding = resolve_platform_gateway_credential_binding(config)
    if binding is None:
        return {}
    return {binding.api_key_env: binding.value}


def bind_platform_gateway_model_credential(model: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a translated model payload and add its runtime-only key reference."""
    resolved = dict(model)
    if not _is_platform_gateway_model(resolved):
        return resolved
    api_key_env = resolved.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env:
        resolved["api_key_env"] = PLATFORM_IGW_API_KEY_ENV
    return resolved


def _selected_model_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    harnesses = config.get("harnesses")
    default_harness = config.get("default_harness")
    if isinstance(harnesses, Mapping) and isinstance(default_harness, str):
        harness = harnesses.get(default_harness)
        if isinstance(harness, Mapping):
            model = harness.get("model")
            if isinstance(model, Mapping):
                return model

    models = config.get("models")
    if not isinstance(models, Mapping):
        return None
    model = models.get("default")
    return model if isinstance(model, Mapping) else None


def _is_platform_gateway_model(model: Mapping[str, Any]) -> bool:
    base_url = model.get("base_url")
    if isinstance(base_url, str):
        return PLATFORM_IGW_PATH_MARKER in base_url

    settings = model.get("settings")
    if not isinstance(settings, Mapping):
        return False
    legacy_base_url = settings.get("base_url")
    return isinstance(legacy_base_url, str) and PLATFORM_IGW_PATH_MARKER in legacy_base_url
