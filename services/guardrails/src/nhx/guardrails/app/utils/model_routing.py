# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for Model Entity-based routing via Inference Gateway.

This module provides functions to resolve Model Entity references (i.e. "default/my-model")
to OpenAI-compatible IGW URLs. This way, guardrail configs simply need a model reference, and
routing is handled by IGW.
"""

import logging

from nemo_helix_plugin.client.adapter import client_from_platform
from nemo_helix_plugin.models.client import ModelsClient
from nhx.common.sdk_factory import get_platform_sdk
from nhx.guardrails.entities.values._private import ModelParameters, RailsConfig

logger = logging.getLogger(__name__)


def parse_model_entity_reference(model: str | None) -> tuple[str, str] | None:
    """Parse a model string as a Model Entity reference.

    Model Entity references are in format: `workspace/model_name`.

    Args:
        model: The model string to parse

    Returns:
        A tuple of (workspace, model_name) if valid, None otherwise
    """
    if not model:
        return None

    # Must contain exactly one slash ('workspace/model_name' format)
    parts = model.split("/")
    if len(parts) != 2 or not all(parts):
        return None

    return (parts[0], parts[1])


def build_openai_gateway_url(model_entity_ref: str) -> str:
    """Build IGW URL from a Model Entity reference.

    Args:
        model_entity_ref: Model Entity reference ('workspace/model_name' format).

    Returns:
        Full IGW URL (i.e. http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-)

    Raises:
        ValueError: If model_entity_ref is not in valid format
    """
    parsed = parse_model_entity_reference(model_entity_ref)
    if not parsed:
        raise ValueError(f"Invalid model entity reference: {model_entity_ref}")

    workspace, _ = parsed

    # Use SDK helper to build IGW OpenAI-compatible URL
    # IGW handles routing the request to the correct Model Provider
    sdk = get_platform_sdk()
    models = client_from_platform(sdk, ModelsClient)
    url = models.get_openai_route_base_url(workspace=workspace)

    return url


def resolve_model_entity_references(rails_config: RailsConfig) -> RailsConfig:
    """Resolve Model Entity references to base_url for all models in the given config.

    For each model with a Model Entity reference (e.g., "default/my-model"),
    this sets its `parameters.base_url` to the IGW URL.

    If `parameters.base_url` is already set, it is preserved. This allows
    users to explicitly set the URL if needed.

    Args:
        rails_config: The RailsConfig to process

    Returns:
        The RailsConfig with resolved base URLs
    """
    if not rails_config.models:
        return rails_config

    for model in rails_config.models:
        # Skip if base_url is already explicitly set
        if model.parameters and model.parameters.get("base_url"):
            logger.debug(f"Model '{model.model}' has `parameters.base_url`, skipping resolution")
            continue

        model_ref = model.model
        parsed = parse_model_entity_reference(model_ref)
        if parsed and model_ref is not None:
            # Resolve to IGW OpenAI-compatible URL
            gateway_url = build_openai_gateway_url(model_ref)

            # Set `parameters.base_url`
            parameters = ModelParameters.model_validate(model.parameters or {})
            parameters.base_url = gateway_url
            model.parameters = parameters

            logger.debug(f"Resolved model '{model_ref}' to use Inference Gateway base URL: {gateway_url}")

    return rails_config
