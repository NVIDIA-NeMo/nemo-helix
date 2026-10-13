# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for LLM clients (ChatNIM and NIM)."""

import os
from typing import Dict, Optional

from nhx.guardrails.app.constants import FALLBACK_DEFAULT_ENDPOINT_URL
from nhx.guardrails.app.utils.context_utils import get_main_model_from_context, get_x_model_auth_token_from_context
from pydantic import SecretStr

DEFAULT_PROVIDER_NAME = os.getenv("DEFAULT_LLM_PROVIDER", "nim")


def get_main_model_api_key() -> Optional[SecretStr]:
    """Returns API key to use to authenticate with the main model.

    The API key is resolved from the X-Model-Authorization header in the incoming request.
    NOTE: If the model is configured to use Inference Gateway, credentials are instead set via Inference Gateway.

    Returns:
        SecretStr if a key is found, None otherwise
    """
    auth_token = get_x_model_auth_token_from_context()
    if auth_token:
        return SecretStr(auth_token)

    return None


def get_provider_from_context() -> str:
    """Get the provider from the main model in context.

    Returns:
        The engine string from the main model, or "nim" as the default
    """
    main_model = get_main_model_from_context()
    if main_model is None:
        return DEFAULT_PROVIDER_NAME

    engine = main_model.engine
    return "nim" if "nim" in engine else engine


def determine_main_model_base_url(values: Dict) -> str:
    """Returns base URL to use for inference requests with the main model.

    The order of priority:
    1. `parameters.base_url` for the main model in the GuardrailConfig.
    2. `NIM_ENDPOINT_URL` environment variable, which falls back to `FALLBACK_DEFAULT_ENDPOINT_URL` if not set.

    Args:
        values: Dictionary containing endpoint_url or other configuration values.

    Returns:
        The base URL to use for inference requests.
    """
    inference_base_url = values.get("endpoint_url", None)

    # By default, use the `NIM_ENDPOINT_URL` env var
    if not inference_base_url:
        inference_base_url = os.environ.get("NIM_ENDPOINT_URL", FALLBACK_DEFAULT_ENDPOINT_URL)

    # If the request's main model contains a base URL, use it
    main_model = get_main_model_from_context()
    main_model_base_url = main_model.parameters.get("base_url") if main_model and main_model.parameters else None

    return main_model_base_url or inference_base_url
