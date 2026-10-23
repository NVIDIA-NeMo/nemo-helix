# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a built-in Harbor agent reaches an Inference Gateway model, from the model connection it declares."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache

from harbor.agents.factory import AgentFactory
from harbor.agents.model_connection import PROVIDERS
from harbor.models.agent.name import AgentName

#: The key a gateway agent gets when its target supplies none. With platform auth off the gateway checks no key, but
#: Claude Code won't start without one.
GATEWAY_PLACEHOLDER_KEY = "inference-gateway-no-key"

#: The API styles the gateway serves on a model's route, and what each client appends to the base URL itself.
_ROUTE_SUFFIX = {"openai": "/v1", "anthropic": ""}


@dataclass(frozen=True)
class GatewayAgentEnv:
    """Where an agent reads its endpoint and key, and how it names the gateway model."""

    base_url_env: str
    api_key_env: str
    route_suffix: str
    model_prefix: str


@cache
def gateway_agent_env(name: str) -> GatewayAgentEnv | None:
    """How the built-in Harbor agent ``name`` calls a gateway model, or None when it can't.

    An agent qualifies when it declares a model connection whose provider the gateway speaks. One that takes its
    provider from the model name is sent ``openai/<name>``, the OpenAI-style route.
    """
    try:
        agent = AgentFactory.get_agent_class(AgentName(name))
    except ValueError:
        return None
    spec = agent.MODEL_CONNECTION
    if spec is None:
        return None
    provider = spec.default_provider or "openai"
    if provider not in _ROUTE_SUFFIX:
        return None
    access = PROVIDERS[provider]
    return GatewayAgentEnv(
        base_url_env=(*spec.base_url_envs, *access.base_url_envs)[0],
        api_key_env=(*spec.api_key_envs, *access.api_key_envs)[0],
        route_suffix=_ROUTE_SUFFIX[provider],
        model_prefix="" if spec.default_provider else f"{provider}/",
    )


def validate_gateway_env(env: GatewayAgentEnv, env_vars: Mapping[str, str], env_secrets: Mapping[str, object]) -> None:
    """Refuse a gateway agent whose base URL or key variable the target also sets itself."""
    if env.base_url_env in env_vars or env.base_url_env in env_secrets:
        raise ValueError(f"{env.base_url_env} is set from the gateway model; remove it from env_vars and env_secrets")
    if env.api_key_env in env_vars:
        raise ValueError(f"{env.api_key_env} for a gateway model belongs in env_secrets, as a platform access key")
