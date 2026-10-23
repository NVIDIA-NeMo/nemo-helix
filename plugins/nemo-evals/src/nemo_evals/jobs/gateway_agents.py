# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Harbor agents that can call an Inference Gateway model, and the variables each one reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class GatewayAgentEnv:
    """Where one agent family reads its endpoint and key."""

    base_url_env: str
    api_key_env: str
    route_suffix: str


_OPENAI = GatewayAgentEnv(base_url_env="OPENAI_BASE_URL", api_key_env="OPENAI_API_KEY", route_suffix="/v1")
_ANTHROPIC = GatewayAgentEnv(base_url_env="ANTHROPIC_BASE_URL", api_key_env="ANTHROPIC_API_KEY", route_suffix="")

#: The key a gateway agent gets when its target supplies none. With platform auth off the gateway checks no key, but
#: Claude Code won't start without one.
GATEWAY_PLACEHOLDER_KEY = "inference-gateway-no-key"

GatewayAgentName: TypeAlias = Literal["codex", "claude-code"]

GATEWAY_AGENTS: Mapping[GatewayAgentName, GatewayAgentEnv] = MappingProxyType(
    {
        "codex": _OPENAI,
        "claude-code": _ANTHROPIC,
    }
)


def validate_gateway_env(env: GatewayAgentEnv, env_vars: Mapping[str, str], env_secrets: Mapping[str, object]) -> None:
    """Refuse a gateway agent whose base URL or key variable the target also sets itself."""
    if env.base_url_env in env_vars or env.base_url_env in env_secrets:
        raise ValueError(f"{env.base_url_env} is set from the gateway model; remove it from env_vars and env_secrets")
    if env.api_key_env in env_vars:
        raise ValueError(f"{env.api_key_env} for a gateway model belongs in env_secrets, as a platform access key")
