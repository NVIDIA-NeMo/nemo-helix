# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Harbor agents that can call an Inference Gateway model, and the variables each one reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class GatewayAgentEnv:
    """Where one agent family reads its endpoint and key."""

    base_url_env: str
    api_key_env: str
    route_suffix: str


_OPENAI = GatewayAgentEnv(base_url_env="OPENAI_BASE_URL", api_key_env="OPENAI_API_KEY", route_suffix="/v1")
_ANTHROPIC = GatewayAgentEnv(base_url_env="ANTHROPIC_BASE_URL", api_key_env="ANTHROPIC_API_KEY", route_suffix="")

#: Keyed by built-in agent name or import path.
GATEWAY_AGENTS: Mapping[str, GatewayAgentEnv] = MappingProxyType(
    {
        "codex": _OPENAI,
        "claude-code": _ANTHROPIC,
        "nemo_evals.harbor.agents:SandboxedCodex": _OPENAI,
    }
)
