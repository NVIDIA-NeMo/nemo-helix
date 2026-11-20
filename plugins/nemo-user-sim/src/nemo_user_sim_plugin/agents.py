# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turns a registered Helix agent into the Fabric config a simulation runs it with."""

from typing import Any

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.entities import Agent
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_agents_plugin.utils import inject_fabric_gateway_url
from nemo_fabric import FabricConfig

AGENT_SPEC_FORMAT = "nemo-agents-spec-v1"


class AgentResolutionError(ValueError):
    pass


async def resolve_agent(entity_client: Any, name: str, workspace: str) -> FabricConfig:
    """Load the agent, bind its models to this workspace's inference gateway and translate it for Fabric."""
    try:
        agent = await entity_client.get(Agent, name=name, workspace=workspace)
    except Exception as exc:
        raise AgentResolutionError(f"Agent '{name}' wasn't found in workspace '{workspace}': {exc}") from exc
    if agent.config_format != AGENT_SPEC_FORMAT:
        raise AgentResolutionError(
            f"Agent '{name}' uses config format '{agent.config_format}'. "
            f"UserSim runs agents defined with '{AGENT_SPEC_FORMAT}'."
        )
    try:
        config = AgentConfig.model_validate(inject_fabric_gateway_url(agent.config, workspace))
        return translate_agent_config(config)
    except Exception as exc:
        raise AgentResolutionError(f"Agent '{name}' couldn't be prepared for Fabric: {exc}") from exc


def default_model(config: FabricConfig) -> str:
    """The agent's default model name, used as its identity in UserSim trajectory IDs."""
    model = (config.models or {}).get("default")
    return model.model if model else "unknown"
