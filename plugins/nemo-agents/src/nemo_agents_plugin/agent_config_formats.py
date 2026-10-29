# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation and deployment resolution for agent config formats."""

from __future__ import annotations

from typing import Any, Protocol

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.entities import NAT_WORKFLOW_CONFIG_FORMAT, NEMO_AGENTS_SPEC_CONFIG_FORMAT
from nemo_agents_plugin.utils import (
    inject_default_model,
    inject_fabric_gateway_url,
    inject_fabric_trace_agent_name,
    inject_gateway_url,
    inject_nemo_trace_fields,
)
from pydantic import ValidationError


class AgentConfigFormatError(ValueError):
    """Base error for unsupported or invalid agent config formats."""


class UnsupportedAgentConfigFormatError(AgentConfigFormatError):
    """Raised when no handler exists for an agent config format."""


class InvalidAgentConfigError(AgentConfigFormatError):
    """Raised when an agent config does not satisfy its format contract."""


class AgentConfigFormatHandler(Protocol):
    """Validate and resolve one persisted agent config format."""

    def validate(self, config: dict[str, Any]) -> dict[str, Any]: ...

    def resolve_for_deployment(
        self,
        config: dict[str, Any],
        *,
        workspace: str,
        agent_name: str,
    ) -> dict[str, Any]: ...


class _NatWorkflowConfigHandler:
    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        return config

    def resolve_for_deployment(
        self,
        config: dict[str, Any],
        *,
        workspace: str,
        agent_name: str,
    ) -> dict[str, Any]:
        resolved = inject_gateway_url(config, workspace)
        resolved = inject_default_model(resolved)
        inject_nemo_trace_fields(resolved, workspace=workspace, agent_name=agent_name)
        return resolved


class _NemoAgentsSpecConfigHandler:
    def validate(self, config: dict[str, Any]) -> dict[str, Any]:
        return self._normalize(config)

    def resolve_for_deployment(
        self,
        config: dict[str, Any],
        *,
        workspace: str,
        agent_name: str,
    ) -> dict[str, Any]:
        resolved = self._normalize(inject_fabric_gateway_url(config, workspace))
        return inject_fabric_trace_agent_name(resolved, agent_name)

    @staticmethod
    def _normalize(config: dict[str, Any]) -> dict[str, Any]:
        try:
            agent_config = AgentConfig.model_validate(config)
        except ValidationError as error:
            raise InvalidAgentConfigError(f"Invalid agent config: {error}") from error
        # Accepted in agent.yaml, where the CLI lifts it into request fields. In
        # a stored or inline config nothing reads it, so a caller who put it
        # there would silently get the defaults instead.
        if agent_config.registration is not None:
            raise InvalidAgentConfigError(
                "Invalid agent config: 'registration' belongs in agent.yaml only. Over the API, set "
                "reference, context_schema, default_input, and post_run_actions as fields of the agent."
            )
        return agent_config.model_dump(exclude_none=True)


_AGENT_CONFIG_FORMAT_HANDLERS: dict[str, AgentConfigFormatHandler] = {
    NAT_WORKFLOW_CONFIG_FORMAT: _NatWorkflowConfigHandler(),
    NEMO_AGENTS_SPEC_CONFIG_FORMAT: _NemoAgentsSpecConfigHandler(),
}


def get_agent_config_format_handler(config_format: str) -> AgentConfigFormatHandler:
    """Return the handler registered for an agent config format."""
    try:
        return _AGENT_CONFIG_FORMAT_HANDLERS[config_format]
    except KeyError as error:
        raise UnsupportedAgentConfigFormatError(f"Unsupported config_format {config_format!r}.") from error


def validate_agent_config(config_format: str, config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an agent config before persistence."""
    handler = get_agent_config_format_handler(config_format)
    return handler.validate(config) if config else config


def resolve_agent_config_for_deployment(
    config_format: str,
    config: dict[str, Any],
    *,
    workspace: str,
    agent_name: str,
) -> dict[str, Any]:
    """Resolve a persisted agent config for deployment."""
    handler = get_agent_config_format_handler(config_format)
    if not config:
        raise InvalidAgentConfigError(
            f"Agent '{workspace}/{agent_name}' has no configuration. "
            "This operation requires an agent registered with a non-empty config."
        )
    return handler.resolve_for_deployment(
        config,
        workspace=workspace,
        agent_name=agent_name,
    )
