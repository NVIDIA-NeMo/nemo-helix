# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent plugin API schema definitions — request bodies and filters.

This module contains only API-layer Pydantic models.  Entity definitions
(classes stored in the entity store) live in :mod:`nemo_agents_plugin.entities`.

Entity objects (subclasses of :class:`~nemo_helix_plugin.entity.NemoEntity`) are
returned directly from route handlers as the API response — no separate
response model is needed.  Use ``NemoListResponse[Agent]`` /
``NemoListResponse[AgentDeployment]`` for list endpoints.

Naming conventions:
- ``CreateXRequest`` / ``UpdateXRequest`` — plain :class:`~pydantic.BaseModel`
  for request bodies.
- ``XFilter`` — extends :class:`~nemo_helix_plugin.schema.NemoFilter` to inherit
  ``extra="forbid"``.
"""

from __future__ import annotations

from typing import Any, Literal

from nemo_agents_plugin.entities import (
    Agent,
    AgentComputeSpec,
    AgentDeployment,
    AgentEnvironment,
    AgentEnvironmentSpec,
    AgentSession,
    DeploymentMode,
    DeploymentStatus,
)
from nemo_helix_plugin.agents.types import (
    CreateComputeSpecRequest as CreateComputeSpecRequest,
)
from nemo_helix_plugin.agents.types import (
    CreateDeploymentRequest as CreateDeploymentRequest,
)
from nemo_helix_plugin.agents.types import (
    CreateEnvironmentRequest as CreateEnvironmentRequest,
)
from nemo_helix_plugin.agents.types import (
    CreateEnvironmentSpecRequest as CreateEnvironmentSpecRequest,
)
from nemo_helix_plugin.agents.types import (
    CreateSessionRequest as CreateSessionRequest,
)
from nemo_helix_plugin.agents.types import (
    RegisterAgentRequest as RegisterAgentRequest,
)
from nemo_helix_plugin.schema import NemoFilter, NemoListResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Filters — extend NemoFilter so extra fields are rejected (extra="forbid")
# ---------------------------------------------------------------------------


class AgentFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/agents``."""

    config_format: str | None = Field(
        default=None,
        description="Filter to agents with this config format.",
    )


class DeploymentFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/deployments``."""

    agent: str | None = Field(
        default=None,
        description="Filter to deployments for this agent name.",
    )
    status: DeploymentStatus | None = Field(
        default=None,
        description="Filter to deployments in this lifecycle status.",
    )


class SessionFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/sessions``."""

    deployment_id: str | None = Field(
        default=None,
        description="Filter to sessions for this deployment ID.",
    )


class EnvironmentFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/environments``."""


class EnvironmentSpecFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/environment-specs``."""


class ComputeSpecFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/compute-specs``."""


class DeploymentModeAvailability(BaseModel):
    """Whether this platform can run deployments in one ``deployment_mode``."""

    mode: DeploymentMode = Field(description="The deployment mode.")
    enabled: bool = Field(
        description="True when a deployment in this mode would run on a matching executor backend.",
    )
    requires_image: bool = Field(
        description="True when a deployment in this mode must set 'image' because no default image is configured.",
    )


class DeploymentModeList(BaseModel):
    """Response for ``GET /v2/workspaces/{workspace}/deployment-modes``."""

    data: list[DeploymentModeAvailability]


# ---------------------------------------------------------------------------
# Agent tools — what an agent's MCP servers expose
# ---------------------------------------------------------------------------

McpServerToolsStatus = Literal["discovered", "error", "not_deployed"]


class AgentTool(BaseModel):
    """One tool an MCP server reports through ``tools/list``."""

    name: str
    title: str | None = Field(default=None, description="Human-readable display name, when the server sets one.")
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for the tool's arguments.")
    enabled: bool = Field(
        description="False when the agent's allowed/blocked tool policy hides this tool from the model.",
    )


class AgentMcpServerTools(BaseModel):
    """The tools one declared MCP server exposes, or why they could not be listed."""

    name: str = Field(description="Server name, as declared under the agent config's ``mcp.servers``.")
    transport: str
    status: McpServerToolsStatus = Field(
        description=(
            "``discovered`` when ``tools`` came from the running server; ``error`` when listing failed "
            "(see ``error``); ``not_deployed`` when the agent has no running deployment to ask."
        ),
    )
    error: str | None = None
    tools: list[AgentTool] = Field(default_factory=list)


class AgentToolsResponse(BaseModel):
    """Response for ``GET /v2/workspaces/{workspace}/agents/{name}/tools``."""

    deployment: str | None = Field(
        default=None,
        description="The running deployment the tools were listed from; null when none is running.",
    )
    servers: list[AgentMcpServerTools]


# ---------------------------------------------------------------------------
# List response type aliases
# ---------------------------------------------------------------------------

AgentPage = NemoListResponse[Agent]
DeploymentPage = NemoListResponse[AgentDeployment]
SessionPage = NemoListResponse[AgentSession]
EnvironmentPage = NemoListResponse[AgentEnvironment]
EnvironmentSpecPage = NemoListResponse[AgentEnvironmentSpec]
ComputeSpecPage = NemoListResponse[AgentComputeSpec]
