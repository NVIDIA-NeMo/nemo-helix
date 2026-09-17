# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent plugin API schema definitions — request bodies and filters.

This module contains only API-layer Pydantic models.  Entity definitions
(classes stored in the entity store) live in :mod:`nemo_agents_plugin.entities`.

Entity objects (subclasses of :class:`~nemo_platform_plugin.entity.NemoEntity`) are
returned directly from route handlers as the API response — no separate
response model is needed.  Use ``NemoListResponse[Agent]`` /
``NemoListResponse[AgentDeployment]`` for list endpoints.

Naming conventions:
- ``CreateXRequest`` / ``UpdateXRequest`` — plain :class:`~pydantic.BaseModel`
  for request bodies.
- ``XFilter`` — extends :class:`~nemo_platform_plugin.schema.NemoFilter` to inherit
  ``extra="forbid"``.
"""

from __future__ import annotations

from nemo_agents_plugin.entities import (
    Agent,
    AgentComputeSpec,
    AgentDeployment,
    AgentEnvironment,
    AgentEnvironmentSpec,
    AgentSession,
    DeploymentStatus,
)
from nemo_platform_plugin.agents.types import (
    CreateAgentRequest as CreateAgentRequest,
)
from nemo_platform_plugin.agents.types import (
    CreateComputeSpecRequest as CreateComputeSpecRequest,
)
from nemo_platform_plugin.agents.types import (
    CreateDeploymentRequest as CreateDeploymentRequest,
)
from nemo_platform_plugin.agents.types import (
    CreateEnvironmentRequest as CreateEnvironmentRequest,
)
from nemo_platform_plugin.agents.types import (
    CreateEnvironmentSpecRequest as CreateEnvironmentSpecRequest,
)
from nemo_platform_plugin.agents.types import (
    CreateSessionRequest as CreateSessionRequest,
)
from nemo_platform_plugin.schema import NemoFilter, NemoListResponse
from pydantic import Field

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


# ---------------------------------------------------------------------------
# List response type aliases
# ---------------------------------------------------------------------------

AgentPage = NemoListResponse[Agent]
DeploymentPage = NemoListResponse[AgentDeployment]
SessionPage = NemoListResponse[AgentSession]
EnvironmentPage = NemoListResponse[AgentEnvironment]
EnvironmentSpecPage = NemoListResponse[AgentEnvironmentSpec]
ComputeSpecPage = NemoListResponse[AgentComputeSpec]
