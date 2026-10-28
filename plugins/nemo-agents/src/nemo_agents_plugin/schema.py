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
from nemo_helix_plugin.api.filter import ComparisonOperation, FilterOperation, LogicalOperation
from nemo_helix_plugin.api.parsed_filter import ENTITY_BASE_FIELDS
from nemo_helix_plugin.schema import NemoFilter, NemoListResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Filters — extend NemoFilter so extra fields are rejected (extra="forbid")
# ---------------------------------------------------------------------------


_LABELS_FIELD = "labels"
_LABELS_PREFIX = f"{_LABELS_FIELD}."


# Implements the hooks ``make_filter_dep`` looks for, so declared fields are
# rewritten to their ``data.*`` entity-store paths and ``labels.<key>`` is
# accepted as a namespace sub-path.
class AgentFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/agents``."""

    config_format: str | None = Field(
        default=None,
        description="Filter to agents with this config format.",
    )
    labels: dict[str, str] | None = Field(
        default=None,
        description="Filter on a label, e.g. `filter[labels.helix-agents/managed-by]=compass`.",
    )

    @classmethod
    def _get_entity_field_map(cls) -> dict[str, str]:
        return {name: f"data.{name}" for name in cls.model_fields if name not in ENTITY_BASE_FIELDS}

    @classmethod
    def _get_entity_namespace_map(cls) -> dict[str, str]:
        return {_LABELS_FIELD: f"data.{_LABELS_FIELD}"}

    @classmethod
    def translate_operation(cls, operation: FilterOperation) -> FilterOperation:
        field_map = cls._get_entity_field_map()

        def _walk(op: FilterOperation) -> FilterOperation:
            if isinstance(op, ComparisonOperation):
                if op.field.startswith(_LABELS_PREFIX):
                    return op.model_copy(update={"field": f"data.{op.field}"})
                mapped = field_map.get(op.field)
                return op if mapped is None else op.model_copy(update={"field": mapped})
            if isinstance(op, LogicalOperation):
                return op.model_copy(update={"operations": [_walk(child) for child in op.operations]})
            return op

        return _walk(operation)


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
# List response type aliases
# ---------------------------------------------------------------------------

AgentPage = NemoListResponse[Agent]
DeploymentPage = NemoListResponse[AgentDeployment]
SessionPage = NemoListResponse[AgentSession]
EnvironmentPage = NemoListResponse[AgentEnvironment]
EnvironmentSpecPage = NemoListResponse[AgentEnvironmentSpec]
ComputeSpecPage = NemoListResponse[AgentComputeSpec]
