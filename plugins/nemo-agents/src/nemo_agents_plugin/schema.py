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

from enum import StrEnum

from nemo_agents_plugin.entities import (
    Agent,
    AgentComputeSpec,
    AgentDeployment,
    AgentEnvironment,
    AgentEnvironmentSpec,
    AgentSession,
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
    DeploymentModeAvailability as DeploymentModeAvailability,
)
from nemo_helix_plugin.agents.types import (
    DeploymentModeList as DeploymentModeList,
)
from nemo_helix_plugin.agents.types import (
    RegisterAgentRequest as RegisterAgentRequest,
)
from nemo_helix_plugin.api.filter import ComparisonOperation, FilterOperation, FilterOperator, LogicalOperation
from nemo_helix_plugin.api.parsed_filter import ENTITY_BASE_FIELDS
from nemo_helix_plugin.filter_ops import RelationshipOperation
from nemo_helix_plugin.schema import NemoFilter, NemoListResponse
from pydantic import Field

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


class SkillSort(StrEnum):
    """Sort fields for ``GET /v2/workspaces/{workspace}/skills``."""

    NAME_ASC = "name"
    NAME_DESC = "-name"
    CREATED_AT_ASC = "created_at"
    CREATED_AT_DESC = "-created_at"
    UPDATED_AT_ASC = "updated_at"
    UPDATED_AT_DESC = "-updated_at"


#: The relationship (registered in the entities service) from a skill head to its revisions, and the
#: revision field holding the parsed ``SKILL.md`` frontmatter (no ``data.`` prefix: the entities
#: service adds it to relationship conditions).
_SKILL_REVISIONS = "revisions"
_REVISION_FRONTMATTER = "frontmatter"


class SkillFilter(NemoFilter):
    """Query filter for ``GET /v2/workspaces/{workspace}/skills``.

    ``description`` and ``tags`` match the ``SKILL.md`` frontmatter of the skill's published
    revisions through the entities service's relationship filter: a skill matches when *any* of its
    revisions matches, not only the latest. ``name`` is the skill name, which always equals the
    frontmatter name.
    """

    description: str | None = Field(
        default=None,
        description="Filter by frontmatter description of any published revision, e.g. "
        "`filter[description][$like]=pdf`.",
    )
    tags: str | None = Field(
        default=None,
        description="Filter by frontmatter `metadata.tags` (not revision tags): matches skills with any published "
        "revision carrying the tag. Supports `$eq` and `$in`.",
    )

    @classmethod
    def _get_entity_field_map(cls) -> dict[str, str]:
        return {"description": f"{_REVISION_FRONTMATTER}.description", "tags": f"{_REVISION_FRONTMATTER}.tags"}

    @classmethod
    def translate_operation(cls, operation: FilterOperation) -> FilterOperation:
        def _walk(op: FilterOperation) -> FilterOperation:
            if isinstance(op, ComparisonOperation):
                if op.field == "description":
                    return RelationshipOperation(
                        relationship=_SKILL_REVISIONS,
                        condition=op.model_copy(update={"field": f"{_REVISION_FRONTMATTER}.description"}),
                    )
                if op.field == "tags":
                    return _has_frontmatter_tag(op)
                return op
            if isinstance(op, LogicalOperation):
                return op.model_copy(update={"operations": [_walk(child) for child in op.operations]})
            return op

        return _walk(operation)


def public_skill_filter(operation: FilterOperation | None) -> dict[str, object] | None:
    """The list response's ``filter`` echo, in the public field names a caller can submit again.

    Reverses :meth:`SkillFilter.translate_operation`: a ``revisions`` relationship condition goes
    back to ``description`` or ``tags``, and an OR of tag conditions back to ``tags: {$in: [...]}``.
    """

    def tag_value(op: FilterOperation) -> object | None:
        if (
            isinstance(op, RelationshipOperation)
            and op.condition is not None
            and op.condition.field == f"{_REVISION_FRONTMATTER}.tags"
        ):
            return op.condition.value
        return None

    def walk(op: FilterOperation) -> dict[str, object]:
        if isinstance(op, RelationshipOperation) and op.condition is not None:
            field = op.condition.field.removeprefix(f"{_REVISION_FRONTMATTER}.")
            if field == "tags":
                return {"tags": {FilterOperator.EQ.value: op.condition.value}}
            return {field: {op.condition.operator.value: op.condition.value}}
        if isinstance(op, LogicalOperation):
            values = [tag_value(child) for child in op.operations]
            if op.operator == FilterOperator.OR and op.operations and all(v is not None for v in values):
                return {"tags": {FilterOperator.IN.value: values}}
            if op.operator == FilterOperator.NOT:
                return {op.operator.value: walk(op.operations[0])}
            return {op.operator.value: [walk(child) for child in op.operations]}
        return op.to_dict()

    return None if operation is None else walk(operation)


def _has_frontmatter_tag(op: ComparisonOperation) -> FilterOperation:
    def contains(value: object) -> RelationshipOperation:
        if not isinstance(value, str) or not value:
            raise ValueError("'tags' filters take a non-empty string")
        return RelationshipOperation(
            relationship=_SKILL_REVISIONS,
            condition=ComparisonOperation(
                operator=FilterOperator.CONTAINS, field=f"{_REVISION_FRONTMATTER}.tags", value=value
            ),
        )

    if op.operator == FilterOperator.EQ:
        return contains(op.value)
    if op.operator == FilterOperator.IN and isinstance(op.value, list) and op.value:
        return LogicalOperation(operator=FilterOperator.OR, operations=[contains(value) for value in op.value])
    raise ValueError("'tags' supports only $eq and a non-empty $in")


# ---------------------------------------------------------------------------
# List response type aliases
# ---------------------------------------------------------------------------

AgentPage = NemoListResponse[Agent]
DeploymentPage = NemoListResponse[AgentDeployment]
SessionPage = NemoListResponse[AgentSession]
EnvironmentPage = NemoListResponse[AgentEnvironment]
EnvironmentSpecPage = NemoListResponse[AgentEnvironmentSpec]
ComputeSpecPage = NemoListResponse[AgentComputeSpec]
