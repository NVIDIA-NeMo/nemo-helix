# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent entity definitions — stored in the NeMo Platform entity store.

This module contains only entity classes (subclasses of
:class:`~nemo_platform_plugin.entity.NemoEntity`).  API request/response schemas and
filter models live in :mod:`nemo_agents_plugin.schema`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from nemo_platform_plugin.agents.types import (
    NAT_WORKFLOW_CONFIG_FORMAT as NAT_WORKFLOW_CONFIG_FORMAT,
)
from nemo_platform_plugin.agents.types import (
    NEMO_AGENTS_SPEC_CONFIG_FORMAT as NEMO_AGENTS_SPEC_CONFIG_FORMAT,
)
from nemo_platform_plugin.agents.types import (
    AgentEnvironmentInline as AgentEnvironmentInline,
)
from nemo_platform_plugin.agents.types import (
    ComputeResources as ComputeResources,
)
from nemo_platform_plugin.agents.types import (
    ComputeSpecInline as ComputeSpecInline,
)
from nemo_platform_plugin.agents.types import (
    DeploymentMode as DeploymentMode,
)
from nemo_platform_plugin.agents.types import (
    DeploymentStatus as DeploymentStatus,
)
from nemo_platform_plugin.agents.types import (
    Endpoint as Endpoint,
)
from nemo_platform_plugin.agents.types import (
    EnvironmentSpecInline as EnvironmentSpecInline,
)
from nemo_platform_plugin.agents.types import (
    McpFulfillment as McpFulfillment,
)
from nemo_platform_plugin.agents.types import (
    ModelProviderOverride as ModelProviderOverride,
)
from nemo_platform_plugin.auth import AuthContext
from nemo_platform_plugin.entity import NemoEntity
from nemo_platform_plugin.refs import FilesetRef
from pydantic import BaseModel, Field, PrivateAttr, computed_field


class SessionStatus(StrEnum):
    """Lifecycle status of an agent session."""

    ACTIVE = "active"
    EXPIRED = "expired"
    LOST = "lost"
    CLOSED = "closed"

    def can_transition_to(self, new_status: SessionStatus) -> bool:
        """Return whether transitioning to *new_status* is allowed."""
        if self == new_status:
            return True

        valid_transitions = {
            SessionStatus.ACTIVE: {
                SessionStatus.EXPIRED,
                SessionStatus.LOST,
                SessionStatus.CLOSED,
            },
            SessionStatus.EXPIRED: {SessionStatus.CLOSED},
            SessionStatus.LOST: {SessionStatus.CLOSED},
            SessionStatus.CLOSED: set(),
        }
        return new_status in valid_transitions[self]


# Runtime backend for an AgentDeployment. ``subprocess`` (the default) runs the
# agent as a local ``nat serve`` process reachable on a loopback ``endpoint``.
# ``docker``/``k8s`` run the agent as a durable container deployment via the
# deployments plugin; their routable address is projected onto ``endpoints``.
# Modes that compile to the nemo-deployments plugin (not local subprocess).
CONTAINER_DEPLOYMENT_MODES: frozenset[str] = frozenset({"docker", "k8s"})


def is_container_deployment_mode(mode: str) -> bool:
    """Return True when *mode* uses the deployments-plugin runner backend."""
    return mode in CONTAINER_DEPLOYMENT_MODES


# ---------------------------------------------------------------------------
# AgentEnvironment composition
# ---------------------------------------------------------------------------
#
# An AgentDeployment (and, later, an AgentInvocationJob) runs against an
# AgentEnvironment: a composition of an EnvironmentSpec (the dependencies an
# agent reaches - model endpoints, secrets, env vars, MCP servers) and a
# ComputeSpec (k8s-style resource requests/limits). Each part varies
# independently and is referenced by ``ref | inline | None`` so a spec can be
# authored once and reused across many Environments.
#
# The specs are also first-class entities (``agent_environment_spec``,
# ``agent_compute_spec``, ``agent_environment``) with their own CRUD APIs. Most
# inline BaseModels come from ``nemo_platform_plugin.agents.types`` so the
# entity and typed-client contracts share one shape. ``AgentInline`` stays local
# because persisted agent configs are mutable framework payloads where
# ``dict[str, Any]`` preserves useful static ergonomics for nested config access.
#
# Environment values compile into two targets at deploy time: the on-disk
# agent.yaml / FabricConfig (env vars, MCP, model provider) and, for container
# modes, the deployments-plugin Container.resources (compute). See
# :mod:`nemo_agents_plugin.environment_resolution` for the merge + snapshot.


# ---------------------------------------------------------------------------
# Canonical Ethos storage convention
# ---------------------------------------------------------------------------
#
# Each agent has exactly one Ethos fileset, named by convention. The fileset can
# hold both the human-readable Ethos document and the machine-readable agent
# config. We do **not** store these locations on the agent - they are fully
# derivable from the agent's workspace and name. Consumers should call the
# file-ref helpers below rather than reconstructing refs inline.
#
# Layout:
#   - Fileset (entity ref):  ``{workspace}/{agent-name}-ethos``
#   - Human-readable Ethos:  ``ETHOS.md``
#   - Machine-readable cfg:  ``agent.yaml``
#   - Ethos file ref:        ``{workspace}/{agent-name}-ethos#ETHOS.md``
#   - Config file ref:       ``{workspace}/{agent-name}-ethos#agent.yaml``
#   - Local cache root:      ``agents/{agent-name}-ethos/``
#
# This is intentionally **not** an Optional field on the Agent. The
# relationship is 1:1 and convention-bound; carrying a stored ref would
# duplicate state with no resilience benefit (rename of either entity
# orphans both representations equally).

ETHOS_FILENAME = "ETHOS.md"
"""Canonical filename inside the agent's Ethos fileset."""

AGENT_SPEC_FILENAME = "AGENT-SPEC.md"
"""Prior contract filename. Staging drops it from the runtime tree."""

AGENT_CONFIG_FILENAME = "agent.yaml"
"""Canonical machine-readable agent config filename in the agent Ethos fileset.

This file is parsed into Agent.config when using the nemo-agents-spec-v1 format.
"""

ETHOS_LOCAL_ROOT = "agents"
"""Local directory holding agent build artifacts."""

# Container deployments deliver the Ethos fileset through a ConfigMap (k8s) or a
# single env var (docker), both of which cap out around 1MiB. Bound the tree at
# both ends of the pipe so an agent root pointed at a whole checkout fails at
# upload time with a clear message instead of at container start.
MAX_ETHOS_STAGED_BYTES = 900_000
"""Maximum total bytes an agent Ethos fileset may contribute to a deployment."""

MAX_ETHOS_STAGED_FILES = 500
"""Maximum number of files an agent Ethos fileset may contain."""


def ethos_fileset_name(agent_name: str) -> str:
    """Return the conventional fileset name holding an agent's Ethos."""
    return f"{agent_name}-ethos"


def ethos_local_path(agent_name: str, root: str | Path = ETHOS_LOCAL_ROOT) -> Path:
    """Return the local write-through cache path for an agent's Ethos."""
    return Path(root) / ethos_fileset_name(agent_name) / ETHOS_FILENAME


def ethos_file_ref(workspace: str, agent_name: str) -> FilesetRef:
    """Return the canonical file ref ``workspace/<name>-ethos#ETHOS.md``.

    Use this anywhere downstream code needs to point at an agent's Ethos -
    do not reconstruct the path inline. If the layout ever changes (e.g.
    moving to a per-agent bundle fileset holding multiple artifacts), this
    is the only function that needs to update.
    """
    return FilesetRef(f"{workspace}/{ethos_fileset_name(agent_name)}#{ETHOS_FILENAME}")


def agent_config_file_ref(workspace: str, agent_name: str) -> FilesetRef:
    """Return the canonical file ref ``workspace/<name>-ethos#agent.yaml``.

    Use this anywhere downstream code needs to point at an agent's config -
    do not reconstruct the path inline. If the layout ever changes (e.g.
    moving to a per-agent bundle fileset holding multiple artifacts), this
    is the only function that needs to update.
    """
    return FilesetRef(f"{workspace}/{ethos_fileset_name(agent_name)}#{AGENT_CONFIG_FILENAME}")


class AgentComputeSpec(NemoEntity, ComputeSpecInline, entity_type="agent_compute_spec"):
    """A reusable compute spec (k8s-style resource requests/limits).

    Entity type: ``agent_compute_spec``
    Referenced by an AgentEnvironment's ``compute_spec`` (by name or inline).
    """


class AgentEnvironmentSpec(NemoEntity, EnvironmentSpecInline, entity_type="agent_environment_spec"):
    """A reusable environment spec (the dependencies an agent reaches).

    Entity type: ``agent_environment_spec``
    Referenced by an AgentEnvironment's ``environment_spec`` (by name or inline).
    """


class AgentEnvironment(NemoEntity, AgentEnvironmentInline, entity_type="agent_environment"):
    """A composition of an environment spec and a compute spec.

    Entity type: ``agent_environment``
    The single thing an AgentDeployment references. Each part is a
    ``ref | inline | None`` union so specs can be authored once and reused.
    """


class AgentInline(BaseModel):
    """Inline Agent - an agent definition without entity identity.

    The shared shape behind the :class:`Agent` entity: the entity embeds these
    fields and adds name/workspace, and a field that accepts an agent can take
    either a ``"workspace/name"`` ref string or this model. Lets a caller
    execute an agent it composes at request time — models chosen per request,
    settings scoped to one run — without first persisting an Agent.

    Deliberately as permissive as the entity: consumers impose their own
    requirements rather than this model narrowing them for everyone. The
    ``agents.execute`` job, for example, accepts only ``nemo-agents-spec-v1``
    and rejects a config that is not a valid agent spec.
    """

    description: str = Field(default="", description="Human-readable description of the agent.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent config dict interpreted according to config_format.",
    )
    config_format: str = Field(
        default=NAT_WORKFLOW_CONFIG_FORMAT,
        description=(
            "platform-internal schema version tag for the agent config dict. "
            "`nat-workflow-v1` is the default legacy NAT workflow format; "
            "`nemo-agents-spec-v1` identifies the Platform-owned agent.yaml spec format."
        ),
    )


# TODO: first-class environment, sandbox, and harness specs are planned for the
# Agent entity. Add those specs to this object once the contract is finalized.
class Agent(NemoEntity, AgentInline, entity_type="agent"):
    """An agent definition — stores agent config and metadata.

    Entity type: ``agent``
    Primary lookup: by ``name`` within a ``workspace``.

    The agent's Ethos files live at the locations returned by
    :func:`ethos_file_ref` and :func:`agent_config_file_ref` — they
    are **not** stored on the entity because the paths are fully derivable
    from ``(workspace, name)``.
    """


class AgentDeployment(NemoEntity, entity_type="agent_deployment"):
    """A running (or pending) deployment of an Agent.

    Entity type: ``agent_deployment``
    Lifecycle: pending → starting → running | failed.
    The :class:`~nemo_agents_plugin.runner.controller.AgentDeploymentController`
    drives state transitions by reconciling this entity against the
    :class:`~nemo_agents_plugin.runner.backend.RunnerBackend`.
    """

    _auth_context: AuthContext | None = PrivateAttr(default=None)

    agent: str = Field(default="", description="Name of the Agent entity this deployment is for.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Resolved agent config with IGW URL injected and any referenced environment spec merged in, "
            "written when the deployment is created."
        ),
    )
    # AgentEnvironment is snapshotted at create time: ``environment`` records the
    # raw request input for provenance, environment-spec content is merged into
    # ``config``, ``compute`` holds the resolved compute snapshot, and ``secrets``
    # holds the resolved secret-env references — all threaded to the container
    # backend. A deployment is not kept in sync with the underlying
    # AgentEnvironment/spec entities after creation.
    environment: str | AgentEnvironmentInline | None = Field(
        default=None,
        description=(
            '"workspace/name" ref to an AgentEnvironment, an inline environment, or None. '
            "Snapshotted at create time for provenance; the resolved values live in config/compute/secrets."
        ),
    )
    compute: ComputeSpecInline | None = Field(
        default=None,
        description=(
            "Resolved compute snapshot from the referenced environment. Compiled into the container "
            "resources for docker/k8s modes; ignored for subprocess."
        ),
    )
    secrets: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Resolved secret env references from the referenced environment, as "
            "ENV_VAR_NAME -> 'workspace/secret-name'. Compiled into secret-backed container env "
            "vars (never plaintext) for docker/k8s modes; ignored for subprocess."
        ),
    )
    status: DeploymentStatus = Field(
        default="pending",
        description="Lifecycle status: pending | starting | running | failed | deleting.",
    )
    deployment_mode: DeploymentMode = Field(
        default="subprocess",
        description=(
            "Runtime backend for this deployment. 'subprocess' (default) reads the loopback "
            "'endpoint'; 'docker'/'k8s' read the projected 'endpoints'."
        ),
    )
    # Dual addressing: subprocess uses loopback ``endpoint``; docker/k8s project
    # routable addresses onto ``endpoints`` and leave ``endpoint`` empty.
    endpoint: str = Field(
        default="", description="Subprocess loopback endpoint of the agent process (e.g. http://localhost:9001)."
    )
    endpoints: list[Endpoint] = Field(
        default_factory=list,
        description=(
            "Routable endpoints for container modes, projected from the deployments-plugin "
            "Deployment. Empty for subprocess mode (which uses 'endpoint')."
        ),
    )
    image: str = Field(
        default="",
        description="Container image for docker/k8s modes. Empty for subprocess; falls back to AgentsConfig.deployments.default_image.",
    )
    use_image_entrypoint: bool = Field(
        default=False,
        description=(
            "Container modes only: preserve the image ENTRYPOINT/CMD instead of injecting "
            "the platform-owned NAT/Fabric server command."
        ),
    )
    plugin_deployment: str = Field(
        default="",
        description=(
            "Name of the linked nemo-deployments Deployment entity. Defaults to this "
            "deployment's name when empty (set by the controller on create)."
        ),
    )
    port: int = Field(default=0, description="Port the agent process is listening on.")
    pid: int = Field(default=0, description="OS process ID of the agent subprocess.")
    error: str = Field(default="", description="Error message if status is 'failed'.")

    @computed_field(json_schema_extra={"nullable": True})
    @property
    def auth_context(self) -> AuthContext | None:
        return self._auth_context

    def with_auth_context(self, auth_context: AuthContext | None) -> Self:
        self._auth_context = auth_context
        return self


class AgentSession(NemoEntity, entity_type="agent_session"):
    """A durable logical conversation associated with an AgentDeployment.

    The session entity owns Platform-level conversation identity only. Live
    harness state remains process-local to the deployment's serving runtime.
    """

    deployment_id: str = Field(
        min_length=1,
        description="Immutable ID of the AgentDeployment this session belongs to.",
    )
    status: SessionStatus = Field(
        default=SessionStatus.ACTIVE,
        description="Lifecycle status: active | expired | lost | closed.",
    )
    first_active_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of the first accepted invocation; immutable once set.",
        json_schema_extra={"nullable": True},
    )
    last_active_at: datetime | None = Field(
        default=None,
        description=(
            "UTC timestamp of the latest accepted or completed invocation; null until the first invocation is accepted."
        ),
        json_schema_extra={"nullable": True},
    )
    expires_at: datetime | None = Field(
        default=None,
        description="UTC rolling idle deadline derived from the latest session activity.",
        json_schema_extra={"nullable": True},
    )
