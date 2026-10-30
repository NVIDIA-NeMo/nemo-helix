# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records and API shapes for agent layers: a task revision's main image with an agent already installed."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import ClassVar

from nemo_evals.content_hash import DIGEST_LENGTH, DIGEST_PATTERN
from nemo_evals.image_builds.models import BuildImage, BuildStatus
from nemo_helix_plugin.entities import EntityBase
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

AGENT_LAYER_BUILD_JOB_SOURCE = "nemo-evals.agent-layer-build"
_LAYER_NAME_PREFIX = "agent-layer."


class AgentLayerRequest(BaseModel):
    """The agent to install into a task revision's main image: a Harbor installed agent, or the registered-agent one.

    Give what a run's Harbor target gives the agent: its built-in ``name``, or its ``import_path``, and its ``kwargs``.
    Only an agent whose setup is an install can be layered: Harbor's installed agents (``codex``, ``claude-code``,
    ...) and registered agents (the SDK's ``FabricInstalledAgent``).
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, description="A Harbor built-in agent, such as `codex`.")
    import_path: str | None = Field(
        default=None, description="The agent class, as `module:Class`; only the registered-agent Fabric agent layers."
    )
    kwargs: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "The agent's Harbor kwargs, as the run gives them. Only what changes the install (its `version`, or a "
            "Fabric agent's package and Python/uv versions) picks the layer; the rest is used to build the agent."
        ),
    )

    @model_validator(mode="after")
    def _one_agent(self) -> AgentLayerRequest:
        if (self.name is None) == (self.import_path is None):
            raise ValueError("give exactly one of name and import_path")
        return self


class AgentLayerEntity(EntityBase):
    """One attempt at building one agent's layer for a task revision: a child of the task.

    Keyed by the revision digest and the agent install's key, so revisions with the same contents and runs with the
    same install share it. Builder shares the image itself across tasks whose environments hold the same files.
    """

    __entity_type__: ClassVar[str] = "agent_layer"

    task: str = Field(description="`<workspace>/<name>` of the task.")
    revision_digest: str = Field(min_length=DIGEST_LENGTH, max_length=DIGEST_LENGTH, pattern=DIGEST_PATTERN)
    agent_key: str = Field(description="The agent install's key (`AgentInstall.layer_key` over the installer).")
    agent: AgentLayerRequest
    attempt: int = Field(ge=1)
    status: BuildStatus = "pending"
    job: str = Field(description="Name of the agent-layer Job that runs this attempt.")
    image: BuildImage | None = Field(default=None, description="The layer's image, once the build has planned it.")
    error: str | None = None


class AgentLayer(BaseModel):
    """An agent layer of one task revision."""

    id: str = Field(description="The layer's attempt number for this revision and agent.")
    task: str
    revision: str = Field(description="Digest of the task revision.")
    agent_key: str
    status: BuildStatus
    job: str
    image: BuildImage | None = None
    error: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_entity(cls, entity: AgentLayerEntity) -> AgentLayer:
        return cls(
            id=str(entity.attempt),
            task=entity.task,
            revision=entity.revision_digest,
            agent_key=entity.agent_key,
            status=entity.status,
            job=entity.job,
            image=entity.image,
            error=entity.error,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )


def layer_entity_name(revision_digest: str, agent_key: str, attempt: int) -> str:
    """Name of one attempt's record under its task: valid as an entity name, unique per contents, agent and attempt."""
    return f"{_LAYER_NAME_PREFIX}{revision_digest[:24]}.{agent_key[:16]}.{attempt}"


def layer_job_name(task: str, revision_digest: str, agent_key: str, attempt: int) -> str:
    """Name of the agent-layer Job for one attempt."""
    key = hashlib.sha256(f"{task}#{revision_digest}|{agent_key}".encode()).hexdigest()[:16]
    return f"agent-layer-build-{key}-{attempt}"
