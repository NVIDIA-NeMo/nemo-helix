# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Requesting and reading agent layers of a task revision."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from nemo_evals.agent_layers.models import (
    AgentLayer,
    AgentLayerEntity,
    AgentLayerRequest,
    layer_entity_name,
    layer_job_name,
)
from nemo_evals.agent_layers.runner import AgentLayerBuildSpec
from nemo_evals.api.task_definitions.harbor import HarborTaskDefinition
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.image_builds.planning import parse_task_config
from nemo_evals.image_builds.service import (
    AuthorizeBuilds,
    BuildJobs,
    ImageBuildJobConflictError,
)
from nemo_evals.jobs.agent_layer_build import agent_layer_build_job_request, installer_image, is_job_for
from nemo_evals.revisions import get_revision
from nemo_evals.task_identity import UnsupportedTaskKindError
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entity_client import NemoEntityConflictError, NemoEntityNotFoundError
from nemo_helix_plugin.filter_ops import ComparisonOperation, FilterOperator, LogicalOperation
from nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer import agent_install

logger = logging.getLogger(__name__)

_ENDED_JOB_STATUSES = frozenset({"completed", "error", "cancelled"})


class AgentNotLayerableError(ValueError):
    """The agent's setup isn't an install that can be done ahead of time."""


def agent_layer_key(request: AgentLayerRequest) -> str:
    """The key of ``request``'s install, independent of any environment: the agent install and the installer.

    Raises:
        AgentNotLayerableError: The agent's setup can't be done ahead of time.
    """
    install = agent_install(
        agent_name=request.name, agent_import_path=request.import_path, agent_dir=None, agent_kwargs=request.kwargs
    )
    if install is None:
        raise AgentNotLayerableError(
            f"agent {request.name or request.import_path} installs nothing a layer could hold, or may do anything in "
            "setup; only Harbor's installed agents and registered agents are layered"
        )
    return install.layer_key(f"installer={installer_image()}")


def expired(layer: AgentLayerEntity, now: datetime) -> bool:
    """Whether runs may no longer start trials from ``layer``."""
    expires = layer.expires_at
    if expires is None:
        return False
    return now >= (expires if expires.tzinfo is not None else expires.replace(tzinfo=timezone.utc))


class AgentLayerService:
    """Agent layers of task revisions, stored under the task and keyed by revision digest and agent install."""

    def __init__(self, entity_client: EntityClient, jobs: BuildJobs, authorize_builds: AuthorizeBuilds) -> None:
        self.entity_client = entity_client
        self.jobs = jobs
        self.authorize_builds = authorize_builds

    async def request_layer(
        self,
        workspace: str,
        name: str,
        revision: str,
        request: AgentLayerRequest,
        *,
        ttl: timedelta,
        now: datetime,
    ) -> AgentLayer | None:
        """Start a layer of a revision for an agent, or return the one already running, or ready and unexpired.

        A new attempt expires ``ttl`` after ``now`` and reuses only a Builder build made within ``ttl`` before it.
        Returns ``None`` when the task doesn't exist.

        Raises:
            RevisionNotFoundError: ``revision`` names no revision of the task.
            UnsupportedTaskKindError: The task isn't a Harbor task.
            ImagePlanError: The task's configuration can't be built at all.
            AgentNotLayerableError: The agent's setup can't be done ahead of time.
            ImageBuildJobConflictError: The layer's Job name is held by another Job.
            ImageBuildForbiddenError: The caller may not submit Builder builds in the workspace.
        """
        found = await self._revision(workspace, name, revision)
        if found is None:
            return None
        await self.authorize_builds(workspace)
        head, task_revision = found
        if not isinstance(task_revision.spec, HarborTaskDefinition):
            raise UnsupportedTaskKindError(
                f"Task {workspace}/{name} is not a Harbor task; only Harbor tasks have images"
            )
        parse_task_config(task_revision.spec.config)
        key = agent_layer_key(request)

        layers = await self._layers(head, task_revision.content_hash, key)
        newest = layers[0] if layers else None
        if newest is not None and newest.status == "ready":
            if not expired(newest, now):
                return AgentLayer.from_entity(newest)
        elif newest is not None and newest.status != "failed":
            job = await self.jobs.get(newest.job)
            if job is None or job.status not in _ENDED_JOB_STATUSES:
                await self._ensure_job(newest)
                return AgentLayer.from_entity(newest)
            newest = await self._fail_abandoned(newest, job.status)
            if newest.status != "failed":
                return AgentLayer.from_entity(newest)

        attempt = 1 if newest is None else newest.attempt + 1
        task = f"{workspace}/{name}"
        layer = AgentLayerEntity(
            name=layer_entity_name(task_revision.content_hash, key, attempt),
            workspace=workspace,
            project=head.project,
            task=task,
            revision_digest=task_revision.content_hash,
            agent_key=key,
            agent=request,
            attempt=attempt,
            job=layer_job_name(task, task_revision.content_hash, key, attempt),
            expires_at=now + ttl,
            built_after=now - ttl,
        )
        layer._parent = head.id
        try:
            layer = await self.entity_client.create(layer)
        except NemoEntityConflictError:
            layer = await self.entity_client.get(AgentLayerEntity, name=layer.name, workspace=workspace, parent=head.id)
        await self._ensure_job(layer)
        return AgentLayer.from_entity(layer)

    async def _revision(self, workspace: str, name: str, revision: str) -> tuple[TaskEntity, TaskRevisionEntity] | None:
        try:
            head = await self.entity_client.get(TaskEntity, name=name, workspace=workspace)
        except NemoEntityNotFoundError:
            return None
        return head, await get_revision(self.entity_client, TaskRevisionEntity, head, revision)

    async def _layers(self, head: TaskEntity, revision_digest: str, agent_key: str | None) -> list[AgentLayerEntity]:
        return await _layers(self.entity_client, head, revision_digest, agent_key)

    async def _fail_abandoned(self, layer: AgentLayerEntity, job_status: str) -> AgentLayerEntity:
        layer.status = "failed"
        layer.error = f"agent-layer job {layer.job} ended ({job_status}) without a result"
        try:
            return await self.entity_client.update(layer)
        except NemoEntityConflictError:
            return await self.entity_client.get(
                AgentLayerEntity, name=layer.name, workspace=layer.workspace, parent=layer.parent
            )

    async def _ensure_job(self, layer: AgentLayerEntity) -> None:
        """Create the layer's Job, or confirm the Job already under its name is this layer's."""
        workspace, _, task = layer.task.partition("/")
        spec = AgentLayerBuildSpec(
            workspace=workspace,
            task=task,
            revision_digest=layer.revision_digest,
            agent_key=layer.agent_key,
            attempt=layer.attempt,
        )
        try:
            await self.jobs.create(agent_layer_build_job_request(spec, name=layer.job))
        except ConflictError:
            existing = await self.jobs.get(layer.job)
            if existing is None or not is_job_for(existing.custom_fields, spec):
                raise ImageBuildJobConflictError(f"Job {layer.job} exists and does not run this layer") from None
            return
        logger.info("Started agent layer %s of %s as job %s", layer.name, layer.task, layer.job)


async def newest_layer(
    entity_client: EntityClient, workspace: str, name: str, revision: str, agent_key: str
) -> AgentLayerEntity | None:
    """The revision's newest layer for ``agent_key``, in any status; None when it has none.

    Raises:
        NemoEntityNotFoundError: The task doesn't exist.
        RevisionNotFoundError: ``revision`` names no revision of the task.
    """
    head = await entity_client.get(TaskEntity, name=name, workspace=workspace)
    task_revision = await get_revision(entity_client, TaskRevisionEntity, head, revision)
    layers = await _layers(entity_client, head, task_revision.content_hash, agent_key)
    return layers[0] if layers else None


async def _layers(
    entity_client: EntityClient, head: TaskEntity, revision_digest: str, agent_key: str | None
) -> list[AgentLayerEntity]:
    operations = [
        ComparisonOperation(field="parent", operator=FilterOperator.EQ, value=head.id),
        ComparisonOperation(field="data.revision_digest", operator=FilterOperator.EQ, value=revision_digest),
    ]
    if agent_key is not None:
        operations.append(ComparisonOperation(field="data.agent_key", operator=FilterOperator.EQ, value=agent_key))
    result = await entity_client.list(
        AgentLayerEntity,
        workspace=head.workspace,
        filter_operation=LogicalOperation(operator=FilterOperator.AND, operations=operations),
        sort="-created_at",
        page=1,
        page_size=100,
    )
    return list(result.data)
