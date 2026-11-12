# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routes a task-builder Job uses to read and record its own build, under
/apis/evals/v2/workspaces/{workspace}/job-runtime.

Only the Job Evals started for a build may use them, and only for that build: Evals grants it the build in the service
attributes of its workload access tokens. The Evals service reads and writes the record for it, acting for the Job's
creator.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, status
from nemo_evals.authz import scope
from nemo_evals.content_hash import DIGEST_LENGTH, DIGEST_PATTERN
from nemo_evals.entities import MAX_NAME_LENGTH, NAME_PATTERN, TaskRevisionEntity
from nemo_evals.image_builds.models import TaskImageBuildEntity
from nemo_evals.image_builds.runner import EntityBuildRecords, TaskImageBuildSpec
from nemo_evals.image_builds.runtime import JOB_BUILD_PATH, TaskImageBuildProgress, TaskImageBuildState
from nemo_evals.jobs.task_image_build import TASK_IMAGE_BUILD_GRANT
from nemo_evals.revisions import RevisionNotFoundError
from nemo_helix_plugin.auth import RuntimeAuthClient, current_auth_client
from nemo_helix_plugin.auth.workload_delegations import current_workload_service_attributes
from nemo_helix_plugin.authz import CallerKind, path_rule
from nemo_helix_plugin.entities import EntityNotFoundError
from nemo_helix_plugin.entity_client import NemoEntitiesClient, get_entity_client

_EVALS_SERVICE = "evals"

router = APIRouter()

TaskName = Annotated[str, Path(max_length=MAX_NAME_LENGTH, pattern=NAME_PATTERN)]
RevisionDigest = Annotated[str, Path(min_length=DIGEST_LENGTH, max_length=DIGEST_LENGTH, pattern=DIGEST_PATTERN)]
Attempt = Annotated[int, Path(ge=1)]


async def _job_build(
    auth: RuntimeAuthClient, entities: NemoEntitiesClient, spec: TaskImageBuildSpec
) -> tuple[EntityBuildRecords, TaskImageBuildEntity, TaskRevisionEntity]:
    if auth.auth_enabled and auth.principal.id != f"service:{_EVALS_SERVICE}":
        granted = current_workload_service_attributes(_EVALS_SERVICE) or {}
        if granted.get(TASK_IMAGE_BUILD_GRANT) != spec.model_dump(mode="json"):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "Only the task-builder Job Evals started for this build may use this route"
            )
    records = EntityBuildRecords(entities, spec)
    try:
        build, revision = await records.load()
    except (EntityNotFoundError, RevisionNotFoundError) as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image build not found: {e}") from e
    return records, build, revision


_CALLERS = [CallerKind.PRINCIPAL, CallerKind.SERVICE_PRINCIPAL]


@router.get(JOB_BUILD_PATH, include_in_schema=False)
@scope.read
@path_rule(callers=_CALLERS, permissions=[])
async def load_job_image_build(
    workspace: str,
    task: TaskName,
    revision_digest: RevisionDigest,
    attempt: Attempt,
    auth: RuntimeAuthClient = Depends(current_auth_client),
    entities: NemoEntitiesClient = Depends(get_entity_client),
) -> TaskImageBuildState:
    """The Job's build record and the task revision it builds."""
    _, build, revision = await _job_build(
        auth,
        entities,
        TaskImageBuildSpec(workspace=workspace, task=task, revision_digest=revision_digest, attempt=attempt),
    )
    return TaskImageBuildState(build=build, revision=revision)


@router.put(JOB_BUILD_PATH, include_in_schema=False)
@scope.write
@path_rule(callers=_CALLERS, permissions=[])
async def record_job_image_build(
    workspace: str,
    task: TaskName,
    revision_digest: RevisionDigest,
    attempt: Attempt,
    progress: TaskImageBuildProgress,
    auth: RuntimeAuthClient = Depends(current_auth_client),
    entities: NemoEntitiesClient = Depends(get_entity_client),
) -> TaskImageBuildEntity:
    """Record the Job's progress on its build."""
    records, build, _ = await _job_build(
        auth,
        entities,
        TaskImageBuildSpec(workspace=workspace, task=task, revision_digest=revision_digest, attempt=attempt),
    )
    build.status = progress.status
    build.images = progress.images
    build.error = progress.error
    return await records.save(build)
