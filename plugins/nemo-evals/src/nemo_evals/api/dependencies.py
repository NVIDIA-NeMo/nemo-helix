# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the evaluator metrics API."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import Depends
from nemo_evals.api.service.metric_service import MetricService
from nemo_evals.api.service.result_service import ResultService
from nemo_evals.api.service.task_service import TaskService
from nemo_evals.api.service.taskset_run_service import TasksetRunService, normalize_run_caller
from nemo_evals.api.service.taskset_service import TasksetService
from nemo_evals.config import get_config
from nemo_evals.image_builds.service import ImageBuildService, JobsServiceBuildJobs, builder_authorizer
from nemo_evals.taskset_runs.store import TasksetRunStore
from nemo_helix_plugin.auth import current_auth_context
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.dependencies import (
    RequestAuthorizer,
    get_effective_principal_id,
    get_nemo_client,
    get_request_authorizer,
)
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity_client import NemoEntitiesClient, get_entity_client
from nemo_helix_plugin.files.client import AsyncFilesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient

_EVALS_SERVICE = "evals"


async def get_taskset_run_service(
    caller: str = Depends(get_effective_principal_id),
    authorize: RequestAuthorizer = Depends(get_request_authorizer),
) -> AsyncGenerator[TasksetRunService, None]:
    """Bind run acceptance to one caller-delegated platform client per request.

    Args:
        caller: Effective principal ID, normalized when empty in development mode.
        authorize: Request-scoped caller source-read authorizer.

    Yields:
        A service using raw acceptance clients and the existing entity resolver.
        The dependency exits the client's context when the request finishes.

    With auth on, the client delegates the caller's full principal (id, email, account id, groups, aliases):
    the Entity Store finds a caller's workspaces through all of them, so a member only through a group would
    otherwise be refused. The normalized ``caller`` id stays the idempotency scope and ``created_by``.
    """
    caller = normalize_run_caller(caller)
    principal = current_auth_context()
    on_behalf_of = principal.to_principal() if principal is not None else caller
    async with get_async_nemo_client(as_service="evals", internal=True, on_behalf_of=on_behalf_of) as async_client:
        entities_client = AsyncEntitiesClient.from_client(async_client)
        files_client = AsyncFilesClient.from_client(async_client)
        yield TasksetRunService(
            TasksetRunStore(entities_client, files_client),
            EntityClient(entities_client),
            async_client,
            authorize,
            get_config(),
            auth_context=principal,
        )


async def get_service_run_store() -> AsyncGenerator[TasksetRunStore, None]:
    """A run store acting as the evals service, for a route that rewrites a run after checking the caller.

    A run may hold fields only a service-principal client reads back (the creator's principal, once runs store it),
    so only it can rewrite the run without erasing them. The route's permission check, as the caller, scopes it to
    one workspace.
    """
    async with get_async_nemo_client(as_service="evals", internal=True) as async_client:
        yield TasksetRunStore(AsyncEntitiesClient.from_client(async_client), AsyncFilesClient.from_client(async_client))


def get_metric_service(
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    async_client: AsyncNemoClient = Depends(get_nemo_client),
) -> MetricService:
    """Provide a MetricService wired to the Entity Store and Files service."""
    return MetricService(entity_client, AsyncFilesClient.from_client(async_client))


def get_result_service(
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
) -> ResultService:
    """Provide a ResultService wired to the Entity Store (read-only over result entities)."""
    return ResultService(entity_client)


def get_task_service(
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    metric_service: MetricService = Depends(get_metric_service),
    async_client: AsyncNemoClient = Depends(get_nemo_client),
    authorize: RequestAuthorizer = Depends(get_request_authorizer),
) -> TaskService:
    """Provide a TaskService. It uses the MetricService to normalize inline task metrics into
    (derived) stored metrics, so a persisted task holds only references."""

    async def authorize_metric_read(workspace: str, name: str) -> None:
        """Check that the caller may read a metric a task references.

        Task write permission does not imply permission to read every metric bound to that task,
        especially across workspaces. The metric service uses service credentials, so storage
        access alone cannot enforce the caller's permissions.
        """
        await authorize("GET", f"/apis/evals/v2/workspaces/{quote(workspace, safe='')}/metrics/{quote(name, safe='')}")

    return TaskService(
        entity_client,
        metric_service,
        AsyncFilesClient.from_client(async_client),
        authorize_metric_read=authorize_metric_read,
    )


def get_image_build_service(
    workspace: str,
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    async_client: AsyncNemoClient = Depends(get_nemo_client),
    authorize: RequestAuthorizer = Depends(get_request_authorizer),
) -> ImageBuildService:
    """Provide an ImageBuildService whose task-builder Jobs are created as the caller."""
    return ImageBuildService(
        entity_client,
        JobsServiceBuildJobs(AsyncJobsClient.from_client(async_client), workspace=workspace),
        builder_authorizer(authorize),
    )


def get_taskset_service(
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    task_service: TaskService = Depends(get_task_service),
    authorize: RequestAuthorizer = Depends(get_request_authorizer),
) -> TasksetService:
    """Provide a TasksetService. It uses the TaskService to validate that each referenced task
    exists when a taskset is created."""

    async def authorize_task_read(workspace: str, name: str) -> None:
        """Provide a taskset service with caller-scoped authorization for referenced tasks.

        Taskset write permission does not imply permission to read every task placed in that taskset.
        Adapt the request-scoped HTTP authorizer to the task-read callback used while resolving each
        referenced task, preserving the caller's identity and token scopes.

        Permission to write the destination taskset does not imply permission to read referenced tasks—especially across workspaces.
        The entity client uses service credentials, so storage access alone cannot enforce the caller's permissions.
        """
        await authorize("GET", f"/apis/evals/v2/workspaces/{quote(workspace, safe='')}/tasks/{quote(name, safe='')}")

    return TasksetService(entity_client, task_service, authorize_task_read=authorize_task_read)


@asynccontextmanager
async def delegated_service_client(request_client: AsyncNemoClient) -> AsyncGenerator[AsyncNemoClient, None]:
    """Yield a client that calls the platform as the evals service, delegating the request's caller.

    Use it for calls only a service principal may make, such as reading a secret value; the target
    service still authorizes the delegated caller. Without a request principal there is nobody to
    delegate, so the request-scoped client is used rather than an undelegated service client.
    """
    caller = current_auth_context()
    if caller is None:
        yield request_client
        return
    async with get_async_nemo_client(
        as_service=_EVALS_SERVICE, internal=True, on_behalf_of=caller.to_principal()
    ) as service_client:
        yield service_client
