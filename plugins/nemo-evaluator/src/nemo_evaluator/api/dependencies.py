# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the evaluator metrics API."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import Depends
from nemo_evaluator.api.service.metric_service import MetricService
from nemo_evaluator.api.service.result_service import ResultService
from nemo_evaluator.api.service.task_service import TaskService
from nemo_evaluator.api.service.taskset_service import TasksetService
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.dependencies import RequestAuthorizer, get_nemo_client, get_request_authorizer
from nemo_helix_plugin.entity_client import NemoEntitiesClient, get_entity_client
from nemo_helix_plugin.files.client import AsyncFilesClient


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
) -> TaskService:
    """Provide a TaskService. It uses the MetricService to normalize inline task metrics into
    (derived) stored metrics, so a persisted task holds only references."""
    return TaskService(entity_client, metric_service, AsyncFilesClient.from_client(async_client))


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
        await authorize(
            "GET", f"/apis/evaluator/v2/workspaces/{quote(workspace, safe='')}/tasks/{quote(name, safe='')}"
        )

    return TasksetService(entity_client, task_service, authorize_task_read=authorize_task_read)
