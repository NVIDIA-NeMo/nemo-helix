# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI dependencies for the evaluator metrics API."""

from __future__ import annotations

from fastapi import Depends
from nemo_evaluator.api.service.metric_service import MetricService
from nemo_evaluator.api.service.result_service import ResultService
from nemo_evaluator.api.service.task_service import TaskService
from nemo_evaluator.api.service.taskset_service import TasksetService
from nemo_platform_plugin.client.client import AsyncNemoClient
from nemo_platform_plugin.dependencies import get_nemo_client
from nemo_platform_plugin.entity_client import NemoEntitiesClient, get_entity_client
from nemo_platform_plugin.files.client import AsyncFilesClient


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
) -> TaskService:
    """Provide a TaskService. It uses the MetricService to normalize inline task metrics into
    (derived) stored metrics, so a persisted task holds only references."""
    return TaskService(entity_client, metric_service)


def get_taskset_service(
    entity_client: NemoEntitiesClient = Depends(get_entity_client),
    task_service: TaskService = Depends(get_task_service),
) -> TasksetService:
    """Provide a TasksetService. It uses the TaskService to validate that each referenced task
    exists when a taskset is created."""
    return TasksetService(entity_client, task_service)
