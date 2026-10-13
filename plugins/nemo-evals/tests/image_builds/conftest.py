# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import Mock

import pytest
from nemo_evals.api.schemas import EvaluatorTaskDefinition, HarborTaskDefinition, MetricRef, TaskInput, TaskInputs
from nemo_evals.api.service.task_service import TaskService
from nemo_evals.api.task_definitions.harbor import HarborArchiveSource, HarborTaskHash
from nemo_evals.entities import TaskEntity, TaskRevisionEntity
from nemo_evals.revisions import get_revision

ARCHIVE_REF = "default/harbor-tasks#sort-task/abc/task_archive"
ARCHIVE_DIGEST = "a" * 64

StoreTask = Callable[..., Awaitable[str]]


class _Metrics:
    async def store_derived_metric(self, metric, *, workspace: str) -> MetricRef:
        return MetricRef(f"{workspace}/derived")

    async def get_metric(self, workspace: str, name: str) -> object | None:
        return object()


@pytest.fixture
def store_task(entity_store, monkeypatch) -> StoreTask:
    """Store a task in ``entity_store`` and return its revision digest."""
    from nhx_evals_sdk.agent_eval.runtimes.harbor.archive import NativeTask

    async def verify(spec, files_client):
        return NativeTask("sort-task", spec.instruction, spec.config)

    monkeypatch.setattr("nemo_evals.api.service.task_service.verify_definition", verify)
    service = TaskService(entity_store, _Metrics(), files_client=Mock())

    async def store(name: str = "sort-task", *, config: dict | None = None, harbor: bool = True) -> str:
        spec = (
            HarborTaskDefinition(
                kind="harbor",
                native_task_id="sort-task",
                harbor_hash=HarborTaskHash(digest="b" * 64, harbor_version="0.20.0"),
                source=HarborArchiveSource(fileset_ref=ARCHIVE_REF, files_hash=ARCHIVE_DIGEST),
                config=config or {},
            )
            if harbor
            else EvaluatorTaskDefinition(
                kind="evaluator",
                intent="Answer.",
                inputs=TaskInputs(instruction="2+2?"),
                metrics=[MetricRef("default/m")],
            )
        )
        await service.create_task(name, TaskInput(spec=spec), workspace="default")
        head = await entity_store.get(TaskEntity, name=name, workspace="default")
        return (await get_revision(entity_store, TaskRevisionEntity, head)).content_hash

    return store
