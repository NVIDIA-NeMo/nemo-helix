# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A task-builder Job's build record, read and written through the Evals job-runtime routes."""

from __future__ import annotations

from abc import abstractmethod

from nemo_evals.entities import TaskRevisionEntity
from nemo_evals.image_builds.models import BuildImage, BuildStatus, TaskImageBuildEntity
from nemo_evals.image_builds.runner import TaskImageBuildSpec
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.endpoint import get, put
from nemo_helix_plugin.client.method import method
from pydantic import BaseModel, ConfigDict

JOB_BUILD_PATH = "/job-runtime/{job}/tasks/{task}/revisions/{revision_digest}/image-builds/{attempt}"
_PATH = "/apis/evals/v2/workspaces/{workspace}" + JOB_BUILD_PATH


class TaskImageBuildState(BaseModel):
    """A build record and the task revision it builds."""

    build: TaskImageBuildEntity
    revision: TaskRevisionEntity


class TaskImageBuildProgress(BaseModel):
    """What a task-builder Job may change on its build record."""

    model_config = ConfigDict(extra="forbid")

    status: BuildStatus
    images: list[BuildImage]
    error: str | None = None


@get(_PATH)
@abstractmethod
def _load(*, workspace: str, job: str, task: str, revision_digest: str, attempt: str) -> TaskImageBuildState: ...


@put(_PATH)
@abstractmethod
def _save(
    *, workspace: str, job: str, task: str, revision_digest: str, attempt: str, body: TaskImageBuildProgress
) -> TaskImageBuildEntity: ...


class _RuntimeClient(AsyncNemoClient):
    load = method(_load)
    save = method(_save)


class RuntimeBuildRecords:
    """The build record of one task-builder Job, through the Evals job-runtime routes."""

    def __init__(self, client: AsyncNemoClient, *, job: str, spec: TaskImageBuildSpec) -> None:
        self._client = _RuntimeClient.from_client(client)
        self._path = {
            "workspace": spec.workspace,
            "job": job,
            "task": spec.task,
            "revision_digest": spec.revision_digest,
            "attempt": str(spec.attempt),
        }

    async def load(self) -> tuple[TaskImageBuildEntity, TaskRevisionEntity]:
        state = (await self._client.load(**self._path)).data()
        return state.build, state.revision

    async def save(self, build: TaskImageBuildEntity) -> TaskImageBuildEntity:
        progress = TaskImageBuildProgress(status=build.status, images=build.images, error=build.error)
        return (await self._client.save(**self._path, body=progress)).data()
