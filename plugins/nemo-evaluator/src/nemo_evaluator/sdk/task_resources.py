# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for managing stored agent-eval tasks (``client.evaluator.tasks``).

Thin client over the evaluator service's ``/tasks`` create/get/list/delete API. A task is sent as a
:class:`TaskInput` (its metrics inline and/or as references to stored metrics) and returned as the
:class:`Task` DTO; the service owns persistence in the entity store.
"""

from __future__ import annotations

from nemo_evaluator.api.schemas import Revision, Task, TaskInput
from nemo_evaluator.sdk.query_params import list_params, project_params, revision_selector
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_helix_plugin.evaluator.types import CreateTaskRequest, ReplaceTaskRequest
from nemo_helix_plugin.schema import Page


class EvaluatorTasksResource:
    """Sync resource mounted as ``client.evaluator.tasks``."""

    def __init__(self, client: EvaluatorClient) -> None:
        self._client = client

    def create(self, name: str, *, task: TaskInput, project: str | None = None, workspace: str | None = None) -> Task:
        """Store a new task (addressed by workspace/name)."""
        response = self._client.create_task(
            name=name,
            workspace=workspace,
            body=CreateTaskRequest(root=task.model_dump(mode="json")),
            query_params=project_params(project),
        )
        return Task.model_validate(response.data().model_dump(mode="json"))

    def replace(self, name: str, *, task: TaskInput, project: str | None = None, workspace: str | None = None) -> Task:
        """Publish a revision of a task, creating it if absent.

        Upsert, so a publisher needs no existence check. Submitting content identical to the current
        revision publishes nothing and returns the task unchanged.

        The response body is the same either way, so this does not report whether a revision was
        cut — the server signals that with 201 vs 200, which is discarded here. Compare the returned
        ``revision`` against a prior read if you need to know."""
        response = self._client.replace_task(
            name=name,
            workspace=workspace,
            body=ReplaceTaskRequest(root=task.model_dump(mode="json")),
            query_params=project_params(project),
        )
        return Task.model_validate(response.data().model_dump(mode="json"))

    def list_revisions(
        self, name: str, *, page: int = 1, page_size: int = 100, workspace: str | None = None
    ) -> Page[Revision]:
        """List a task's published revisions, newest first."""
        response = self._client.list_task_revisions(
            name=name,
            workspace=workspace,
            query_params={"page": page, "page_size": page_size},
        )
        page_result = response.page()
        return Page[Revision].model_validate(
            {
                "data": [revision.model_dump(mode="json") for revision in page_result.items],
                "pagination": page_result.metadata,
            }
        )

    def tag(self, name: str, *, tag: str, revision: str, workspace: str | None = None) -> Task:
        """Point ``tag`` at an existing revision, named by digest or by another tag.

        Both selectors are keyword-only: ``tag`` names the pointer being written and ``revision``
        names what it points at, and two bare strings in a row gave no hint which was which.
        """
        response = self._client.tag_task_revision(
            name=name,
            tag=tag,
            workspace=workspace,
            query_params={"revision": revision},
        )
        return Task.model_validate(response.data().model_dump(mode="json"))

    def retrieve(
        self, name: str, *, revision: str | None = None, tag: str | None = None, workspace: str | None = None
    ) -> Task:
        """Get a stored task by name, or as of a published revision.

        Pass ``revision`` for a content digest or ``tag`` for a named pointer — not both. With
        neither, this returns the task's current content.
        """
        selector = revision_selector(revision, tag)
        if selector is not None:
            response = self._client.get_task_revision(name=name, revision=selector, workspace=workspace)
        else:
            response = self._client.get_task(name=name, workspace=workspace)
        return Task.model_validate(response.data().model_dump(mode="json"))

    def list(
        self, *, workspace: str | None = None, page: int = 1, page_size: int = 100, sort: str | None = None
    ) -> Page[Task]:
        """List stored tasks in a workspace."""
        response = self._client.list_tasks(
            workspace=workspace,
            query_params=list_params(page, page_size, sort),
        )
        page_result = response.page()
        return Page[Task].model_validate(
            {
                "data": [task.model_dump(mode="json") for task in page_result.items],
                "pagination": page_result.metadata,
                "sort": sort,
                "filter": None,
            }
        )

    def delete(self, name: str, *, workspace: str | None = None) -> None:
        """Delete a stored task by name."""
        self._client.delete_task(name=name, workspace=workspace).data()


class AsyncEvaluatorTasksResource:
    """Async resource mounted as ``client.evaluator.tasks``."""

    def __init__(self, client: AsyncEvaluatorClient) -> None:
        self._client = client

    async def create(
        self, name: str, *, task: TaskInput, project: str | None = None, workspace: str | None = None
    ) -> Task:
        """Store a new task (addressed by workspace/name)."""
        response = await self._client.create_task(
            name=name,
            workspace=workspace,
            body=CreateTaskRequest(root=task.model_dump(mode="json")),
            query_params=project_params(project),
        )
        return Task.model_validate(response.data().model_dump(mode="json"))

    async def replace(
        self, name: str, *, task: TaskInput, project: str | None = None, workspace: str | None = None
    ) -> Task:
        """Publish a revision of a task, creating it if absent.

        Upsert, so a publisher needs no existence check. Submitting content identical to the current
        revision publishes nothing and returns the task unchanged.

        The response body is the same either way, so this does not report whether a revision was
        cut — the server signals that with 201 vs 200, which is discarded here. Compare the returned
        ``revision`` against a prior read if you need to know."""
        response = await self._client.replace_task(
            name=name,
            workspace=workspace,
            body=ReplaceTaskRequest(root=task.model_dump(mode="json")),
            query_params=project_params(project),
        )
        return Task.model_validate(response.data().model_dump(mode="json"))

    async def list_revisions(
        self, name: str, *, page: int = 1, page_size: int = 100, workspace: str | None = None
    ) -> Page[Revision]:
        """List a task's published revisions, newest first."""
        response = await self._client.list_task_revisions(
            name=name,
            workspace=workspace,
            query_params={"page": page, "page_size": page_size},
        )
        page_result = response.page()
        return Page[Revision].model_validate(
            {
                "data": [revision.model_dump(mode="json") for revision in page_result.items],
                "pagination": page_result.metadata,
            }
        )

    async def tag(self, name: str, *, tag: str, revision: str, workspace: str | None = None) -> Task:
        """Point ``tag`` at an existing revision, named by digest or by another tag.

        Both selectors are keyword-only: ``tag`` names the pointer being written and ``revision``
        names what it points at, and two bare strings in a row gave no hint which was which.
        """
        response = await self._client.tag_task_revision(
            name=name,
            tag=tag,
            workspace=workspace,
            query_params={"revision": revision},
        )
        return Task.model_validate(response.data().model_dump(mode="json"))

    async def retrieve(
        self, name: str, *, revision: str | None = None, tag: str | None = None, workspace: str | None = None
    ) -> Task:
        """Get a stored task by name, or as of a published revision.

        Pass ``revision`` for a content digest or ``tag`` for a named pointer — not both. With
        neither, this returns the task's current content.
        """
        selector = revision_selector(revision, tag)
        if selector is not None:
            response = await self._client.get_task_revision(name=name, revision=selector, workspace=workspace)
        else:
            response = await self._client.get_task(name=name, workspace=workspace)
        return Task.model_validate(response.data().model_dump(mode="json"))

    async def list(
        self, *, workspace: str | None = None, page: int = 1, page_size: int = 100, sort: str | None = None
    ) -> Page[Task]:
        """List stored tasks in a workspace."""
        response = await self._client.list_tasks(
            workspace=workspace,
            query_params=list_params(page, page_size, sort),
        )
        page_result = response.page()
        return Page[Task].model_validate(
            {
                "data": [task.model_dump(mode="json") for task in page_result.items],
                "pagination": page_result.metadata,
                "sort": sort,
                "filter": None,
            }
        )

    async def delete(self, name: str, *, workspace: str | None = None) -> None:
        """Delete a stored task by name."""
        response = await self._client.delete_task(name=name, workspace=workspace)
        response.data()
