# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for managing stored agent-eval tasks (``client.evaluator.tasks``).

Client wrappers over the evaluator service's ``/tasks`` create/get/list/delete API. A task is sent as a
:class:`TaskInput` (its metrics inline and/or as references to stored metrics) and returned as the
:class:`Task` DTO; the service owns persistence in the entity store.

Local sources are prepared and verified before the resource writes a task entity.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any, overload

from nemo_evaluator.api.fields import TaskRef
from nemo_evaluator.api.schemas import Revision, Task, TaskInput
from nemo_evaluator.entities import MAX_NAME_LENGTH, NAME_PATTERN
from nemo_evaluator.sdk.query_params import list_params, project_params, revision_selector
from nemo_evaluator.sdk.task_preparation import TaskPublicationError, prepare_task, prepare_task_async
from nemo_evaluator.shared.metric_bundles.bundles import MetricBundlePackager
from nemo_evaluator_sdk.agent_eval.runtimes.harbor_tasks import HarborAgentEvalTask
from nemo_evaluator_sdk.agent_eval.tasks import AgentEvalTask
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_helix_plugin.evaluator.types import CreateTaskRequest, ReplaceTaskRequest
from nemo_helix_plugin.files.client import AsyncFilesClient, FilesClient
from nemo_helix_plugin.schema import Page


class EvaluatorTasksResource:
    """Sync resource mounted as ``client.evaluator.tasks``."""

    def __init__(self, client: EvaluatorClient) -> None:
        self._client = client

    def _write(
        self,
        name: str,
        *,
        task: TaskInput | AgentEvalTask,
        project: str | None,
        workspace: str | None,
        metric_bundle_packager: MetricBundlePackager | None,
        fileset_ref: str | None,
        path_prefix: str | None,
        send: Callable[..., Any],
        request: type[CreateTaskRequest] | type[ReplaceTaskRequest],
    ) -> Task:
        """Shared body of ``create`` and ``replace``; ``send`` and ``request`` select the endpoint."""
        workspace = self._client.resolve_workspace(workspace)
        if len(name) > MAX_NAME_LENGTH or re.fullmatch(NAME_PATTERN, name) is None:
            raise ValueError("Invalid task registration name")
        TaskRef(f"{workspace}/{name}")
        needs_preparation = isinstance(task, AgentEvalTask)
        if needs_preparation:
            prepared = prepare_task(
                task,
                files_client=FilesClient.from_client(self._client),
                workspace=workspace,
                fileset_ref=fileset_ref,
                path_prefix=path_prefix,
                metric_bundle_packager=metric_bundle_packager,
            )
        else:
            if any(value is not None for value in (fileset_ref, path_prefix, metric_bundle_packager)):
                raise ValueError("Preparation options cannot be used with TaskInput")
            prepared = task
        try:
            response = send(
                name=name,
                workspace=workspace,
                body=request(root=prepared.model_dump(mode="json")),
                query_params=project_params(project),
            )
            return Task.model_validate(response.data().model_dump(mode="json"))
        except Exception as exc:
            if needs_preparation:
                raise TaskPublicationError(prepared) from exc
            raise

    @overload
    def create(
        self,
        name: str,
        *,
        task: HarborAgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
    ) -> Task: ...

    @overload
    def create(
        self,
        name: str,
        *,
        task: AgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> Task: ...

    @overload
    def create(
        self, name: str, *, task: TaskInput, project: str | None = None, workspace: str | None = None
    ) -> Task: ...

    def create(
        self,
        name: str,
        *,
        task: TaskInput | AgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
    ) -> Task:
        """Prepare and create a task; an existing name conflicts.

        Algorithm:
            - Resolve the workspace and validate the registration name.
            - Prepare local sources, or validate that a ready ``TaskInput`` has no preparation options.
            - Create the entity and preserve prepared input in ``TaskPublicationError`` if registration fails.
        """
        return self._write(
            name,
            task=task,
            project=project,
            workspace=workspace,
            metric_bundle_packager=metric_bundle_packager,
            fileset_ref=fileset_ref,
            path_prefix=path_prefix,
            send=self._client.create_task,
            request=CreateTaskRequest,
        )

    @overload
    def replace(
        self,
        name: str,
        *,
        task: HarborAgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
    ) -> Task: ...

    @overload
    def replace(
        self,
        name: str,
        *,
        task: AgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> Task: ...

    @overload
    def replace(
        self, name: str, *, task: TaskInput, project: str | None = None, workspace: str | None = None
    ) -> Task: ...

    def replace(
        self,
        name: str,
        *,
        task: TaskInput | AgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
    ) -> Task:
        """Prepare and replace a task, creating it when absent.

        Algorithm:
            - Resolve the workspace and validate the registration name.
            - Prepare local sources, or validate that a ready ``TaskInput`` has no preparation options.
            - Upsert the entity and preserve prepared input in ``TaskPublicationError`` if registration fails.
        """
        return self._write(
            name,
            task=task,
            project=project,
            workspace=workspace,
            metric_bundle_packager=metric_bundle_packager,
            fileset_ref=fileset_ref,
            path_prefix=path_prefix,
            send=self._client.replace_task,
            request=ReplaceTaskRequest,
        )

    @overload
    def prepare(
        self,
        task: HarborAgentEvalTask,
        *,
        workspace: str | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> TaskInput: ...

    @overload
    def prepare(
        self,
        task: AgentEvalTask,
        *,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> TaskInput: ...

    def prepare(
        self,
        task: AgentEvalTask,
        *,
        workspace: str | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> TaskInput:
        """Prepare scoring and upload/verify Harbor content without creating entities."""
        return prepare_task(
            task,
            files_client=FilesClient.from_client(self._client),
            workspace=self._client.resolve_workspace(workspace),
            fileset_ref=fileset_ref,
            path_prefix=path_prefix,
            metric_bundle_packager=metric_bundle_packager,
        )

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

    async def _write(
        self,
        name: str,
        *,
        task: TaskInput | AgentEvalTask,
        project: str | None,
        workspace: str | None,
        metric_bundle_packager: MetricBundlePackager | None,
        fileset_ref: str | None,
        path_prefix: str | None,
        send: Callable[..., Awaitable[Any]],
        request: type[CreateTaskRequest] | type[ReplaceTaskRequest],
    ) -> Task:
        """Shared body of ``create`` and ``replace``; ``send`` and ``request`` select the endpoint."""
        workspace = self._client.resolve_workspace(workspace)
        if len(name) > MAX_NAME_LENGTH or re.fullmatch(NAME_PATTERN, name) is None:
            raise ValueError("Invalid task registration name")
        TaskRef(f"{workspace}/{name}")
        needs_preparation = isinstance(task, AgentEvalTask)
        if needs_preparation:
            prepared = await prepare_task_async(
                task,
                files_client=AsyncFilesClient.from_client(self._client),
                workspace=workspace,
                fileset_ref=fileset_ref,
                path_prefix=path_prefix,
                metric_bundle_packager=metric_bundle_packager,
            )
        else:
            if any(value is not None for value in (fileset_ref, path_prefix, metric_bundle_packager)):
                raise ValueError("Preparation options cannot be used with TaskInput")
            prepared = task
        try:
            response = await send(
                name=name,
                workspace=workspace,
                body=request(root=prepared.model_dump(mode="json")),
                query_params=project_params(project),
            )
            return Task.model_validate(response.data().model_dump(mode="json"))
        except Exception as exc:
            if needs_preparation:
                raise TaskPublicationError(prepared) from exc
            raise

    @overload
    async def create(
        self,
        name: str,
        *,
        task: HarborAgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
    ) -> Task: ...

    @overload
    async def create(
        self,
        name: str,
        *,
        task: AgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> Task: ...

    @overload
    async def create(
        self, name: str, *, task: TaskInput, project: str | None = None, workspace: str | None = None
    ) -> Task: ...

    async def create(
        self,
        name: str,
        *,
        task: TaskInput | AgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
    ) -> Task:
        """Asynchronously prepare and create a task; an existing name conflicts.

        Algorithm:
            - Resolve the workspace and validate the registration name.
            - Prepare local sources asynchronously, or reject preparation options for a ready ``TaskInput``.
            - Create the entity and preserve prepared input in ``TaskPublicationError`` if registration fails.
        """
        return await self._write(
            name,
            task=task,
            project=project,
            workspace=workspace,
            metric_bundle_packager=metric_bundle_packager,
            fileset_ref=fileset_ref,
            path_prefix=path_prefix,
            send=self._client.create_task,
            request=CreateTaskRequest,
        )

    @overload
    async def replace(
        self,
        name: str,
        *,
        task: HarborAgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
    ) -> Task: ...

    @overload
    async def replace(
        self,
        name: str,
        *,
        task: AgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> Task: ...

    @overload
    async def replace(
        self, name: str, *, task: TaskInput, project: str | None = None, workspace: str | None = None
    ) -> Task: ...

    async def replace(
        self,
        name: str,
        *,
        task: TaskInput | AgentEvalTask,
        project: str | None = None,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
    ) -> Task:
        """Asynchronously prepare and replace a task, creating it when absent.

        Algorithm:
            - Resolve the workspace and validate the registration name.
            - Prepare local sources asynchronously, or reject preparation options for a ready ``TaskInput``.
            - Upsert the entity and preserve prepared input in ``TaskPublicationError`` if registration fails.
        """
        return await self._write(
            name,
            task=task,
            project=project,
            workspace=workspace,
            metric_bundle_packager=metric_bundle_packager,
            fileset_ref=fileset_ref,
            path_prefix=path_prefix,
            send=self._client.replace_task,
            request=ReplaceTaskRequest,
        )

    @overload
    async def prepare(
        self,
        task: HarborAgentEvalTask,
        *,
        workspace: str | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> TaskInput: ...

    @overload
    async def prepare(
        self,
        task: AgentEvalTask,
        *,
        workspace: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> TaskInput: ...

    async def prepare(
        self,
        task: AgentEvalTask,
        *,
        workspace: str | None = None,
        fileset_ref: str | None = None,
        path_prefix: str | None = None,
        metric_bundle_packager: MetricBundlePackager | None = None,
    ) -> TaskInput:
        """Prepare scoring and upload/verify Harbor content without creating entities."""
        return await prepare_task_async(
            task,
            files_client=AsyncFilesClient.from_client(self._client),
            workspace=self._client.resolve_workspace(workspace),
            fileset_ref=fileset_ref,
            path_prefix=path_prefix,
            metric_bundle_packager=metric_bundle_packager,
        )

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
