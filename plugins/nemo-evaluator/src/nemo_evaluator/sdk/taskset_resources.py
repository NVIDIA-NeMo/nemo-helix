# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDK resources for managing stored tasksets (``client.evaluator.tasksets``).

Thin client over the evaluator service's ``/tasksets`` create/get/list/delete API. A taskset is sent
as a :class:`TasksetInput` (its members as references to stored tasks) and returned as the
:class:`Taskset` DTO; the service owns persistence in the entity store.
"""

from __future__ import annotations

from nemo_evaluator.api.schemas import Revision, Taskset, TasksetInput
from nemo_evaluator.sdk.query_params import list_params, project_params, revision_selector
from nemo_helix_plugin.evaluator.client import AsyncEvaluatorClient, EvaluatorClient
from nemo_helix_plugin.evaluator.types import CreateTasksetRequest, ReplaceTasksetRequest
from nemo_helix_plugin.schema import Page


class EvaluatorTasksetsResource:
    """Sync resource mounted as ``client.evaluator.tasksets``."""

    def __init__(self, client: EvaluatorClient) -> None:
        self._client = client

    def create(
        self, name: str, *, taskset: TasksetInput, project: str | None = None, workspace: str | None = None
    ) -> Taskset:
        """Store a new taskset (addressed by workspace/name)."""
        response = self._client.create_taskset(
            name=name,
            workspace=workspace,
            body=CreateTasksetRequest(root=taskset.model_dump(mode="json")),
            query_params=project_params(project),
        )
        return Taskset.model_validate(response.data().model_dump(mode="json"))

    def replace(
        self, name: str, *, taskset: TasksetInput, project: str | None = None, workspace: str | None = None
    ) -> Taskset:
        """Publish a revision of a taskset, creating it if absent.

        Members are re-resolved to exact revision digests on every call, so identical member names
        can still publish a new revision if a member task published in the meantime.

        The response body is the same either way, so this does not report whether a revision was
        cut — the server signals that with 201 vs 200, which is discarded here. Compare the returned
        ``revision`` against a prior read if you need to know."""
        response = self._client.replace_taskset(
            name=name,
            workspace=workspace,
            body=ReplaceTasksetRequest(root=taskset.model_dump(mode="json")),
            query_params=project_params(project),
        )
        return Taskset.model_validate(response.data().model_dump(mode="json"))

    def list_revisions(
        self, name: str, *, page: int = 1, page_size: int = 100, workspace: str | None = None
    ) -> Page[Revision]:
        """List a taskset's published revisions, newest first."""
        response = self._client.list_taskset_revisions(
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

    def tag(self, name: str, *, tag: str, revision: str, workspace: str | None = None) -> Taskset:
        """Point ``tag`` at an existing revision, named by digest or by another tag.

        Both selectors are keyword-only: ``tag`` names the pointer being written and ``revision``
        names what it points at, and two bare strings in a row gave no hint which was which.
        """
        response = self._client.tag_taskset_revision(
            name=name,
            tag=tag,
            workspace=workspace,
            query_params={"revision": revision},
        )
        return Taskset.model_validate(response.data().model_dump(mode="json"))

    def retrieve(
        self, name: str, *, revision: str | None = None, tag: str | None = None, workspace: str | None = None
    ) -> Taskset:
        """Get a stored taskset by name, or as of a published revision.

        Pass ``revision`` for a content digest or ``tag`` for a named pointer — not both. With
        neither, this returns the taskset's current membership.
        """
        selector = revision_selector(revision, tag)
        if selector is not None:
            response = self._client.get_taskset_revision(name=name, revision=selector, workspace=workspace)
        else:
            response = self._client.get_taskset(name=name, workspace=workspace)
        return Taskset.model_validate(response.data().model_dump(mode="json"))

    def list(
        self, *, workspace: str | None = None, page: int = 1, page_size: int = 100, sort: str | None = None
    ) -> Page[Taskset]:
        """List stored tasksets in a workspace."""
        response = self._client.list_tasksets(
            workspace=workspace,
            query_params=list_params(page, page_size, sort),
        )
        page_result = response.page()
        return Page[Taskset].model_validate(
            {
                "data": [taskset.model_dump(mode="json") for taskset in page_result.items],
                "pagination": page_result.metadata,
                "sort": sort,
                "filter": None,
            }
        )

    def delete(self, name: str, *, workspace: str | None = None) -> None:
        """Delete a stored taskset by name."""
        self._client.delete_taskset(name=name, workspace=workspace).data()


class AsyncEvaluatorTasksetsResource:
    """Async resource mounted as ``client.evaluator.tasksets``."""

    def __init__(self, client: AsyncEvaluatorClient) -> None:
        self._client = client

    async def create(
        self, name: str, *, taskset: TasksetInput, project: str | None = None, workspace: str | None = None
    ) -> Taskset:
        """Store a new taskset (addressed by workspace/name)."""
        response = await self._client.create_taskset(
            name=name,
            workspace=workspace,
            body=CreateTasksetRequest(root=taskset.model_dump(mode="json")),
            query_params=project_params(project),
        )
        return Taskset.model_validate(response.data().model_dump(mode="json"))

    async def replace(
        self, name: str, *, taskset: TasksetInput, project: str | None = None, workspace: str | None = None
    ) -> Taskset:
        """Publish a revision of a taskset, creating it if absent.

        Members are re-resolved to exact revision digests on every call, so identical member names
        can still publish a new revision if a member task published in the meantime.

        The response body is the same either way, so this does not report whether a revision was
        cut — the server signals that with 201 vs 200, which is discarded here. Compare the returned
        ``revision`` against a prior read if you need to know."""
        response = await self._client.replace_taskset(
            name=name,
            workspace=workspace,
            body=ReplaceTasksetRequest(root=taskset.model_dump(mode="json")),
            query_params=project_params(project),
        )
        return Taskset.model_validate(response.data().model_dump(mode="json"))

    async def list_revisions(
        self, name: str, *, page: int = 1, page_size: int = 100, workspace: str | None = None
    ) -> Page[Revision]:
        """List a taskset's published revisions, newest first."""
        response = await self._client.list_taskset_revisions(
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

    async def tag(self, name: str, *, tag: str, revision: str, workspace: str | None = None) -> Taskset:
        """Point ``tag`` at an existing revision, named by digest or by another tag.

        Both selectors are keyword-only: ``tag`` names the pointer being written and ``revision``
        names what it points at, and two bare strings in a row gave no hint which was which.
        """
        response = await self._client.tag_taskset_revision(
            name=name,
            tag=tag,
            workspace=workspace,
            query_params={"revision": revision},
        )
        return Taskset.model_validate(response.data().model_dump(mode="json"))

    async def retrieve(
        self, name: str, *, revision: str | None = None, tag: str | None = None, workspace: str | None = None
    ) -> Taskset:
        """Get a stored taskset by name, or as of a published revision.

        Pass ``revision`` for a content digest or ``tag`` for a named pointer — not both. With
        neither, this returns the taskset's current membership.
        """
        selector = revision_selector(revision, tag)
        if selector is not None:
            response = await self._client.get_taskset_revision(name=name, revision=selector, workspace=workspace)
        else:
            response = await self._client.get_taskset(name=name, workspace=workspace)
        return Taskset.model_validate(response.data().model_dump(mode="json"))

    async def list(
        self, *, workspace: str | None = None, page: int = 1, page_size: int = 100, sort: str | None = None
    ) -> Page[Taskset]:
        """List stored tasksets in a workspace."""
        response = await self._client.list_tasksets(
            workspace=workspace,
            query_params=list_params(page, page_size, sort),
        )
        page_result = response.page()
        return Page[Taskset].model_validate(
            {
                "data": [taskset.model_dump(mode="json") for taskset in page_result.items],
                "pagination": page_result.metadata,
                "sort": sort,
                "filter": None,
            }
        )

    async def delete(self, name: str, *, workspace: str | None = None) -> None:
        """Delete a stored taskset by name."""
        response = await self._client.delete_taskset(name=name, workspace=workspace)
        response.data()
