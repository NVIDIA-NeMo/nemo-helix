# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``builder-queue``: start queued build jobs, oldest first, while fewer than ``max_concurrent_builds`` run.

A submit with the cap set writes a :class:`QueuedBuild` instead of creating the job. Each pass this controller
forgets the started builds whose jobs ended (failing any image they left unpublished), then creates jobs for the oldest queued ones until the cap is met,
each as the user who submitted it, and moves that build's images from ``queued`` to ``pending``.

Every step is safe to repeat: a job that already exists for the same request is adopted, so a pass that dies
between creating a job and marking its build started just finishes the work on the next pass.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import ClassVar

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, QueuedBuild
from nemo_builder_plugin.submit import fail_rows
from nemo_helix_plugin.auth import RuntimePrincipal
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client.errors import ConflictError, NemoHTTPError, NotFoundError
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities import EntityClient, EntityConflictError, EntityNotFoundError
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

_ENDED = set(HelixJobStatus.terminals())
ClientFactory = Callable[[str | RuntimePrincipal | None], AbstractAsyncContextManager[AsyncNemoClient]]


@asynccontextmanager
async def _client(on_behalf_of: str | RuntimePrincipal | None) -> AsyncGenerator[AsyncNemoClient]:
    async with get_async_nemo_client(as_service="builder", internal=True, on_behalf_of=on_behalf_of) as client:
        yield client


def _submitter(build: QueuedBuild) -> str | RuntimePrincipal | None:
    """Whom the job is created as: the stored principal, else (auth off) the creator's id."""
    if build.auth_context is not None:
        return build.auth_context.to_principal()
    return build.created_by


class BuildQueueController(NemoController):
    """Start queued build jobs while fewer than ``max_concurrent_builds`` run (see module docstring)."""

    name: ClassVar[str] = "builder-queue"
    dependencies: ClassVar[list[str]] = ["entities", "jobs"]

    def __init__(self, client_factory: ClientFactory = _client, cap: Callable[[], int | None] | None = None) -> None:
        super().__init__()
        self._client_factory = client_factory
        self._cap = cap or (lambda: BuilderConfig.get().max_concurrent_builds)
        self._context: AbstractAsyncContextManager[AsyncNemoClient] | None = None
        self._entities: EntityClient | None = None
        self._jobs: AsyncJobsClient | None = None

    @property
    def interval_seconds(self) -> float:
        return 10.0

    async def on_startup(self) -> None:
        self._context = self._client_factory(None)
        service = await self._context.__aenter__()
        self._entities = EntityClient(AsyncEntitiesClient.from_client(service))
        self._jobs = AsyncJobsClient.from_client(service)

    async def on_shutdown(self) -> None:
        if self._context is not None:
            await self._context.__aexit__(None, None, None)

    async def list_objects(self) -> list:
        """Unused: ``reconcile`` works over the whole queue at once, to count what runs."""
        return []

    async def reconcile_one(self, obj: object) -> None:
        raise NotImplementedError

    async def reconcile(self) -> None:
        # ponytail: one global FIFO, so a workspace that submits 100 builds holds every slot until they drain.
        # Per-workspace fairness would take turns across workspaces in this loop.
        cap = self._cap()
        running = 0
        waiting: list[QueuedBuild] = []
        for build in await self._queue():
            if build.state == "queued":
                waiting.append(build)
            elif await self._still_running(build):
                running += 1
            else:
                await self._end(build)
        for build in waiting:
            if cap is not None and running >= cap:
                return
            try:
                if await self._start(build):
                    running += 1
            except Exception:
                # Jobs or the entity store failing now fails the next build too; try them all next pass.
                logger.exception("could not start build %s; retrying next pass", _label(build))
                return

    async def _queue(self) -> list[QueuedBuild]:
        """Every queued and started build in every workspace, oldest first."""
        assert self._entities is not None
        builds: list[QueuedBuild] = []
        page = 1
        while True:
            result = await self._entities.list(QueuedBuild, workspace="-", sort="created_at", page=page)
            builds.extend(result.data)
            if not result.data or page >= result.pagination.total_pages:
                return builds
            page += 1

    async def _still_running(self, build: QueuedBuild) -> bool:
        assert self._jobs is not None
        try:
            job = (await self._jobs.get_job(workspace=build.workspace, name=build.name)).data()
        except NotFoundError:
            return False
        except Exception:
            # Unknown counts against the cap, so a Jobs outage can't start more builds than it allows.
            logger.exception("could not read job %s; counting it as running", _label(build))
            return True
        return job.status not in _ENDED

    async def _start(self, build: QueuedBuild) -> bool:
        """Create ``build``'s job and move its images to ``pending``. False if the job was refused for good."""
        assert self._jobs is not None and self._entities is not None
        request = CreateHelixJobRequest.model_validate(build.request)
        async with self._jobs_as(_submitter(build)) as jobs:
            try:
                await jobs.create_job(workspace=build.workspace, body=request)
            except ConflictError:
                job = (await self._jobs.get_job(workspace=build.workspace, name=build.name)).data()
                if (job.custom_fields or {}).get("request_digest") != (request.custom_fields or {}).get(
                    "request_digest"
                ):
                    await self._refuse(
                        build, f"not built: job {build.name!r} is held by a job this request did not create"
                    )
                    return False
            except NemoHTTPError as exc:
                if exc.status_code >= 500 or exc.status_code == 429:
                    raise
                await self._refuse(
                    build, f"not built: Jobs refused the build job (HTTP {exc.status_code}): {exc.detail}"
                )
                return False
        await self._release_images(build)
        build.state = "started"
        await self._entities.update(build)
        logger.info("started queued build %s", _label(build))
        return True

    @asynccontextmanager
    async def _jobs_as(self, principal: str | RuntimePrincipal | None) -> AsyncGenerator[AsyncJobsClient]:
        async with self._client_factory(principal) as client:
            yield AsyncJobsClient.from_client(client)

    async def _release_images(self, build: QueuedBuild) -> None:
        """Move ``build``'s ``queued`` images to ``pending``, now that their job exists."""
        assert self._entities is not None
        for row in await self._rows(build):
            if row.status != "queued":
                continue
            row.status = "pending"
            try:
                await self._entities.update(row)
            except EntityConflictError:
                # Changed since it was read: its push step finished it first.
                pass

    async def _refuse(self, build: QueuedBuild, detail: str) -> None:
        logger.warning("dropping queued build %s: %s", _label(build), sanitize_for_log(detail))
        assert self._entities is not None
        await fail_rows(self._entities, await self._rows(build), detail)
        await self._forget(build)

    async def _end(self, build: QueuedBuild) -> None:
        """Forget ``build``, whose job ended; any image it did not publish by then never will be, so fail it."""
        assert self._entities is not None
        await fail_rows(self._entities, await self._rows(build), "not built: its build job ended without publishing it")
        await self._forget(build)

    async def _rows(self, build: QueuedBuild) -> list[ContainerImage]:
        assert self._entities is not None
        rows: list[ContainerImage] = []
        for name in build.images:
            try:
                rows.append(await self._entities.get(ContainerImage, name=name, workspace=build.workspace))
            except EntityNotFoundError:
                continue
        return rows

    async def _forget(self, build: QueuedBuild) -> None:
        assert self._entities is not None
        try:
            await self._entities.delete(QueuedBuild, build.name, workspace=build.workspace)
        except EntityNotFoundError:
            pass


def _label(build: QueuedBuild) -> str:
    return f"{sanitize_for_log(build.workspace)}/{sanitize_for_log(build.name)}"
