# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``builder-queue``: start queued build jobs, oldest first, while fewer than ``max_concurrent_builds`` run.

Every submit writes a :class:`QueuedBuild` instead of creating the job. Each pass this controller
forgets the started builds whose jobs ended (failing any image they left unpublished), then creates jobs for the oldest queued ones until the cap is met,
or for all of them with no cap set, each as the user who submitted it, and moves that build's images from ``queued``
to ``pending``.

Every ten minutes it also fails the ``queued`` images no queued build holds, ten minutes after they were written: a
submit stopped between writing them and queuing their build, and nothing else would ever start or fail them. It
first claims the build's name with an ``orphaned`` placeholder, so that it and a retry queuing the same build can't
both act: whichever creates its entry first wins, and the other's create conflicts.

Every step is safe to repeat: a job that already exists for the same request is adopted, so a pass that dies
between creating a job and marking its build started just finishes the work on the next pass.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
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
from pydantic import ValidationError

logger = logging.getLogger(__name__)

_ENDED = set(HelixJobStatus.terminals())
ClientFactory = Callable[[str | RuntimePrincipal | None], AbstractAsyncContextManager[AsyncNemoClient]]

ORPHANED = "not built: its submit stopped before queuing the build"
#: How old a ``queued`` image with no queued build must be before it is failed: far longer than a submit takes between
#: writing its rows and its queued build.
ORPHAN_GRACE = timedelta(minutes=10)
#: Orphans are rare, and finding them lists every queued image, so the check runs this seldom.
ORPHAN_CHECK_SECONDS = 600.0


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
        self._next_orphan_check = 0.0

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
        builds = await self._queue()
        await self._advance(builds)
        if time.monotonic() >= self._next_orphan_check:
            self._next_orphan_check = time.monotonic() + ORPHAN_CHECK_SECONDS
            await self._fail_orphans({(build.workspace, build.name) for build in builds})

    async def _advance(self, builds: list[QueuedBuild]) -> None:
        """Settle the builds whose jobs ended, then start the oldest queued ones while the cap allows."""
        # ponytail: one global FIFO, so a workspace that submits 100 builds holds every slot until they drain.
        # Per-workspace fairness would take turns across workspaces in this loop.
        cap = self._cap()
        running = 0
        waiting: list[QueuedBuild] = []
        for build in builds:
            if build.state == "orphaned":
                # A placeholder the orphan check couldn't finish with: fail its images, then forget it.
                await self._settle(build, ORPHANED)
            elif build.state == "queued":
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
        """Create ``build``'s job and move its images to ``pending``.

        False if the job was refused for good, or if none of its images still waits for it.
        """
        assert self._jobs is not None and self._entities is not None
        try:
            request = CreateHelixJobRequest.model_validate(build.request)
        except ValidationError as exc:
            await self._refuse(build, f"not built: the queued job request is invalid ({exc.error_count()} error(s))")
            return False
        if not any(row.status in ("queued", "pending") for row in await self._rows(build)):
            # Every image settled first, as a retry's do when the orphan check fails them just before the retry
            # queues their build: a job now could publish none of them. A pending image still waits: an earlier
            # pass released it to the job, then stopped before saving the build as started, and the create below
            # adopts that job.
            logger.info("dropping queued build %s: none of its images still waits for its job", _label(build))
            await self._forget(build)
            return False
        async with self._jobs_as(_submitter(build)) as jobs:
            try:
                await jobs.create_job(workspace=build.workspace, body=request)
            except ConflictError as exc:
                try:
                    job = (await self._jobs.get_job(workspace=build.workspace, name=build.name)).data()
                except NotFoundError:
                    # A 409 with no job behind it is a refusal, such as a secret the job names going missing.
                    await self._refuse(build, f"not built: Jobs refused the build job (HTTP 409): {exc.detail}")
                    return False
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
        await self._settle(build, detail)

    async def _end(self, build: QueuedBuild) -> None:
        """Forget ``build``, whose job ended; any image it did not publish by then never will be, so fail it."""
        await self._settle(build, "not built: its build job ended without publishing it")

    async def _fail_orphans(self, queued: set[tuple[str, str]]) -> None:
        """Fail the ``queued`` images older than :data:`ORPHAN_GRACE` that no queued build holds.

        ``queued`` names the builds this pass read. Before failing a build's images, it creates an ``orphaned``
        placeholder under the build's name. If a retry has queued the build since this pass read the queue, that
        create conflicts and the images are kept. If the placeholder comes first, a retry's create conflicts
        instead, so the retry never queues images that are being failed. The placeholder goes once its images have
        failed; one a check couldn't finish is settled by the next pass.
        """
        assert self._entities is not None
        try:
            rows = await self._queued_rows()
        except Exception:
            logger.exception("could not list the queued images; retrying next check")
            return
        now = datetime.now(UTC)
        orphans: dict[tuple[str, str], list[ContainerImage]] = {}
        for row in rows:
            key = (row.workspace, row.provenance.job.partition("/")[2])
            if key not in queued and _age(row, now) >= ORPHAN_GRACE:
                orphans.setdefault(key, []).append(row)
        for (workspace, job), images in orphans.items():
            label = f"{sanitize_for_log(workspace)}/{sanitize_for_log(job)}"
            placeholder = QueuedBuild(
                name=job, workspace=workspace, request={}, images=[image.name for image in images], state="orphaned"
            )
            try:
                await self._entities.create(placeholder)
            except EntityConflictError:
                # A retry has queued the build since this pass read the queue: its images are held again.
                continue
            except Exception:
                logger.exception("could not claim build %s; retrying next check", label)
                continue
            logger.warning("failing %d queued image(s) of %s, which no queued build holds", len(images), label)
            await self._settle(placeholder, ORPHANED)

    async def _queued_rows(self) -> list[ContainerImage]:
        """Every ``queued`` image in every workspace."""
        assert self._entities is not None
        rows: list[ContainerImage] = []
        page = 1
        while True:
            result = await self._entities.list(
                ContainerImage, workspace="-", filter_obj={"status": "queued"}, sort="created_at", page=page
            )
            rows.extend(result.data)
            if not result.data or page >= result.pagination.total_pages:
                return rows
            page += 1

    async def _settle(self, build: QueuedBuild, detail: str) -> None:
        """Fail ``build``'s waiting images, then forget it; kept for the next pass if any image couldn't be failed."""
        assert self._entities is not None
        if await fail_rows(self._entities, await self._rows(build), detail):
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


def _age(row: ContainerImage, now: datetime) -> timedelta:
    """How long ago ``row`` was written; zero if the store didn't say, so an undated row is never taken for old."""
    created = row.created_at
    if created is None:
        return timedelta(0)
    return now - (created if created.tzinfo is not None else created.replace(tzinfo=UTC))
