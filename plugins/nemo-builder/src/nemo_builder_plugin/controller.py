# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``builder-settle``: fail the ``pending`` images no job will complete, in every workspace.

Each pass reads every ``pending`` image, then :func:`~nemo_builder_plugin.settle.settle`\\ s them as the builder: an
image whose job has ended, or that has had no job for :data:`~nemo_builder_plugin.settle.JOB_GRACE`, is written
``failed``. It reads them all before writing any, so a write can't move a row out of a page it hasn't read yet.

Reading one image settles it too; listing doesn't, so a listing's ``status`` is only as fresh as this controller's
last pass. A pass that can't list or settle something leaves it for the next.

Deprecated, for upgrades from 0.10: each pass settles the ``queued`` images 0.10's builder left as it does the
``pending`` ones, then deletes the :class:`~nemo_builder_plugin.entities.QueuedBuild`\\ s it left, each of which holds
its submitter's principal. ``builder-queue``, 0.10's controller, is a deprecated name for this one.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import ClassVar, TypeVar

from nemo_builder_plugin.entities import ContainerImage, QueuedBuild
from nemo_builder_plugin.settle import WAITING, JobStatus, job_status_from, settle
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities import EntityClient, EntityNotFoundError
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.entity import NemoEntity
from nemo_helix_plugin.jobs.client import AsyncJobsClient
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

ClientFactory = Callable[[], AbstractAsyncContextManager[AsyncNemoClient]]
_Entity = TypeVar("_Entity", bound=NemoEntity)


@asynccontextmanager
async def _client() -> AsyncGenerator[AsyncNemoClient]:
    async with get_async_nemo_client(as_service="builder", internal=True) as client:
        yield client


class SettleController(NemoController):
    """Fail the abandoned ``pending`` images (see module docstring)."""

    name: ClassVar[str] = "builder-settle"
    dependencies: ClassVar[list[str]] = ["entities", "jobs"]

    def __init__(self, client_factory: ClientFactory = _client) -> None:
        super().__init__()
        self._client_factory = client_factory
        self._context: AbstractAsyncContextManager[AsyncNemoClient] | None = None
        self._entities: EntityClient | None = None
        self._job_status: JobStatus | None = None

    @property
    def interval_seconds(self) -> float:
        return 30.0

    async def on_startup(self) -> None:
        self._context = self._client_factory()
        client = await self._context.__aenter__()
        self._entities = EntityClient(AsyncEntitiesClient.from_client(client))
        self._job_status = job_status_from(AsyncJobsClient.from_client(client))

    async def on_shutdown(self) -> None:
        if self._context is not None:
            await self._context.__aexit__(None, None, None)

    async def list_objects(self) -> list:
        """Unused: ``reconcile`` settles every image at once, asking about each job once."""
        return []

    async def reconcile_one(self, obj: object) -> None:
        raise NotImplementedError

    async def reconcile(self) -> None:
        assert self._entities is not None and self._job_status is not None
        images = [image for status in WAITING for image in await self._every(ContainerImage, {"status": status})]
        if images:
            settled = await settle(self._entities, images, self._job_status)
            failed = sum(1 for image in settled if image.status == "failed")
            if failed:
                logger.info("failed %d abandoned image(s) of %d waiting", failed, len(images))
        await self._forget_queued_builds()

    async def _forget_queued_builds(self) -> None:
        """Delete the :class:`QueuedBuild`\\ s 0.10's builder left. Deprecated: nothing writes one any more."""
        assert self._entities is not None
        for build in await self._every(QueuedBuild):
            try:
                await self._entities.delete(QueuedBuild, build.name, workspace=build.workspace)
            except EntityNotFoundError:
                pass
            except Exception:
                logger.exception(
                    "could not delete queued build %s/%s; retrying next pass",
                    sanitize_for_log(build.workspace),
                    sanitize_for_log(build.name),
                )

    async def _every(self, entity_type: type[_Entity], filter_obj: dict | None = None) -> list[_Entity]:
        """Every ``entity_type`` in every workspace, every page read before the caller writes any."""
        assert self._entities is not None
        entities: list[_Entity] = []
        page = 1
        while True:
            result = await self._entities.list(
                entity_type, workspace="-", filter_obj=filter_obj, sort="created_at", page=page
            )
            entities.extend(result.data)
            if not result.data or page >= result.pagination.total_pages:
                return entities
            page += 1


class BuildQueueController(SettleController):
    """Deprecated since 0.11: ``builder-queue``, 0.10's name for ``builder-settle``, which it runs.

    So a 0.10 selection naming it keeps working. Select ``builder-settle`` instead. Under a controller group, which
    runs both, each pass settles twice; that's safe, since each write is conditional.
    """

    name: ClassVar[str] = "builder-queue"

    async def on_startup(self) -> None:
        logger.warning(
            "the builder-queue controller is deprecated: it runs builder-settle, which a controller selection should "
            "name instead; Jobs' build profile limits now do what its queue did"
        )
        await super().on_startup()
