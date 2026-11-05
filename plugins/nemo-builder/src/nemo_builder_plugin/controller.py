# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``builder-settle``: fail the ``pending`` images no job will complete, in every workspace.

Each pass reads every ``pending`` image, then :func:`~nemo_builder_plugin.settle.settle`\\ s them as the builder: an
image whose job has ended, or that has had no job for :data:`~nemo_builder_plugin.settle.JOB_GRACE`, is written
``failed``. It reads them all before writing any, so a write can't move a row out of a page it hasn't read yet.

Reading one image settles it too; listing doesn't, so a listing's ``status`` is only as fresh as this controller's
last pass. A pass that can't list or settle something leaves it for the next.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import ClassVar

from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.settle import JobStatus, job_status_from, settle
from nemo_helix_plugin.client.client import AsyncNemoClient
from nemo_helix_plugin.client_provider import get_async_nemo_client
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities import EntityClient
from nemo_helix_plugin.entities.client import AsyncEntitiesClient
from nemo_helix_plugin.jobs.client import AsyncJobsClient

logger = logging.getLogger(__name__)

ClientFactory = Callable[[], AbstractAsyncContextManager[AsyncNemoClient]]


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
        images = await self._pending()
        if not images:
            return
        settled = await settle(self._entities, images, self._job_status)
        failed = sum(1 for image in settled if image.status == "failed")
        if failed:
            logger.info("failed %d abandoned image(s) of %d pending", failed, len(images))

    async def _pending(self) -> list[ContainerImage]:
        """Every ``pending`` image in every workspace."""
        assert self._entities is not None
        images: list[ContainerImage] = []
        page = 1
        while True:
            result = await self._entities.list(
                ContainerImage, workspace="-", filter_obj={"status": "pending"}, sort="created_at", page=page
            )
            images.extend(result.data)
            if not result.data or page >= result.pagination.total_pages:
                return images
            page += 1
