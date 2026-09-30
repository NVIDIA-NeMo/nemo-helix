# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The failure sweep -- the one writer of ``pending -> failed`` (Requirement 6).

A row goes ``ready`` only on a verified signature, which the build's last step delivers
to the signature route (``completion.py``). This loop owns the other transition: it fails a row
whose producer will never deliver any. Every interval it pages through ``pending`` rows, oldest
first, and asks Jobs about each row's job:

- **The job has ended** -- completed, errored or cancelled -- and the row is still ``pending``:
  ``failed``, with the job's status as the detail. A job's last step delivers its signatures before it
  exits, so a row still pending when its job has ended has no signature coming.
- **The job does not exist**, and the row is older than the grace period: ``failed``. Rows are
  written before the job, so a submit that died between the two leaves rows whose producer never
  ran.
- **Otherwise**: nothing. A running job's rows wait.

**It reads no registry, keeps no retry budget, and holds no credential.** The registry is read by
whoever signs, at signing time; the sweep only needs to know whether a job has ended.

**Every write is conditional on the version the row was read at**, so the sweep can never
overwrite a ``ready`` that a signature wrote a moment earlier: that write is refused, and the row is
left as the signature left it. And a sweep that wedges only delays *failures* -- no row ever needs
it to become ``ready``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import Protocol

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, Provenance
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.client.response import NemoResponse
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities import EntityConflictError, ListResponse
from nemo_helix_plugin.entity_client import NemoEntitiesClient
from nemo_helix_plugin.jobs.schemas import HelixJobStatus, HelixJobStatusResponse
from nemo_helix_plugin.log_utils import sanitize_for_log

logger = logging.getLogger(__name__)

#: Rows read per page. Every page is read, oldest first, so no row waits behind newer ones.
_PAGE_SIZE = 200

_ENDED = frozenset(HelixJobStatus.terminals())


class _Jobs(Protocol):
    """The one Jobs call this loop makes.

    ``def ... -> Awaitable``, not ``async def``: the generated client's method returns an Awaitable,
    and a protocol declaring a coroutine would reject it.
    """

    def get_job_status(
        self, *, workspace: str | None = None, name: str
    ) -> Awaitable[NemoResponse[HelixJobStatusResponse]]: ...


class _EntityStore(Protocol):
    """The two operations this loop performs: find ``pending`` rows, and fail one conditionally."""

    async def list(
        self,
        entity_type: type[ContainerImage],
        *,
        workspace: str,
        filter_obj: dict[str, object] | None = None,
        sort: str | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> ListResponse[ContainerImage]: ...

    async def update(self, entity: ContainerImage) -> ContainerImage: ...


class BuilderController(NemoController):
    """Fails ``pending`` rows whose job has ended, or never appeared.

    One replica, ``Recreate``, as every controller here -- but a second would be harmless, not
    merely unlikely to collide: every write carries the version it read, and there is no state
    in memory to disagree about.
    """

    name = "builder"
    dependencies = ["entities", "jobs"]

    def __init__(self) -> None:
        self._entities: _EntityStore | None = None
        self._jobs: _Jobs | None = None
        self._interval_seconds: float = 30.0
        self._grace = timedelta(minutes=5)

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    async def on_startup(self) -> None:
        from nemo_helix_plugin.client.adapter import client_from_platform
        from nemo_helix_plugin.entities.client import AsyncEntitiesClient
        from nemo_helix_plugin.jobs.client import AsyncJobsClient
        from nemo_helix_plugin.sdk_provider import get_async_platform_sdk

        config = BuilderConfig.get()
        # `internal=True` keeps a loop that polls every few seconds out of the access log.
        sdk = get_async_platform_sdk(as_service="builder", internal=True)
        self._entities = NemoEntitiesClient(client_from_platform(sdk, AsyncEntitiesClient))
        self._jobs = client_from_platform(sdk, AsyncJobsClient)
        self._interval_seconds = config.sweep_interval_seconds
        self._grace = timedelta(seconds=config.job_creation_grace_seconds)
        logger.info("builder failure sweep started; every %ss", self._interval_seconds)

    @property
    def entities(self) -> _EntityStore:
        if self._entities is None:
            raise RuntimeError("controller accessed before on_startup()")
        return self._entities

    @property
    def jobs(self) -> _Jobs:
        if self._jobs is None:
            raise RuntimeError("controller accessed before on_startup()")
        return self._jobs

    async def list_objects(self) -> list:
        """Every ``pending`` row, across workspaces.

        ``-`` is the cross-workspace sentinel: this loop is the platform's, not a tenant's.

        **Oldest first, and every page.** The store's default order is newest first, so reading
        one page would let a stream of new submits -- or rows waiting on long builds -- keep
        older rows, whose jobs had long ended, from ever being looked at.
        """
        rows: list[ContainerImage] = []
        page = 1
        while True:
            response = await self.entities.list(
                ContainerImage,
                workspace="-",
                filter_obj={"status": "pending"},
                sort="created_at",
                page=page,
                page_size=_PAGE_SIZE,
            )
            rows.extend(response.data)
            if page >= response.pagination.total_pages or not response.data:
                return rows
            page += 1

    async def _why_failed(self, row: ContainerImage, origin: Provenance) -> str | None:
        """Why ``row`` will never complete, or None while it still might.

        **Only a 404 means "not found", and only once the row is old enough.** Rows are written
        before their job, so a cycle between the two finds no job for a row whose job is about to
        exist -- and failing it then would be permanent, since ``failed`` is terminal, while the job
        goes on to build. Within the grace period a missing job is "not yet".

        **Every other error means "don't know yet".** A 503, a network blip or a token hiccup says
        nothing about the build; the row waits for the next cycle. A Jobs service that stays down
        keeps rows ``pending``, which is the truthful state and is what health alerting is for.
        """
        workspace, _, name = origin.job.partition("/")
        try:
            status = (await self.jobs.get_job_status(workspace=workspace, name=name)).data()
        except NotFoundError:
            created = row.created_at
            if created is not None and created.tzinfo is None:
                # The entity store keeps UTC without saying so.
                created = created.replace(tzinfo=UTC)
            if created is not None and datetime.now(UTC) - created < self._grace:
                return None
            return "its build job was never created, or no longer exists"
        except Exception:
            logger.warning(
                "could not read job %s for image %s; retrying next cycle",
                sanitize_for_log(origin.job),
                sanitize_for_log(row.name),
                exc_info=True,
            )
            return None
        if status.status not in _ENDED:
            return None
        return f"its build job ended as {status.status.value} without delivering a signature for it"

    async def reconcile_one(self, obj: object) -> None:
        row = obj if isinstance(obj, ContainerImage) else ContainerImage.model_validate(obj)
        if row.status != "pending":
            return

        detail = await self._why_failed(row, row.provenance)
        if detail is None:
            return
        row.status = "failed"
        row.status_detail = detail
        try:
            await self.entities.update(row)
        except EntityConflictError:
            # The row changed since it was read: its signature arrived. Whatever it is now stands.
            logger.info(
                "image %s/%s changed while being failed; leaving it as it is",
                sanitize_for_log(row.workspace),
                sanitize_for_log(row.name),
            )
            return
        logger.warning(
            "image %s/%s failed: %s",
            sanitize_for_log(row.workspace),
            sanitize_for_log(row.name),
            sanitize_for_log(detail),
        )
