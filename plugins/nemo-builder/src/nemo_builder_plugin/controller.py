# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reconciler -- the **only** writer of a ``ContainerImage``'s observed state.

Nothing downstream is testable without it: rows are created ``pending`` at submit and only this
loop moves them, so a wedged reconciler means no image ever reaches ``ready`` and every build
appears to hang. The failure is total rather than degraded, which is why the attempt budget and
the "is this even mine" checks are here rather than deferred.

**It selects by query, not by event stream.** ``status == "pending"`` is the whole selector,
which keeps it correct under a poll-only loop and means a missed event cannot strand a row.

**It asks the registry even when the job failed.** A failed job is not a per-image verdict: the
build step attempts every spec and does not abort the set, so a job can exit non-zero having
pushed some of its images. Reading the job's status and stopping there would fail rows whose
images are sitting in the registry.

**It writes once.** ``digest``, ``manifest_digest``, ``tag`` and ``signature`` are set in the
transition to ``ready``, and ``ready`` is terminal. A rebuild creates a new row rather than
mutating one a consumer may have pinned.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, JobOrigin, Signature
from nemo_builder_plugin.registry import (
    ReferenceNotFound,
    RegistryClient,
    RegistryError,
    ResolvedImage,
    signature_tag,
)
from nemo_helix_plugin.client.errors import NotFoundError
from nemo_helix_plugin.controller import NemoController
from nemo_helix_plugin.entities import ListResponse
from nemo_helix_plugin.entity_client import NemoEntitiesClient

logger = logging.getLogger(__name__)

#: Terminal job statuses. A row whose job is still running is simply not ready to look at yet.
_SUCCEEDED = {"completed"}
_FAILED = {"error", "cancelled", "failed"}
_TERMINAL = _SUCCEEDED | _FAILED

#: How many registry passes a row gets before it is failed. The budget exists for one case only:
#: a job that SUCCEEDED against a registry that has not caught up -- its image, or its signature,
#: not there yet. A row whose job already failed and whose reference does not resolve is failed
#: immediately -- there is nothing to wait for. A registry that ERRORS spends none of it: an
#: outage says nothing about the image, and failing rows through one would fail them for good.
MAX_ATTEMPTS = 10

#: How long a row's job may be missing before that means it will never exist. Rows are written
#: before the job, so for a moment after every submit the job is legitimately not there yet.
JOB_CREATION_GRACE = timedelta(minutes=5)

#: Rows read per page. Every page is read, oldest first, so no row waits behind newer ones.
_PAGE_SIZE = 200


class _Registry(Protocol):
    """What the reconciler asks a registry for, and no more.

    Narrower than `RegistryClient` so a test double is a legitimate implementation rather than
    something smuggled past the type checker -- and so it stays obvious that this loop can only
    READ. It cannot push, and the credential behind it does not need to.
    """

    def resolve(self, registry: str, repository: str, reference: str) -> ResolvedImage: ...

    def exists(self, registry: str, repository: str, reference: str) -> bool: ...

    def close(self) -> None: ...


class _JobStatusResponse(Protocol):
    def data(self) -> Any: ...


class _Jobs(Protocol):
    """The one Jobs call this loop makes. Narrow for the same reason as `_Registry`."""

    # `def ... -> Awaitable`, not `async def`: the generated client's method returns an Awaitable,
    # and a protocol declaring a coroutine would reject it.
    def get_job_status(self, *, workspace: str, name: str) -> Awaitable[_JobStatusResponse]: ...


class _EntityStore(Protocol):
    """The two operations this loop performs: find `pending` rows, and write one once."""

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


@dataclass(frozen=True, slots=True)
class _JobOutcome:
    """Three facts about the producing job.

    `succeeded` is its own field rather than something recovered from `detail`: branching on a
    substring of a human-readable message is how a reworded log line silently changes behaviour.
    """

    terminal: bool
    succeeded: bool
    detail: str | None


class BuilderController(NemoController):
    """Converges ``pending`` ``ContainerImage`` rows against the registry.

    Single-writer as a *deployment* property -- one replica, ``Recreate`` -- so there is no
    leader election here. A second replica would not corrupt a row: every update carries the
    ``db_version`` the row was read at, so the slower writer gets a conflict rather than
    overwriting a digest. It would only duplicate registry traffic, and the attempt counts,
    which live in memory, would be per replica.
    """

    name = "builder"
    dependencies = ["entities", "jobs"]

    def __init__(self) -> None:
        self._entities: _EntityStore | None = None
        self._jobs: _Jobs | None = None
        self._registry: _Registry | None = None
        #: Keyed by (workspace, name): row names are unique only within a workspace.
        self._attempts: dict[tuple[str, str], int] = {}
        self._interval_seconds: float = 10.0

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
        self._registry = RegistryClient(
            username=config.registry_username or None,
            password=config.registry_password or None,
            insecure=config.registry_insecure,
        )
        self._interval_seconds = float(config.reconcile_interval_seconds)
        logger.info("builder reconciler started; signature storage=%s", config.signature_storage)

    async def on_shutdown(self) -> None:
        if self._registry is not None:
            self._registry.close()

    @property
    def entities(self) -> _EntityStore:
        if self._entities is None:
            raise RuntimeError("controller accessed before on_startup()")
        return self._entities

    async def list_objects(self) -> list:
        """Every `pending` row, across workspaces.

        `-` is the cross-workspace sentinel: this loop is the platform's, not a tenant's.

        **Oldest first, and every page.** The store's default order is newest first, so reading
        one page meant a stream of new submits -- or rows stuck waiting on their jobs -- kept
        older rows, whose jobs had long finished, from ever being looked at.
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

    async def _job_outcome(self, row: ContainerImage, origin: JobOrigin) -> _JobOutcome:
        """What the producing job did, as three explicit facts rather than a message to re-parse.

        **Only a 404 means "not found", and only once the row is old enough.** Rows are written
        before their job, so a cycle that lands between the two finds no job for a row whose job
        is about to exist -- and failing it then is permanent, since `failed` is never
        re-selected, while the job goes on to build and push. Within
        :data:`JOB_CREATION_GRACE` of the row's creation a missing job is "not yet". After it, a
        job that does not exist counts as terminal *and* failed: a creation that never landed,
        whose row must not sit `pending` forever.

        **Every other error means "don't know yet", not "failed".** An earlier version caught
        every exception and called it not-found. A single 503, network blip or token hiccup while
        the build was still running then met an image not yet pushed, and the row was failed --
        terminally, since `failed` is never re-selected. Over hours of polling that is not an edge
        case. Now such a row simply waits for the next cycle; a Jobs service that stays down keeps
        rows `pending`, which is the truthful state and is what health alerting is for.
        """
        workspace, _, name = origin.job.partition("/")
        assert self._jobs is not None
        try:
            status = (await self._jobs.get_job_status(workspace=workspace, name=name)).data()
        except NotFoundError:
            created = row.created_at
            if created is not None and datetime.now(UTC) - created < JOB_CREATION_GRACE:
                logger.info("job %s for image %s does not exist yet", origin.job, row.name)
                return _JobOutcome(terminal=False, succeeded=False, detail=None)
            logger.warning("job %s not found for image %s; treating as terminal", origin.job, row.name)
            return _JobOutcome(terminal=True, succeeded=False, detail="build job not found")
        except Exception:
            logger.warning(
                "could not read status of job %s for image %s; retrying next cycle",
                origin.job,
                row.name,
                exc_info=True,
            )
            return _JobOutcome(terminal=False, succeeded=False, detail=None)

        value = getattr(status.status, "value", str(status.status)).lower()
        if value not in _TERMINAL:
            return _JobOutcome(terminal=False, succeeded=False, detail=None)
        return _JobOutcome(
            terminal=True,
            succeeded=value in _SUCCEEDED,
            detail=f"build job ended as {value}",
        )

    async def _fail(self, row: ContainerImage, detail: str) -> None:
        row.status = "failed"
        row.status_detail = detail
        await self.entities.update(row)
        self._attempts.pop((row.workspace, row.name), None)
        logger.warning("image %s failed: %s", row.name, detail)

    async def _spend_attempt(self, row: ContainerImage, key: tuple[str, str], detail: str) -> None:
        """Count one pass that found the job done and the registry not yet showing its result."""
        attempts = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempts
        if attempts >= MAX_ATTEMPTS:
            await self._fail(row, f"{detail} after {attempts} attempts")
        else:
            logger.info("image %s: %s (attempt %d of %d)", row.name, detail, attempts, MAX_ATTEMPTS)

    async def reconcile_one(self, obj: object) -> None:
        row = obj if isinstance(obj, ContainerImage) else ContainerImage.model_validate(obj)
        config = BuilderConfig.get()
        origin = row.provenance.built_by

        # A registered row was written `ready` by the route that resolved it before the row
        # existed, so there is nothing here to converge. It is never selected in practice; this
        # is the guard that keeps it that way if the query ever widens.
        if not isinstance(origin, JobOrigin):
            return

        job = await self._job_outcome(row, origin)
        if not job.terminal:
            return  # still building; not an error, just early

        assert self._registry is not None
        key = (row.workspace, row.name)

        try:
            resolved = self._registry.resolve(row.registry, row.repository, origin.system_tag)
            # Requirement 8 makes signing a MUST on everything this system builds, and a MUST
            # that nothing checks is a comment. Presence only -- see RegistryClient.exists.
            signed = self._registry.exists(row.registry, row.repository, signature_tag(resolved.digest))
        except ReferenceNotFound:
            # The build failed for THIS image, or the registry has not caught up. The two are
            # distinguishable by whether the job itself failed.
            if not job.succeeded:
                await self._fail(row, f"{job.detail}; no image was pushed for this spec")
            else:
                await self._spend_attempt(row, key, f"{origin.system_tag} never resolved")
            return
        except RegistryError as exc:
            # Says nothing about the image, so it spends nothing: the row waits, `pending` being
            # the truthful state, and a registry that stays unreachable is for health alerting.
            logger.warning("image %s: registry error, retrying next cycle: %s", row.name, exc)
            return

        if not signed:
            if not job.succeeded:
                await self._fail(row, f"{job.detail}; the image was pushed but never signed")
            else:
                await self._spend_attempt(
                    row, key, "image was pushed but no signature is present; an unsigned build is not a success"
                )
            return

        row.digest = resolved.digest
        row.manifest_digest = resolved.manifest_digest
        row.tag = origin.system_tag
        row.signature = Signature(storage=config.signature_storage)
        row.status = "ready"
        row.status_detail = None
        await self.entities.update(row)
        self._attempts.pop(key, None)

        logger.info(
            "image %s ready: %s/%s@%s%s",
            row.name,
            row.registry,
            row.repository,
            resolved.digest,
            "" if resolved.digest == resolved.manifest_digest else f" (manifest {resolved.manifest_digest})",
        )
