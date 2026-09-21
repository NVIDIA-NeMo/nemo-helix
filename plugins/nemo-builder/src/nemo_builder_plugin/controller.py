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
from dataclasses import dataclass
from typing import Protocol

from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, JobOrigin, Signature
from nemo_builder_plugin.registry import (
    ReferenceNotFound,
    RegistryClient,
    RegistryError,
    ResolvedImage,
    signature_tag,
)
from nemo_platform_plugin.controller import NemoController
from nemo_platform_plugin.entities import ListResponse
from nemo_platform_plugin.entity_client import NemoEntitiesClient

logger = logging.getLogger(__name__)

#: Terminal job statuses. A row whose job is still running is simply not ready to look at yet.
_SUCCEEDED = {"completed"}
_FAILED = {"error", "cancelled", "failed"}
_TERMINAL = _SUCCEEDED | _FAILED

#: How many registry passes a row gets before it is failed. The budget exists for one case only:
#: a job that SUCCEEDED against a registry that has not caught up. A row whose job already failed
#: and whose reference does not resolve is failed immediately -- there is nothing to wait for --
#: so without a budget the loop would retry forever against a case that may never resolve.
MAX_ATTEMPTS = 10


class _Registry(Protocol):
    """What the reconciler asks a registry for, and no more.

    Narrower than `RegistryClient` so a test double is a legitimate implementation rather than
    something smuggled past the type checker -- and so it stays obvious that this loop can only
    READ. It cannot push, and the credential behind it does not need to.
    """

    def resolve(self, registry: str, repository: str, reference: str) -> ResolvedImage: ...

    def exists(self, registry: str, repository: str, reference: str) -> bool: ...

    def close(self) -> None: ...


class _EntityStore(Protocol):
    """The two operations this loop performs: find `pending` rows, and write one once."""

    async def list(
        self,
        entity_type: type[ContainerImage],
        *,
        workspace: str,
        filter_obj: dict[str, object] | None = None,
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
    leader election here and no optimistic-concurrency dance. If that ever becomes two replicas,
    this class is wrong rather than merely slower.
    """

    name = "builder"
    dependencies = ["entities", "jobs"]

    def __init__(self) -> None:
        self._entities: _EntityStore | None = None
        self._jobs = None
        self._registry: _Registry | None = None
        self._attempts: dict[str, int] = {}
        self._interval_seconds: float = 10.0

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    async def on_startup(self) -> None:
        from nemo_platform_plugin.client.adapter import client_from_platform
        from nemo_platform_plugin.entities.client import AsyncEntitiesClient
        from nemo_platform_plugin.jobs.client import AsyncJobsClient
        from nemo_platform_plugin.sdk_provider import get_async_platform_sdk

        config = BuilderConfig.get()
        # `internal=True` keeps a loop that polls every few seconds out of the access log.
        sdk = get_async_platform_sdk(as_service="builder", internal=True)
        self._entities = NemoEntitiesClient(client_from_platform(sdk, AsyncEntitiesClient))
        self._jobs = client_from_platform(sdk, AsyncJobsClient)
        self._registry = RegistryClient(
            username=config.registry_username or None,
            password=config.registry_password or None,
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
        """
        response = await self.entities.list(
            ContainerImage, workspace="-", filter_obj={"status": "pending"}, page_size=200
        )
        return list(response.data)

    async def _job_outcome(self, row: ContainerImage, origin: JobOrigin) -> _JobOutcome:
        """What the producing job did, as three explicit facts rather than a message to re-parse.

        A job that cannot be found at all counts as terminal *and* failed. The only way to reach
        that state is a job creation that never landed, and a row whose producer will never run
        must not sit `pending` forever -- exactly one writer owns this row's observed state, and
        it has to answer eventually.
        """
        workspace, _, name = origin.job.partition("/")
        assert self._jobs is not None
        try:
            status = (await self._jobs.get_job_status(workspace=workspace, name=name)).data()
        except Exception:
            logger.warning("job %s not found for image %s; treating as terminal", origin.job, row.name)
            return _JobOutcome(terminal=True, succeeded=False, detail="build job not found")

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
        self._attempts.pop(row.name, None)
        logger.warning("image %s failed: %s", row.name, detail)

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
        attempts = self._attempts.get(row.name, 0) + 1
        self._attempts[row.name] = attempts

        try:
            resolved = self._registry.resolve(row.registry, row.repository, origin.system_tag)
        except ReferenceNotFound:
            # The build failed for THIS image, or the registry has not caught up. The two are
            # distinguishable by whether the job itself failed.
            if not job.succeeded:
                await self._fail(row, f"{job.detail}; no image was pushed for this spec")
            elif attempts >= MAX_ATTEMPTS:
                await self._fail(row, f"{origin.system_tag} never resolved after {attempts} attempts")
            return
        except RegistryError as exc:
            if attempts >= MAX_ATTEMPTS:
                await self._fail(row, f"registry error after {attempts} attempts: {exc}")
            else:
                logger.warning("image %s: registry error (attempt %d): %s", row.name, attempts, exc)
            return

        # Requirement 8 makes signing a MUST on everything this system builds, and a MUST that
        # nothing checks is a comment. Presence only -- see RegistryClient.exists.
        signed = self._registry.exists(row.registry, row.repository, signature_tag(resolved.digest))
        if not signed:
            await self._fail(row, "image was pushed but no signature is present; an unsigned build is not a success")
            return

        row.digest = resolved.digest
        row.manifest_digest = resolved.manifest_digest
        row.tag = origin.system_tag
        row.signature = Signature(storage=config.signature_storage)
        row.status = "ready"
        row.status_detail = None
        await self.entities.update(row)
        self._attempts.pop(row.name, None)

        logger.info(
            "image %s ready: %s/%s@%s%s",
            row.name,
            row.registry,
            row.repository,
            resolved.digest,
            "" if resolved.digest == resolved.manifest_digest else f" (manifest {resolved.manifest_digest})",
        )
