# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reconciler's decisions, against a fake registry.

The interesting behaviour is all in *when it looks* and *what it concludes* -- and every branch
below is one a real build reaches. The registry protocol itself is exercised separately.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from nemo_builder_plugin.controller import MAX_ATTEMPTS, BuilderController, _JobOutcome
from nemo_builder_plugin.entities import ContainerImage, JobOrigin, Provenance, RegisteredOrigin
from nemo_builder_plugin.registry import ReferenceNotFound, RegistryError, ResolvedImage, signature_tag
from nemo_platform_plugin.client.errors import InternalServerError, NotFoundError
from nemo_platform_plugin.entities import ListResponse, PaginationInfo

DIGEST = "sha256:" + "c" * 64


class FakeRegistry:
    """Resolves what it was told to, and records what it was asked."""

    def __init__(self, *, resolves: ResolvedImage | None = None, signed: bool = True, error: Exception | None = None):
        self._resolves = resolves
        self._signed = signed
        self._error = error
        self.asked: list[str] = []

    def resolve(self, registry: str, repository: str, reference: str) -> ResolvedImage:
        self.asked.append(reference)
        if self._error:
            raise self._error
        if reference.endswith(".sig"):
            if not self._signed:
                raise ReferenceNotFound(reference)
            return ResolvedImage(digest=DIGEST, manifest_digest=DIGEST, media_type="x")
        if self._resolves is None:
            raise ReferenceNotFound(reference)
        return self._resolves

    def exists(self, registry: str, repository: str, reference: str) -> bool:
        try:
            self.resolve(registry, repository, reference)
        except ReferenceNotFound:
            return False
        return True

    def close(self) -> None:
        pass


class FakeEntities:
    def __init__(self) -> None:
        self.updated: list[ContainerImage] = []

    async def list(
        self,
        entity_type: type[ContainerImage],
        *,
        workspace: str,
        filter_obj: dict[str, object] | None = None,
        page_size: int = 100,
    ) -> ListResponse[ContainerImage]:
        return ListResponse(
            data=[],
            pagination=PaginationInfo(page=1, page_size=page_size, current_page_size=0, total_pages=0, total_results=0),
        )

    async def update(self, entity: ContainerImage) -> ContainerImage:
        self.updated.append(entity)
        return entity


def _row(name: str = "demo-1-0") -> ContainerImage:
    return ContainerImage(
        name=name,
        workspace="default",
        registry="reg.example.com",
        repository="team/main",
        provenance=Provenance(
            backend="execution",
            built_by=JobOrigin(build_set="demo", revision=1, job="default/demo-1", system_tag="default--demo-1"),
        ),
    )


class StubbedController(BuilderController):
    """Overrides the one collaborator that needs a live Jobs service.

    A subclass rather than monkeypatching: the override is then type-checked against the real
    signature, so a change to `_job_outcome` breaks these tests at check time instead of
    silently leaving them exercising a method that no longer exists.
    """

    def __init__(self, registry: FakeRegistry, entities: FakeEntities, job_status: str) -> None:
        super().__init__()
        self._registry = registry
        self._entities = entities
        self._job_status = job_status

    async def _job_outcome(self, row: ContainerImage, origin: JobOrigin) -> _JobOutcome:
        if self._job_status == "running":
            return _JobOutcome(terminal=False, succeeded=False, detail=None)
        return _JobOutcome(
            terminal=True,
            succeeded=self._job_status == "completed",
            detail=f"build job ended as {self._job_status}",
        )


def _controller(registry: FakeRegistry, *, job_status: str = "completed") -> tuple[BuilderController, FakeEntities]:
    entities = FakeEntities()
    return StubbedController(registry, entities, job_status), entities


class TestWhenItLooks:
    @pytest.mark.asyncio
    async def test_it_waits_while_the_job_is_running(self) -> None:
        registry = FakeRegistry()
        controller, entities = _controller(registry, job_status="running")
        await controller.reconcile_one(_row())
        assert registry.asked == []
        assert entities.updated == []

    @pytest.mark.asyncio
    async def test_a_registered_row_is_never_reconciled(self) -> None:
        """It was written `ready` by the route that resolved it before the row existed."""
        registry = FakeRegistry()
        controller, entities = _controller(registry)
        row = _row()
        row.provenance = Provenance(backend="registered", built_by=RegisteredOrigin(image_ref="reg.example.com/x:v1"))
        await controller.reconcile_one(row)
        assert registry.asked == []
        assert entities.updated == []

    @pytest.mark.asyncio
    async def test_it_asks_the_registry_even_when_the_job_failed(self) -> None:
        """A failed job is not a per-image verdict: the build attempts every spec and does not
        abort the set, so a job can exit non-zero having pushed some of its images."""
        registry = FakeRegistry(resolves=ResolvedImage(digest=DIGEST, manifest_digest=DIGEST, media_type="m"))
        controller, entities = _controller(registry, job_status="error")
        await controller.reconcile_one(_row())
        assert registry.asked[0] == "default--demo-1"
        assert entities.updated[0].status == "ready"


class TestWhatItWrites:
    @pytest.mark.asyncio
    async def test_a_resolved_signed_image_becomes_ready(self) -> None:
        manifest = "sha256:" + "d" * 64
        registry = FakeRegistry(resolves=ResolvedImage(digest=DIGEST, manifest_digest=manifest, media_type="i"))
        controller, entities = _controller(registry)
        await controller.reconcile_one(_row())

        row = entities.updated[0]
        assert row.status == "ready"
        assert row.digest == DIGEST
        assert row.manifest_digest == manifest
        assert row.tag == "default--demo-1"
        assert row.image_ref == f"reg.example.com/team/main@{DIGEST}"

    @pytest.mark.asyncio
    async def test_the_signature_records_presence_and_claims_nothing_more(self) -> None:
        registry = FakeRegistry(resolves=ResolvedImage(digest=DIGEST, manifest_digest=DIGEST, media_type="m"))
        controller, entities = _controller(registry)
        await controller.reconcile_one(_row())

        signature = entities.updated[0].signature
        assert signature is not None
        assert signature.storage == "tag"
        # Presence, not validity. Anything able to write to the repository can place a
        # well-formed signature made with any key.
        assert signature.verified_against is None

    @pytest.mark.asyncio
    async def test_it_looks_for_the_signature_at_the_digest_not_the_tag(self) -> None:
        registry = FakeRegistry(resolves=ResolvedImage(digest=DIGEST, manifest_digest=DIGEST, media_type="m"))
        controller, _ = _controller(registry)
        await controller.reconcile_one(_row())
        assert registry.asked[1] == signature_tag(DIGEST)


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_pushed_but_unsigned_image_fails(self) -> None:
        """Requirement 8 makes signing a MUST, and an unsigned build is not a degraded success."""
        registry = FakeRegistry(
            resolves=ResolvedImage(digest=DIGEST, manifest_digest=DIGEST, media_type="m"), signed=False
        )
        controller, entities = _controller(registry)
        await controller.reconcile_one(_row())

        row = entities.updated[0]
        assert row.status == "failed"
        assert "signature" in (row.status_detail or "")
        assert row.digest is None, "a failed row must not carry an observed digest"

    @pytest.mark.asyncio
    async def test_a_failed_job_whose_image_is_absent_fails_at_once(self) -> None:
        """There is nothing to wait for, so it does not burn the attempt budget."""
        controller, entities = _controller(FakeRegistry(resolves=None), job_status="error")
        await controller.reconcile_one(_row())
        assert entities.updated[0].status == "failed"

    @pytest.mark.asyncio
    async def test_a_succeeded_job_whose_image_is_absent_retries_then_gives_up(self) -> None:
        """The budget exists for exactly this case: a registry that has not caught up."""
        controller, entities = _controller(FakeRegistry(resolves=None), job_status="completed")
        row = _row()

        for _ in range(MAX_ATTEMPTS - 1):
            await controller.reconcile_one(row)
        assert entities.updated == [], "gave up too early"

        await controller.reconcile_one(row)
        assert entities.updated[0].status == "failed"
        assert "never resolved" in (entities.updated[0].status_detail or "")

    @pytest.mark.asyncio
    async def test_a_registry_error_retries_rather_than_failing_the_row(self) -> None:
        """A registry being briefly unreachable is not a verdict about the image."""
        controller, entities = _controller(FakeRegistry(error=RegistryError("503")), job_status="completed")
        await controller.reconcile_one(_row())
        assert entities.updated == []


class FakeJobs:
    """A Jobs client whose status read fails in a chosen way."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def get_job_status(self, *, workspace: str, name: str) -> Any:
        raise self._error


def _real_controller(registry: FakeRegistry, jobs: FakeJobs) -> tuple[BuilderController, FakeEntities]:
    """The REAL `_job_outcome`, unlike `StubbedController`, which replaces it."""
    controller = BuilderController()
    entities = FakeEntities()
    controller._registry = registry
    controller._entities = entities
    controller._jobs = jobs
    return controller, entities


def _http_error(cls: type[Exception], status: int) -> Exception:
    return cls(httpx.Response(status, request=httpx.Request("GET", "http://jobs.invalid/status")))


class TestJobStatusErrorsAreClassifiedHonestly:
    """Found by adversarial review: every exception used to mean "job not found".

    A single transient error while the build was still running then met an image not yet pushed,
    and a healthy build's row was failed -- terminally.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("connection reset"),
            _http_error(InternalServerError, 503),
        ],
        ids=["network", "http-503"],
    )
    async def test_a_transient_error_leaves_an_in_flight_build_pending(self, error: Exception) -> None:
        controller, entities = _real_controller(FakeRegistry(resolves=None), FakeJobs(error))
        row = _row()
        await controller.reconcile_one(row)
        assert row.status == "pending"
        assert entities.updated == [], "a transient error must not write a verdict"

    @pytest.mark.asyncio
    async def test_a_genuinely_missing_job_still_fails_the_row(self) -> None:
        """A job creation that never landed must not leave a row pending forever."""
        controller, entities = _real_controller(FakeRegistry(resolves=None), FakeJobs(_http_error(NotFoundError, 404)))
        await controller.reconcile_one(_row())
        assert entities.updated[0].status == "failed"
        assert "not found" in (entities.updated[0].status_detail or "")
