# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build queue controller, against fakes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import nemo_builder_plugin.controller as controller_module
import pytest
from nemo_builder_plugin.controller import ORPHAN_CHECK_SECONDS, ORPHAN_GRACE, ORPHANED, BuildQueueController
from nemo_builder_plugin.entities import ContainerImage, Provenance, QueuedBuild
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.entities import EntityConflictError, EntityNotFoundError
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.spec import HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

DIGEST = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


async def _nothing(_entity: Any) -> None:
    return None


def _response(code: int) -> httpx.Response:
    return httpx.Response(code, json={"detail": "nope"}, request=httpx.Request("POST", "http://jobs"))


class FakeEntities:
    def __init__(self) -> None:
        self.entities: dict[str, Any] = {}
        self.fail_update: set[str] = set()
        self.fail_list: set[type] = set()
        #: Awaited at the start of every update, to change the store while the controller writes to it.
        self.on_update: Callable[[Any], Awaitable[None]] = _nothing

    async def create(self, entity):
        if entity.name in self.entities:
            raise EntityConflictError(entity.name)
        self.entities[entity.name] = entity.model_copy(deep=True)
        return entity

    async def list(self, entity_type, *, workspace, sort, page, filter_obj=None):
        if entity_type in self.fail_list:
            raise RuntimeError("entity store down")
        found = [e for e in self.entities.values() if isinstance(e, entity_type)]
        for field, value in (filter_obj or {}).items():
            found = [e for e in found if getattr(e, field) == value]
        # Copies, as the store returns: a change the controller fails to save must not stick.
        return SimpleNamespace(data=[e.model_copy(deep=True) for e in found], pagination=SimpleNamespace(total_pages=1))

    async def get(self, entity_type, *, name, workspace):
        if name not in self.entities:
            raise EntityNotFoundError(name)
        return self.entities[name].model_copy(deep=True)

    async def update(self, entity):
        await self.on_update(entity)
        if entity.name in self.fail_update:
            self.fail_update.discard(entity.name)
            raise RuntimeError("entity store down")
        self.entities[entity.name] = entity
        return entity

    async def delete(self, entity_type, name, *, workspace):
        del self.entities[name]


class FakeJobs:
    def __init__(self) -> None:
        self.jobs: dict[str, SimpleNamespace] = {}
        self.refuse: set[str] = set()
        self.conflict: set[str] = set()

    async def create_job(self, *, workspace: str, body: CreateHelixJobRequest):
        assert body.name is not None
        if body.name in self.refuse:
            raise PermissionDeniedError(_response(403))
        if body.name in self.jobs or body.name in self.conflict:
            raise ConflictError(_response(409))
        self.jobs[body.name] = SimpleNamespace(status=HelixJobStatus.CREATED, custom_fields=body.custom_fields)

    async def get_job(self, *, workspace: str, name: str):
        if name not in self.jobs:
            raise NotFoundError(_response(404))
        return SimpleNamespace(data=lambda: self.jobs[name])


def _queue(entities: FakeEntities, job: str, digest: str = DIGEST) -> None:
    image = f"{job}.img"
    entities.entities[image] = ContainerImage(
        name=image,
        workspace="default",
        status="queued",
        registry="reg.example.com",
        repository=f"team/{job}",
        provenance=Provenance(build_set=job, revision=1, spec="img", job=f"default/{job}", request_digest=digest),
    )
    request = CreateHelixJobRequest(
        name=job,
        source="builder.build",
        spec={},
        platform_spec=HelixJobSpec(
            steps=[HelixJobStepSpec(name="push", executor=CPUExecutionProvider(container=ContainerSpec(image="i")))]
        ),
        custom_fields={"images": [image], "request_digest": digest},
    )
    entities.entities[job] = QueuedBuild(
        name=job, workspace="default", request=request.model_dump(mode="json"), images=[image]
    )


@pytest.mark.asyncio
async def test_queue_starts_builds_in_order_within_the_cap() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = BuildQueueController(cap=lambda: 1)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    for job in ("build-a", "build-b", "build-c"):
        _queue(entities, job)

    def status(job: str) -> str:
        return entities.entities[f"{job}.img"].status

    await controller.reconcile()
    assert set(jobs.jobs) == {"build-a"}
    assert (status("build-a"), status("build-b"), entities.entities["build-a"].state) == (
        "pending",
        "queued",
        "started",
    )

    # Still running: the cap holds.
    await controller.reconcile()
    assert set(jobs.jobs) == {"build-a"}

    # a publishes and ends: its entry goes and b takes the slot. c's job is refused, so c fails rather than waiting.
    entities.entities["build-a.img"].status = "ready"
    jobs.jobs["build-a"].status = HelixJobStatus.COMPLETED
    jobs.refuse.add("build-c")
    await controller.reconcile()
    assert "build-a" not in entities.entities and set(jobs.jobs) == {"build-a", "build-b"}
    assert status("build-a") == "ready"
    # b's job errors before publishing: its image fails instead of staying pending.
    jobs.jobs["build-b"].status = HelixJobStatus.ERROR
    await controller.reconcile()
    assert status("build-b") == "failed" and "build-b" not in entities.entities
    assert status("build-c") == "failed" and "build-c" not in entities.entities

    # A job already created for the same request (a pass that died before marking it started) is adopted.
    _queue(entities, "build-d")
    jobs.jobs["build-d"] = SimpleNamespace(status=HelixJobStatus.ACTIVE, custom_fields={"request_digest": DIGEST})
    await controller.reconcile()
    assert (status("build-d"), entities.entities["build-d"].state) == ("pending", "started")

    # One held by a different request fails the build.
    _queue(entities, "build-e", digest=OTHER)
    jobs.jobs["build-d"].status = HelixJobStatus.COMPLETED
    jobs.jobs["build-e"] = SimpleNamespace(status=HelixJobStatus.ACTIVE, custom_fields={"request_digest": DIGEST})
    await controller.reconcile()
    assert status("build-e") == "failed" and "build-e" not in entities.entities


@pytest.mark.asyncio
async def test_with_no_cap_every_queued_build_starts_on_the_next_pass() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = BuildQueueController(cap=lambda: None)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    for job in ("build-a", "build-b", "build-c"):
        _queue(entities, job)

    await controller.reconcile()
    assert set(jobs.jobs) == {"build-a", "build-b", "build-c"}
    assert {entities.entities[f"{job}.img"].status for job in jobs.jobs} == {"pending"}

    # A job that errors before publishing fails its image, cap or no cap, rather than leaving it pending.
    jobs.jobs["build-b"].status = HelixJobStatus.ERROR
    await controller.reconcile()
    assert entities.entities["build-b.img"].status == "failed" and "build-b" not in entities.entities
    assert entities.entities["build-a.img"].status == "pending"


@pytest.mark.asyncio
async def test_refused_builds_do_not_block_the_queue() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = BuildQueueController(cap=lambda: 1)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    for job in ("bad-request", "missing-secret", "good"):
        _queue(entities, job)
    entities.entities["bad-request"].request = {"name": "bad-request"}
    # Jobs answers 409 with no job behind it, as it does for a secret the job names going missing.
    jobs.conflict.add("missing-secret")
    # The first attempt to fail its image doesn't stick, so its entry must outlive this pass.
    entities.fail_update.add("missing-secret.img")

    await controller.reconcile()
    assert entities.entities["bad-request.img"].status == "failed" and "bad-request" not in entities.entities
    assert entities.entities["missing-secret.img"].status == "queued" and "missing-secret" in entities.entities
    assert set(jobs.jobs) == {"good"} and entities.entities["good.img"].status == "pending"

    # Retried once a slot is free.
    jobs.jobs["good"].status = HelixJobStatus.COMPLETED
    await controller.reconcile()
    assert entities.entities["missing-secret.img"].status == "failed" and "missing-secret" not in entities.entities


def _orphan(entities: FakeEntities, job: str, *, age: timedelta) -> str:
    """A ``queued`` image of ``job``, written ``age`` ago, with no queued build: as a submit that stopped leaves."""
    image = f"{job}.img"
    row = ContainerImage(
        name=image,
        workspace="default",
        status="queued",
        registry="reg.example.com",
        repository=f"team/{job}",
        provenance=Provenance(build_set=job, revision=1, spec="img", job=f"default/{job}", request_digest=DIGEST),
    )
    row._created_at = datetime.now(UTC) - age
    entities.entities[image] = row
    return image


def _held(entities: FakeEntities, job: str) -> BuildQueueController:
    """A controller capped at 0, so ``job``'s queued build stays queued however old its image is."""
    controller = BuildQueueController(cap=lambda: 0)
    vars(controller).update(_entities=entities, _jobs=FakeJobs(), _jobs_as=lambda principal: nullcontext(None))
    _queue(entities, job)
    entities.entities[f"{job}.img"]._created_at = datetime.now(UTC) - ORPHAN_GRACE * 3
    return controller


@pytest.mark.asyncio
async def test_queued_images_no_queued_build_holds_fail_once_old() -> None:
    entities = FakeEntities()
    controller = _held(entities, "waiting")
    old = _orphan(entities, "stopped", age=ORPHAN_GRACE + timedelta(minutes=1))
    young = _orphan(entities, "submitting", age=timedelta(seconds=5))

    await controller.reconcile()
    assert (entities.entities[old].status, entities.entities[old].status_detail) == ("failed", ORPHANED)
    # Its placeholder goes once its images have failed.
    assert "stopped" not in entities.entities
    assert entities.entities[young].status == "queued"
    assert entities.entities["waiting.img"].status == "queued" and "waiting" in entities.entities


@pytest.mark.asyncio
async def test_a_retry_that_queued_the_build_first_keeps_its_images() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = BuildQueueController(cap=lambda: None)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    _queue(entities, "requeued")
    entities.entities["requeued.img"]._created_at = datetime.now(UTC) - ORPHAN_GRACE * 3

    # The pass read no queued build for it, but a retry queued it before the check could claim its name.
    await controller._fail_orphans(set())
    assert entities.entities["requeued.img"].status == "queued"
    assert entities.entities["requeued"].state == "queued"

    # And the retry's build starts.
    await controller.reconcile()
    assert set(jobs.jobs) == {"requeued"} and entities.entities["requeued.img"].status == "pending"


@pytest.mark.asyncio
async def test_the_orphan_check_runs_every_ten_minutes_and_survives_a_failed_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = FakeEntities()
    controller = BuildQueueController(cap=lambda: None)
    vars(controller).update(_entities=entities, _jobs=FakeJobs(), _jobs_as=lambda principal: nullcontext(None))
    clock = [1000.0]
    monkeypatch.setattr(controller_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    await controller.reconcile()
    image = _orphan(entities, "stopped", age=ORPHAN_GRACE * 2)
    clock[0] += ORPHAN_CHECK_SECONDS / 2
    await controller.reconcile()
    assert entities.entities[image].status == "queued"

    # A listing that fails is logged; the pass still finishes, and the next check catches up.
    clock[0] += ORPHAN_CHECK_SECONDS
    entities.fail_list.add(ContainerImage)
    await controller.reconcile()
    assert entities.entities[image].status == "queued"

    entities.fail_list.clear()
    clock[0] += ORPHAN_CHECK_SECONDS
    await controller.reconcile()
    assert entities.entities[image].status == "failed"


@pytest.mark.asyncio
async def test_a_build_none_of_whose_images_still_waits_is_dropped_not_started() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = BuildQueueController(cap=lambda: None)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    # A retry queued it again just after the orphan check failed its image.
    _queue(entities, "retried")
    entities.entities["retried.img"].status = "failed"
    entities.entities["retried.img"].status_detail = ORPHANED

    await controller.reconcile()
    assert jobs.jobs == {} and "retried" not in entities.entities
    assert entities.entities["retried.img"].status == "failed"


@pytest.mark.asyncio
async def test_a_pass_that_stops_after_releasing_the_images_leaves_the_job_to_the_next_pass() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = BuildQueueController(cap=lambda: 1)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    _queue(entities, "interrupted")
    # The pass creates the job and moves the image to pending, then fails to save the build as started.
    entities.fail_update.add("interrupted")
    await controller.reconcile()
    assert set(jobs.jobs) == {"interrupted"}
    assert (entities.entities["interrupted.img"].status, entities.entities["interrupted"].state) == (
        "pending",
        "queued",
    )

    # The next pass adopts the job rather than dropping the build: it holds the slot...
    _queue(entities, "next")
    await controller.reconcile()
    assert entities.entities["interrupted"].state == "started" and set(jobs.jobs) == {"interrupted"}

    # ...and its image fails when the job does, rather than staying pending.
    jobs.jobs["interrupted"].status = HelixJobStatus.ERROR
    await controller.reconcile()
    assert entities.entities["interrupted.img"].status == "failed" and "interrupted" not in entities.entities
    assert set(jobs.jobs) == {"interrupted", "next"}


@pytest.mark.asyncio
async def test_a_retry_after_the_check_claims_the_build_cannot_queue_images_being_failed() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = BuildQueueController(cap=lambda: None)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    image = _orphan(entities, "stopped", age=ORPHAN_GRACE * 2)
    retry: list[str] = []

    async def retry_while_failing(entity: Any) -> None:
        # A retry of the stopped submit queues the build after the check has claimed its name, while it fails the
        # image: the retry's create must conflict, not queue an image that is about to fail.
        if isinstance(entity, ContainerImage) and not retry:
            try:
                await entities.create(QueuedBuild(name="stopped", workspace="default", request={}, images=[image]))
                retry.append("queued")
            except EntityConflictError:
                retry.append("conflict")

    entities.on_update = retry_while_failing
    await controller._fail_orphans(set())
    assert retry == ["conflict"]
    assert (entities.entities[image].status, entities.entities[image].status_detail) == ("failed", ORPHANED)
    assert "stopped" not in entities.entities

    await controller.reconcile()
    assert jobs.jobs == {}


@pytest.mark.asyncio
async def test_a_placeholder_an_unfinished_check_left_is_settled_and_holds_no_slot() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = BuildQueueController(cap=lambda: 1)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    image = _orphan(entities, "stopped", age=ORPHAN_GRACE * 2)
    entities.entities["stopped"] = QueuedBuild(
        name="stopped", workspace="default", request={}, images=[image], state="orphaned"
    )
    _queue(entities, "other")

    await controller.reconcile()
    assert (entities.entities[image].status, entities.entities[image].status_detail) == ("failed", ORPHANED)
    assert "stopped" not in entities.entities
    assert set(jobs.jobs) == {"other"}
