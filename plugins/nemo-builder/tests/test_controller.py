# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build queue controller, against fakes."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Literal

import httpx
import nemo_builder_plugin.controller as controller_module
import pytest
from nemo_builder_plugin.controller import (
    ENDED_UNPUBLISHED,
    NEVER_CREATED,
    ORPHAN_GRACE,
    SWEEP_SECONDS,
    BuildQueueController,
)
from nemo_builder_plugin.entities import ContainerImage, Provenance, QueuedBuild
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError, PermissionDeniedError
from nemo_helix_plugin.entities import EntityNotFoundError
from nemo_helix_plugin.jobs.providers import ContainerSpec, CPUExecutionProvider
from nemo_helix_plugin.jobs.schemas import HelixJobStatus
from nemo_helix_plugin.jobs.spec import HelixJobSpec, HelixJobStepSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

DIGEST = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


def _response(code: int) -> httpx.Response:
    return httpx.Response(code, json={"detail": "nope"}, request=httpx.Request("POST", "http://jobs"))


class FakeEntities:
    def __init__(self) -> None:
        self.entities: dict[str, Any] = {}
        self.fail_update: set[str] = set()
        self.fail_list: set[type] = set()

    async def list(self, entity_type, *, workspace, sort, page, filter_obj=None):
        if entity_type in self.fail_list:
            raise RuntimeError("entity store down")
        found = [e for e in self.entities.values() if isinstance(e, entity_type)]
        for field, value in (filter_obj or {}).items():
            found = [e for e in found if getattr(e, field) == value]
        return SimpleNamespace(data=[e.model_copy(deep=True) for e in found], pagination=SimpleNamespace(total_pages=1))

    async def get(self, entity_type, *, name, workspace):
        if name not in self.entities:
            raise EntityNotFoundError(name)
        return self.entities[name].model_copy(deep=True)

    async def update(self, entity):
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
        #: Called on every read of a job, to change the world while the controller reads it.
        self.on_get: Callable[[], None] = lambda: None

    async def create_job(self, *, workspace: str, body: CreateHelixJobRequest):
        assert body.name is not None
        if body.name in self.refuse:
            raise PermissionDeniedError(_response(403))
        if body.name in self.jobs or body.name in self.conflict:
            raise ConflictError(_response(409))
        self.jobs[body.name] = SimpleNamespace(status=HelixJobStatus.CREATED, custom_fields=body.custom_fields)

    async def get_job(self, *, workspace: str, name: str):
        self.on_get()
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


def _row(
    entities: FakeEntities, job: str, *, status: Literal["queued", "pending"] = "pending", age: timedelta = timedelta(0)
) -> str:
    """A waiting image of ``job`` with no queued build, written ``age`` ago."""
    image = f"{job}.img"
    row = ContainerImage(
        name=image,
        workspace="default",
        status=status,
        registry="reg.example.com",
        repository=f"team/{job}",
        provenance=Provenance(build_set=job, revision=1, spec="img", job=f"default/{job}", request_digest=DIGEST),
    )
    row._created_at = datetime.now(UTC) - age
    entities.entities[image] = row
    return image


def _sweeper(entities: FakeEntities, jobs: FakeJobs) -> BuildQueueController:
    controller = BuildQueueController(cap=lambda: None)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    return controller


@pytest.mark.asyncio
async def test_sweep_fails_images_whose_unqueued_job_ended() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = _sweeper(entities, jobs)
    # Submitted without the cap: a job, but no queued build.
    for job, status in (("errored", HelixJobStatus.ERROR), ("cancelled", HelixJobStatus.CANCELLED)):
        _row(entities, job)
        jobs.jobs[job] = SimpleNamespace(status=status, custom_fields={})
    running = _row(entities, "running")
    jobs.jobs["running"] = SimpleNamespace(status=HelixJobStatus.ACTIVE, custom_fields={})

    await controller.reconcile()
    for job in ("errored", "cancelled"):
        row = entities.entities[f"{job}.img"]
        assert (row.status, row.status_detail) == ("failed", ENDED_UNPUBLISHED)
    assert entities.entities[running].status == "pending"


@pytest.mark.asyncio
async def test_sweep_fails_images_whose_job_was_never_created_once_the_grace_passes() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = _sweeper(entities, jobs)
    # A submit that died, or that Jobs refused, after writing its rows: no job and no queued build.
    old = _row(entities, "orphan", age=ORPHAN_GRACE + timedelta(minutes=1))
    old_queued = _row(entities, "orphan-queued", status="queued", age=ORPHAN_GRACE + timedelta(minutes=1))
    # One still being submitted.
    young = _row(entities, "in-flight", age=timedelta(seconds=5))

    await controller.reconcile()
    for image in (old, old_queued):
        row = entities.entities[image]
        assert (row.status, row.status_detail) == ("failed", NEVER_CREATED)
    assert entities.entities[young].status == "pending"


@pytest.mark.asyncio
async def test_sweep_leaves_the_queues_builds_to_the_queue() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    # Capped at 0, so the queued build stays queued with no job, however old its image is.
    controller = BuildQueueController(cap=lambda: 0)
    vars(controller).update(_entities=entities, _jobs=jobs, _jobs_as=lambda principal: nullcontext(jobs))
    _queue(entities, "waiting")
    entities.entities["waiting.img"]._created_at = datetime.now(UTC) - ORPHAN_GRACE * 3

    await controller.reconcile()
    assert entities.entities["waiting.img"].status == "queued" and "waiting" in entities.entities


@pytest.mark.asyncio
async def test_sweep_leaves_an_image_published_after_it_was_listed() -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = _sweeper(entities, jobs)
    image = _row(entities, "racing")
    jobs.jobs["racing"] = SimpleNamespace(status=HelixJobStatus.COMPLETED, custom_fields={})
    # The push step publishes it after the sweep listed it, while the sweep reads its job.
    jobs.on_get = lambda: setattr(entities.entities[image], "status", "ready")

    await controller.reconcile()
    assert entities.entities[image].status == "ready"


@pytest.mark.asyncio
async def test_sweep_runs_once_a_minute_and_survives_a_failed_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    entities, jobs = FakeEntities(), FakeJobs()
    controller = _sweeper(entities, jobs)
    clock = [1000.0]
    monkeypatch.setattr(controller_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    await controller.reconcile()
    image = _row(entities, "ended")
    jobs.jobs["ended"] = SimpleNamespace(status=HelixJobStatus.ERROR, custom_fields={})
    # Within the minute: the queue runs, the sweep doesn't.
    clock[0] += SWEEP_SECONDS / 2
    await controller.reconcile()
    assert entities.entities[image].status == "pending"

    # A listing that fails is logged; the pass still finishes, and the next sweep catches up.
    clock[0] += SWEEP_SECONDS
    entities.fail_list.add(ContainerImage)
    await controller.reconcile()
    assert entities.entities[image].status == "pending"

    entities.fail_list.clear()
    clock[0] += SWEEP_SECONDS
    await controller.reconcile()
    assert entities.entities[image].status == "failed"
