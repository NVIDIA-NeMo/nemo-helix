# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build queue controller, against fakes."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from nemo_builder_plugin.controller import BuildQueueController
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

    async def list(self, entity_type, *, workspace, sort, page):
        found = [e for e in self.entities.values() if isinstance(e, entity_type)]
        return SimpleNamespace(data=found, pagination=SimpleNamespace(total_pages=1))

    async def get(self, entity_type, *, name, workspace):
        if name not in self.entities:
            raise EntityNotFoundError(name)
        return self.entities[name]

    async def update(self, entity):
        self.entities[entity.name] = entity
        return entity

    async def delete(self, entity_type, name, *, workspace):
        del self.entities[name]


class FakeJobs:
    def __init__(self) -> None:
        self.jobs: dict[str, SimpleNamespace] = {}
        self.refuse: set[str] = set()

    async def create_job(self, *, workspace: str, body: CreateHelixJobRequest):
        assert body.name is not None
        if body.name in self.refuse:
            raise PermissionDeniedError(_response(403))
        if body.name in self.jobs:
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
