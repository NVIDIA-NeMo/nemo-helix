# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submit ordering, against fakes.

What is being tested is a *sequence*, not a return value: rows exist before the job does, and a
losing racer adopts rather than duplicates -- when, and only when, it is the same request. All
of that is invisible in a response body, and it is the reason the design has no
orphan-detection sweep.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, JobOrigin
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.submit import (
    JOB_SOURCE,
    BuildConflict,
    InvalidBuildRequest,
    SubmitResult,
    request_digest,
    submit_build_set,
)
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities import EntityConflictError
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest


class FakeEntityClient:
    """Records the order of operations, and enforces the unique-name constraint."""

    def __init__(self) -> None:
        self.rows: dict[str, ContainerImage] = {}
        self.events: list[str] = []

    async def create(self, entity: ContainerImage) -> ContainerImage:
        if entity.name in self.rows:
            self.events.append(f"conflict:{entity.name}")
            raise EntityConflictError(f"{entity.name} exists")
        self.events.append(f"create:{entity.name}")
        self.rows[entity.name] = entity
        return entity

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage:
        self.events.append(f"get:{name}")
        return self.rows[name]


class FakeJobs:
    """Enforces the unique job name, as Jobs does, and remembers what each job recorded."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.jobs: dict[str, CreateHelixJobRequest] = {}

    async def create_job(self, request: CreateHelixJobRequest) -> CreateHelixJobRequest:
        assert request.name is not None
        if request.name in self.jobs:
            self.events.append("job_conflict")
            raise ConflictError(
                httpx.Response(409, json={"detail": "exists"}, request=httpx.Request("POST", "http://j"))
            )
        self.events.append("create_job")
        self.jobs[request.name] = request
        return request

    async def get_job_fields(self, name: str) -> Mapping[str, Any]:
        self.events.append(f"get_job:{name}")
        return self.jobs[name].custom_fields or {}


def _config() -> BuilderConfig:
    return BuilderConfig(
        registry="reg.example.com",
        push_credential_secret="registry-push-credential",
        signing_key="k8s://nhx-builds/cosign-key",
    )


def _set(n: int = 2, *, tag: str = "v1") -> BuildSet:
    return BuildSet(
        name="demo",
        revision=1,
        build_specs=[
            BuildSpec(
                name=f"img{i}",
                source=FileSetSource(fileset="fs-a"),
                output=BuildOutput(repository=f"team/img{i}", tag=tag),
            )
            for i in range(n)
        ],
    )


async def _submit(
    entity_client: FakeEntityClient, jobs: FakeJobs | None = None, build_set: BuildSet | None = None
) -> SubmitResult:
    jobs = jobs or FakeJobs(entity_client.events)
    return await submit_build_set(
        build_set or _set(),
        config=_config(),
        workspace="default",
        entity_client=entity_client,
        create_job=jobs.create_job,
        get_job_fields=jobs.get_job_fields,
    )


class TestOrdering:
    @pytest.mark.asyncio
    async def test_rows_are_created_before_the_job(self) -> None:
        """The inversion that removes the orphan problem.

        If the job were created first, a pod dying between push and row-write would leave a
        pushed image with nothing pointing at it.
        """
        client = FakeEntityClient()
        await _submit(client)
        assert client.events == ["create:demo-1-0", "create:demo-1-1", "create_job"]

    @pytest.mark.asyncio
    async def test_rows_are_created_pending_with_nothing_observed(self) -> None:
        client = FakeEntityClient()
        result = await _submit(client)
        assert [i.status for i in result.images] == ["pending", "pending"]
        assert all(i.digest is None and i.signature is None for i in result.images)

    @pytest.mark.asyncio
    async def test_every_row_records_the_request_that_wrote_it(self) -> None:
        client = FakeEntityClient()
        result = await _submit(client)
        origins = [image.provenance.built_by for image in result.images]
        assert all(isinstance(o, JobOrigin) and o.request_digest == request_digest(_set()) for o in origins)


class TestResubmission:
    """Settled by the unique-name index, not by a read-then-write -- and by the request digest,
    because a name collision alone cannot say whether it is the same request."""

    @pytest.mark.asyncio
    async def test_the_same_request_again_adopts_the_rows_and_the_job(self) -> None:
        """A client retry after a lost response, a double submit, or a concurrent twin."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        first = await _submit(client, jobs)
        client.events.clear()

        second = await _submit(client, jobs)
        assert client.events == [
            "conflict:demo-1-0",
            "get:demo-1-0",
            "conflict:demo-1-1",
            "get:demo-1-1",
            "job_conflict",
            "get_job:demo-1",
        ]
        assert [i.name for i in second.images] == [i.name for i in first.images]
        assert len(client.rows) == 2 and len(jobs.jobs) == 1

    @pytest.mark.asyncio
    async def test_a_different_request_at_the_same_revision_is_refused_and_adopts_nothing(self) -> None:
        """Adopted, the rows would describe a request the job never built."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        await _submit(client, jobs)
        client.events.clear()

        with pytest.raises(BuildConflict, match="new revision"):
            await _submit(client, jobs, _set(tag="v2"))
        assert client.events == ["conflict:demo-1-0", "get:demo-1-0"]

    @pytest.mark.asyncio
    async def test_a_job_name_held_by_another_request_is_refused(self) -> None:
        """Rows reconciled against a job this request did not create would describe its output."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        jobs.jobs["demo-1"] = CreateHelixJobRequest.model_construct(name="demo-1", custom_fields={"owner": "someone"})
        with pytest.raises(BuildConflict, match="not created by this request"):
            await _submit(client, jobs)


class TestRefusals:
    @pytest.mark.asyncio
    async def test_a_request_the_plan_refuses_writes_nothing(self) -> None:
        """Two specs publishing one caller tag: the caller's error, raised before any row."""
        client = FakeEntityClient()
        build_set = BuildSet(
            name="demo",
            revision=1,
            build_specs=[
                BuildSpec(
                    name=f"img{i}",
                    source=FileSetSource(fileset="fs-a"),
                    output=BuildOutput(repository="team/same", tag="v1"),
                )
                for i in range(2)
            ],
        )
        with pytest.raises(InvalidBuildRequest, match="both publish"):
            await _submit(client, build_set=build_set)
        assert client.events == []


class TestTheJobRequest:
    @pytest.mark.asyncio
    async def test_the_job_names_this_service_so_the_reconciler_finds_its_own(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        await _submit(client, jobs)
        request = jobs.jobs["demo-1"]
        assert request.source == JOB_SOURCE
        assert request.custom_fields == {"images": ["demo-1-0", "demo-1-1"], "request_digest": request_digest(_set())}

    @pytest.mark.asyncio
    async def test_each_row_resolves_the_tag_its_own_image_pushes_and_no_other(self) -> None:
        """Composed once, handed to both, and distinct per image.

        Row i's system tag must appear among image i's push targets, and among NO other image's.
        The second half is the collision the adversarial review found: with a per-set tag every
        row resolved the same string, so two images in one repository recorded one digest.
        """
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        result = await _submit(client, jobs, _set(3))
        pushed = [image["tags"] for image in jobs.jobs["demo-1"].platform_spec.steps[2].config["images"]]

        for index, row in enumerate(result.images):
            origin = row.provenance.built_by
            assert isinstance(origin, JobOrigin)
            owners = [i for i, tags in enumerate(pushed) if any(t.endswith(f":{origin.system_tag}") for t in tags)]
            assert owners == [index], f"row {row.name} tag {origin.system_tag!r} is pushed by images {owners}"
