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
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from nemo_builder_plugin.backend import BackendRefused
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.steps import PushStepConfig
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
        #: Rows whose next update is refused, as a row changed since it was read would be.
        self.changed: set[str] = set()

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

    async def update(self, entity: ContainerImage) -> ContainerImage:
        if entity.name in self.changed:
            self.events.append(f"update_conflict:{entity.name}")
            raise EntityConflictError(f"{entity.name} changed")
        self.events.append(f"update:{entity.name}:{entity.status}")
        self.rows[entity.name] = entity
        return entity


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


PUBLIC_KEY = (
    ec.generate_private_key(ec.SECP256R1())
    .public_key()
    .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    .decode()
)


def _config(**overrides: object) -> BuilderConfig:
    settings: dict[str, object] = {
        "registry": "reg.example.com",
        "sandbox_image": "kaniko.example.com/executor:debug",
        "credential_broker": "http://nhx-build-broker.nhx-build-broker.svc:8080",
        "signing_public_key": PUBLIC_KEY,
    }
    return BuilderConfig.model_validate(settings | overrides)


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
    entity_client: FakeEntityClient,
    jobs: FakeJobs | None = None,
    build_set: BuildSet | None = None,
    config: BuilderConfig | None = None,
) -> SubmitResult:
    jobs = jobs or FakeJobs(entity_client.events)
    return await submit_build_set(
        build_set or _set(),
        backend=ExecutionBackend(config or _config()),
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
        assert all(image.provenance.request_digest == request_digest(_set()) for image in result.images)

    @pytest.mark.asyncio
    async def test_every_row_names_its_job_and_system_tag(self) -> None:
        client = FakeEntityClient()
        result = await _submit(client)
        provenance = result.images[0].provenance
        assert (provenance.job, provenance.system_tag) == ("default/demo-1", "default--demo-1-0")


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
    async def test_a_larger_request_at_a_used_revision_writes_nothing(self) -> None:
        """Every submit starts at row 0, which any earlier submit of the revision wrote, so a
        different request is refused before it writes a row of its own."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        await _submit(client, jobs, _set(1))
        client.events.clear()

        with pytest.raises(BuildConflict, match="new revision"):
            await _submit(client, jobs, _set(3))
        assert client.events == ["conflict:demo-1-0", "get:demo-1-0"]

    @pytest.mark.asyncio
    async def test_a_job_name_held_by_another_request_is_refused_and_its_rows_failed(self) -> None:
        """Left pending, the rows would be the other job's: the broker grants a job its rows by name."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        jobs.jobs["demo-1"] = CreateHelixJobRequest.model_construct(name="demo-1", custom_fields={"owner": "someone"})
        with pytest.raises(BuildConflict, match="not created by this request"):
            await _submit(client, jobs)
        assert client.events == [
            "create:demo-1-0",
            "create:demo-1-1",
            "job_conflict",
            "get_job:demo-1",
            "update:demo-1-0:failed",
            "update:demo-1-1:failed",
        ]
        assert all(row.status_detail and "held by a job" in row.status_detail for row in client.rows.values())

    @pytest.mark.asyncio
    async def test_submitting_again_into_the_held_name_writes_nothing_more(self) -> None:
        """The rows are already failed, and failed is final."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        jobs.jobs["demo-1"] = CreateHelixJobRequest.model_construct(name="demo-1", custom_fields={"owner": "someone"})
        with pytest.raises(BuildConflict):
            await _submit(client, jobs)
        client.events.clear()

        with pytest.raises(BuildConflict, match="not created by this request"):
            await _submit(client, jobs)
        assert not [event for event in client.events if event.startswith("update")]

    @pytest.mark.asyncio
    async def test_a_row_that_changed_meanwhile_is_left_as_it_is(self) -> None:
        """Failing is conditional, like every write to a row: a refused write is not retried."""
        client = FakeEntityClient()
        client.changed.add("demo-1-0")
        jobs = FakeJobs(client.events)
        jobs.jobs["demo-1"] = CreateHelixJobRequest.model_construct(name="demo-1", custom_fields={"owner": "someone"})
        with pytest.raises(BuildConflict):
            await _submit(client, jobs)
        assert client.events[-2:] == ["update_conflict:demo-1-0", "update:demo-1-1:failed"]


class TestRefusals:
    @pytest.mark.asyncio
    async def test_a_deployment_that_cannot_build_writes_nothing(self) -> None:
        """Refused by the backend, before any row: a 409, since the request itself is fine."""
        client = FakeEntityClient()
        with pytest.raises(BackendRefused, match="credential_broker"):
            await _submit(client, config=_config(credential_broker=None))
        assert client.events == []

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
    async def test_the_job_names_this_service(self) -> None:
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
        row resolved the same string, so two images in one repository recorded one digest -- and
        the broker, which signs what a row's system tag names, would sign one image for both.
        """
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        result = await _submit(client, jobs, _set(3))
        push = PushStepConfig.model_validate(jobs.jobs["demo-1"].platform_spec.steps[2].config)
        pushed = [image.refs for image in push.images]

        for index, row in enumerate(result.images):
            origin = row.provenance
            owners = [i for i, tags in enumerate(pushed) if any(t.endswith(f":{origin.system_tag}") for t in tags)]
            assert owners == [index], f"row {row.name} tag {origin.system_tag!r} is pushed by images {owners}"
