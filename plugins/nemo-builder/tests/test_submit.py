# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submit ordering, against fakes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.identity import compose_system_tag
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.steps import PushStepConfig
from nemo_builder_plugin.submit import (
    JOB_SOURCE,
    BuildConflict,
    InvalidBuildRequest,
    MissingSecrets,
    SubmitResult,
    request_digest,
    submit_build_set,
)
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities import EntityConflictError
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfile
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest

#: The build's Jobs execution profiles, as the quickstart has them.
PROFILES = [
    KubernetesJobExecutionProfile.model_validate(
        {"profile": name, "config": {"namespace": "nhx-builds", "storage": {"pvc_name": "nhx-build-work"}}}
    )
    for name in ("build-fetch", "build-control", "build-push")
]


class FakeEntityClient:
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


def _config(**overrides: object) -> BuilderConfig:
    settings: dict[str, object] = {
        "registry": "reg.example.com",
        "sandbox": {"image": "kaniko.example.com/executor:debug"},
    }
    return BuilderConfig.model_validate(settings | overrides)


def _set(n: int = 2, *, tag: str = "v1", names: list[str] | None = None) -> BuildSet:
    return BuildSet(
        name="demo",
        revision=1,
        build_specs=[
            BuildSpec(
                name=name,
                source=FileSetSource(fileset="fs-a"),
                output=BuildOutput(repository=f"team/{name}", tag=tag),
            )
            for name in names or [f"img{i}" for i in range(n)]
        ],
    )


async def _submit(
    entity_client: FakeEntityClient,
    jobs: FakeJobs | None = None,
    build_set: BuildSet | None = None,
    config: BuilderConfig | None = None,
    unreadable: frozenset[str] = frozenset(),
) -> SubmitResult:
    jobs = jobs or FakeJobs(entity_client.events)

    async def secret_readable(workspace: str, name: str) -> bool:
        return f"{workspace}/{name}" not in unreadable

    return await submit_build_set(
        build_set or _set(),
        backend=ExecutionBackend(config or _config(), PROFILES),
        workspace="default",
        entity_client=entity_client,
        create_job=jobs.create_job,
        get_job_fields=jobs.get_job_fields,
        secret_readable=secret_readable,
    )


class TestOrdering:
    @pytest.mark.asyncio
    async def test_rows_are_created_before_the_job(self) -> None:
        client = FakeEntityClient()
        await _submit(client)
        assert client.events == ["create:demo-1.img0", "create:demo-1.img1", "create_job"]

    @pytest.mark.asyncio
    async def test_rows_are_created_pending_with_nothing_observed(self) -> None:
        client = FakeEntityClient()
        result = await _submit(client)
        assert [i.status for i in result.images] == ["pending", "pending"]
        assert all(i.digest is None for i in result.images)

    @pytest.mark.asyncio
    async def test_every_row_records_the_request_that_wrote_it(self) -> None:
        client = FakeEntityClient()
        result = await _submit(client)
        assert all(image.provenance.request_digest == request_digest(_set()) for image in result.images)

    @pytest.mark.asyncio
    async def test_every_row_names_its_job_and_spec(self) -> None:
        client = FakeEntityClient()
        result = await _submit(client)
        provenance = result.images[0].provenance
        assert (provenance.job, provenance.spec) == ("default/demo-1", "img0")


class TestResubmission:
    """Settled by the unique-name index, not a read-then-write, and by the request digest."""

    @pytest.mark.asyncio
    async def test_the_same_request_again_adopts_the_rows_and_the_job(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        first = await _submit(client, jobs)
        client.events.clear()

        second = await _submit(client, jobs)
        assert client.events == [
            "conflict:demo-1.img0",
            "get:demo-1.img0",
            "conflict:demo-1.img1",
            "get:demo-1.img1",
            "job_conflict",
            "get_job:demo-1",
        ]
        assert [i.name for i in second.images] == [i.name for i in first.images]
        assert len(client.rows) == 2 and len(jobs.jobs) == 1

    @pytest.mark.asyncio
    async def test_a_different_request_at_the_same_revision_is_refused_and_adopts_nothing(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        await _submit(client, jobs)
        client.events.clear()

        with pytest.raises(BuildConflict, match="new revision"):
            await _submit(client, jobs, _set(tag="v2"))
        assert client.events == ["conflict:demo-1.img0", "get:demo-1.img0"]

    @pytest.mark.asyncio
    async def test_rows_written_before_a_collision_are_failed(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        await _submit(client, jobs, _set(1))
        client.events.clear()

        with pytest.raises(BuildConflict, match="new revision"):
            await _submit(client, jobs, _set(names=["extra", "img0"]))
        assert client.events == [
            "create:demo-1.extra",
            "conflict:demo-1.img0",
            "get:demo-1.img0",
            "update:demo-1.extra:failed",
        ]
        assert client.rows["demo-1.img0"].status == "pending"

    @pytest.mark.asyncio
    async def test_a_job_name_held_by_another_request_is_refused_and_its_rows_failed(self) -> None:
        """Left pending, they would wait on a job this request didn't create."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        jobs.jobs["demo-1"] = CreateHelixJobRequest.model_construct(name="demo-1", custom_fields={"owner": "someone"})
        with pytest.raises(BuildConflict, match="not created by this request"):
            await _submit(client, jobs)
        assert client.events == [
            "create:demo-1.img0",
            "create:demo-1.img1",
            "job_conflict",
            "get_job:demo-1",
            "update:demo-1.img0:failed",
            "update:demo-1.img1:failed",
        ]
        assert all(row.status_detail and "held by a job" in row.status_detail for row in client.rows.values())

    @pytest.mark.asyncio
    async def test_submitting_again_into_the_held_name_writes_nothing_more(self) -> None:
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
        client = FakeEntityClient()
        client.changed.add("demo-1.img0")
        jobs = FakeJobs(client.events)
        jobs.jobs["demo-1"] = CreateHelixJobRequest.model_construct(name="demo-1", custom_fields={"owner": "someone"})
        with pytest.raises(BuildConflict):
            await _submit(client, jobs)
        assert client.events[-2:] == ["update_conflict:demo-1.img0", "update:demo-1.img1:failed"]


class TestRefusals:
    @pytest.mark.asyncio
    async def test_a_deployment_that_cannot_build_writes_nothing(self) -> None:
        client = FakeEntityClient()
        with pytest.raises(BackendRejectedError, match="sandbox.image"):
            await _submit(client, config=_config(sandbox={"image": None}))
        assert client.events == []

    @pytest.mark.asyncio
    async def test_a_secret_the_push_step_needs_but_the_caller_cannot_read_writes_nothing(self) -> None:
        client = FakeEntityClient()
        with pytest.raises(MissingSecrets, match="default/builder-signing-key$"):
            await _submit(client, unreadable=frozenset({"default/builder-signing-key"}))
        assert client.events == []

    @pytest.mark.asyncio
    async def test_a_secret_the_deployment_turned_off_is_not_required(self) -> None:
        client = FakeEntityClient()
        unreadable = frozenset({"default/builder-signing-key"})
        await _submit(client, config=_config(signing_key_secret=None), unreadable=unreadable)
        assert client.events[-1] == "create_job"

    @pytest.mark.asyncio
    async def test_a_request_the_plan_refuses_writes_nothing(self) -> None:
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
        assert request.custom_fields == {
            "images": ["demo-1.img0", "demo-1.img1"],
            "request_digest": request_digest(_set()),
        }

    @pytest.mark.asyncio
    async def test_each_row_resolves_the_tag_its_own_image_pushes_and_no_other(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        result = await _submit(client, jobs, _set(3))
        push = PushStepConfig.model_validate(jobs.jobs["demo-1"].platform_spec.steps[2].config)
        pushed = [image.refs for image in push.images]

        for index, row in enumerate(result.images):
            tag = compose_system_tag(row.workspace, row.name)
            owners = [i for i, tags in enumerate(pushed) if any(t.endswith(f":{tag}") for t in tags)]
            assert owners == [index], f"row {row.name} tag {tag!r} is pushed by images {owners}"
