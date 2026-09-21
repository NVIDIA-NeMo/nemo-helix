# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submit ordering, against fakes.

What is being tested is a *sequence*, not a return value: rows exist before the job does, and a
losing racer adopts rather than duplicates. Both are invisible in a response body and both are
the reason the design has no orphan-detection sweep.
"""

from __future__ import annotations

import pytest
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, JobOrigin
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.submit import JOB_SOURCE, SubmitResult, submit_build_set
from nemo_platform_plugin.entities import EntityConflictError
from nemo_platform_plugin.jobs.types import CreatePlatformJobRequest


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


def _config() -> BuilderConfig:
    return BuilderConfig(
        default_registry="reg.example.com",
        push_secret="my-reg-secret",
        signing_key="k8s://nmp-builds/cosign-key",
    )


def _set(n: int = 2) -> BuildSet:
    return BuildSet(
        name="demo",
        revision=1,
        build_specs=[
            BuildSpec(
                name=f"img{i}",
                source=FileSetSource(fileset="fs-a"),
                output=BuildOutput(repository=f"team/img{i}", tag="v1"),
            )
            for i in range(n)
        ],
    )


async def _submit(entity_client: FakeEntityClient, events: list[str]) -> SubmitResult:
    async def create_job(request: CreatePlatformJobRequest) -> CreatePlatformJobRequest:
        events.append("create_job")
        return request

    return await submit_build_set(
        _set(),
        config=_config(),
        workspace="default",
        entity_client=entity_client,
        create_job=create_job,
    )


class TestOrdering:
    @pytest.mark.asyncio
    async def test_rows_are_created_before_the_job(self) -> None:
        """The inversion that removes the orphan problem.

        If the job were created first, a pod dying between push and row-write would leave a
        pushed image with nothing pointing at it.
        """
        client = FakeEntityClient()
        await _submit(client, client.events)
        assert client.events == ["create:demo-1-0", "create:demo-1-1", "create_job"]

    @pytest.mark.asyncio
    async def test_rows_are_created_pending_with_nothing_observed(self) -> None:
        client = FakeEntityClient()
        result = await _submit(client, client.events)
        assert [i.status for i in result.images] == ["pending", "pending"]
        assert all(i.digest is None and i.signature is None for i in result.images)


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_a_losing_racer_adopts_the_existing_rows(self) -> None:
        """Settled by the unique-name index, not by a read-then-write.

        Row names are deterministic from the request, so the row that already exists is the row
        this request would have created. Adopting is correct; failing would make a duplicate
        submission an error for no reason, and re-reading first would be a race.
        """
        client = FakeEntityClient()
        await _submit(client, client.events)
        client.events.clear()

        await _submit(client, client.events)
        assert client.events == [
            "conflict:demo-1-0",
            "get:demo-1-0",
            "conflict:demo-1-1",
            "get:demo-1-1",
            "create_job",
        ]
        assert len(client.rows) == 2


class TestTheJobRequest:
    @pytest.mark.asyncio
    async def test_the_job_names_this_service_so_the_reconciler_finds_its_own(self) -> None:
        client = FakeEntityClient()
        captured: list[CreatePlatformJobRequest] = []

        async def create_job(request: CreatePlatformJobRequest) -> CreatePlatformJobRequest:
            captured.append(request)
            return request

        await submit_build_set(
            _set(),
            config=_config(),
            workspace="default",
            entity_client=client,
            create_job=create_job,
        )
        request = captured[0]
        assert request.source == JOB_SOURCE
        assert request.name == "demo-1"
        assert request.custom_fields == {"images": ["demo-1-0", "demo-1-1"]}

    @pytest.mark.asyncio
    async def test_the_system_tag_on_the_rows_matches_the_one_the_build_pushes(self) -> None:
        """Composed once and handed to both. Composing it twice is a drift bug whose only
        symptom is a reconciler resolving a tag nothing ever pushed."""
        client = FakeEntityClient()
        captured: list[CreatePlatformJobRequest] = []

        async def create_job(request: CreatePlatformJobRequest) -> CreatePlatformJobRequest:
            captured.append(request)
            return request

        result = await submit_build_set(
            _set(1),
            config=_config(),
            workspace="default",
            entity_client=client,
            create_job=create_job,
        )
        origin = result.images[0].provenance.built_by
        assert isinstance(origin, JobOrigin)

        pushed_tags = captured[0].platform_spec.steps[2].config["images"][0]["tags"]
        assert any(t.endswith(f":{origin.system_tag}") for t in pushed_tags), (
            f"row says {origin.system_tag!r}, build pushes {pushed_tags!r}"
        )
