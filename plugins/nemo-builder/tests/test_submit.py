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
from nemo_builder_plugin.entities import ContainerImage, QueuedBuild
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.identity import ImageReference, compose_system_tag
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource, ImageSource
from nemo_builder_plugin.steps import FetchStepConfig, PushStepConfig, SuperviseStepConfig
from nemo_builder_plugin.submit import (
    JOB_SOURCE,
    BuildConflict,
    InvalidBuildRequest,
    MissingSecrets,
    SubmitResult,
    UpstreamUnavailable,
    request_digest,
    submit_build_set,
)
from nemo_builder_plugin.upstream import UpstreamError, UpstreamNotFound
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities import EntityConflictError, EntityNotFoundError
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
        if name not in self.rows:
            raise EntityNotFoundError(f"{name} not found")
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
    upstream: Mapping[str, str | Exception] | None = None,
    queue: bool = False,
) -> SubmitResult:
    jobs = jobs or FakeJobs(entity_client.events)

    async def secret_readable(workspace: str, name: str) -> bool:
        return f"{workspace}/{name}" not in unreadable

    async def resolve_upstream(reference: ImageReference, platform: str) -> str:
        entity_client.events.append(f"resolve:{reference}:{platform}")
        found = (upstream or {})[str(reference)]
        if isinstance(found, Exception):
            raise found
        return found

    return await submit_build_set(
        build_set or _set(),
        backend=ExecutionBackend(config or _config(), PROFILES),
        workspace="default",
        entity_client=entity_client,
        create_job=jobs.create_job,
        get_job_fields=jobs.get_job_fields,
        secret_readable=secret_readable,
        resolve_upstream=resolve_upstream,
        default_platform=(config or _config()).default_platform,
        queue=queue,
    )


class TestOrdering:
    @pytest.mark.asyncio
    async def test_a_capped_submit_queues_the_job_instead_of_creating_it(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        result = await _submit(client, jobs, queue=True)
        assert [i.status for i in result.images] == ["queued", "queued"]
        assert client.events == ["create:demo-1.img0", "create:demo-1.img1", "create:builder-demo-1"]
        assert jobs.jobs == {}
        queued = client.rows["builder-demo-1"]
        assert isinstance(queued, QueuedBuild)
        assert (queued.state, queued.images) == ("queued", ["demo-1.img0", "demo-1.img1"])
        request = CreateHelixJobRequest.model_validate(queued.request)
        assert (request.name, request.source) == ("builder-demo-1", JOB_SOURCE)

        # A resubmit adopts the rows and the queue entry; it doesn't queue the build twice.
        client.events.clear()
        await _submit(client, jobs, queue=True)
        assert "conflict:builder-demo-1" in client.events
        assert jobs.jobs == {}

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
        assert all(
            image.provenance.request_digest == request_digest(_set().with_default_platform("linux/amd64"))
            for image in result.images
        )

    @pytest.mark.asyncio
    async def test_every_row_names_its_job_and_spec(self) -> None:
        client = FakeEntityClient()
        result = await _submit(client)
        provenance = result.images[0].provenance
        assert (provenance.job, provenance.spec) == ("default/builder-demo-1", "img0")


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
            "get_job:builder-demo-1",
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
        jobs.jobs["builder-demo-1"] = CreateHelixJobRequest.model_construct(
            name="builder-demo-1", custom_fields={"owner": "someone"}
        )
        with pytest.raises(BuildConflict, match="not created by this request"):
            await _submit(client, jobs)
        assert client.events == [
            "create:demo-1.img0",
            "create:demo-1.img1",
            "job_conflict",
            "get_job:builder-demo-1",
            "update:demo-1.img0:failed",
            "update:demo-1.img1:failed",
        ]
        assert all(row.status_detail and "held by a job" in row.status_detail for row in client.rows.values())

    @pytest.mark.asyncio
    async def test_submitting_again_into_the_held_name_writes_nothing_more(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        jobs.jobs["builder-demo-1"] = CreateHelixJobRequest.model_construct(
            name="builder-demo-1", custom_fields={"owner": "someone"}
        )
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
        jobs.jobs["builder-demo-1"] = CreateHelixJobRequest.model_construct(
            name="builder-demo-1", custom_fields={"owner": "someone"}
        )
        with pytest.raises(BuildConflict):
            await _submit(client, jobs)
        assert client.events[-2:] == ["update_conflict:demo-1.img0", "update:demo-1.img1:failed"]


class TestRefusals:
    @pytest.mark.asyncio
    async def test_a_deployment_that_cannot_build_writes_nothing(self) -> None:
        client = FakeEntityClient()
        with pytest.raises(BackendRejectedError, match="builder.registry"):
            await _submit(client, config=_config(registry=None))
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
        request = jobs.jobs["builder-demo-1"]
        assert request.source == JOB_SOURCE
        assert request.custom_fields == {
            "images": ["demo-1.img0", "demo-1.img1"],
            "request_digest": request_digest(_set().with_default_platform("linux/amd64")),
        }

    @pytest.mark.asyncio
    async def test_each_row_resolves_the_tag_its_own_image_pushes_and_no_other(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        result = await _submit(client, jobs, _set(3))
        push = PushStepConfig.model_validate(jobs.jobs["builder-demo-1"].platform_spec.steps[2].config)
        pushed = [image.refs for image in push.images]

        for index, row in enumerate(result.images):
            tag = compose_system_tag(row.workspace, row.name)
            owners = [i for i, tags in enumerate(pushed) if any(t.endswith(f":{tag}") for t in tags)]
            assert owners == [index], f"row {row.name} tag {tag!r} is pushed by images {owners}"


UBUNTU = "docker.io/library/ubuntu:24.04"
UBUNTU_AMD64 = "sha256:" + "c" * 64


def _copy_set(*images: str, build: bool = False) -> BuildSet:
    specs = [BuildSpec(name=f"copy{i}", source=ImageSource(image=image)) for i, image in enumerate(images)]
    if build:
        specs.append(BuildSpec(name="built", source=FileSetSource(fileset="fs-a")))
    return BuildSet(name="demo", revision=1, build_specs=specs)


class TestCopies:
    @pytest.mark.asyncio
    async def test_a_copy_records_the_manifest_its_tag_named_at_submit(self) -> None:
        entities = FakeEntityClient()
        result = await _submit(entities, build_set=_copy_set("ubuntu:24.04"), upstream={UBUNTU: UBUNTU_AMD64})

        upstream = result.images[0].provenance.upstream
        assert upstream is not None
        assert (upstream.image, upstream.manifest_digest) == (UBUNTU, UBUNTU_AMD64)
        assert entities.events[:2] == ["get:demo-1.copy0", f"resolve:{UBUNTU}:linux/amd64"]

    @pytest.mark.asyncio
    async def test_a_resubmitted_copy_reuses_its_pinned_digest_without_asking_the_registry(self) -> None:
        """Evals resubmits a copy every time a task needs it, so a rate limit there mustn't fail an image we have."""
        entities = FakeEntityClient()
        jobs = FakeJobs(entities.events)
        await _submit(entities, jobs, build_set=_copy_set("ubuntu:24.04"), upstream={UBUNTU: UBUNTU_AMD64})
        resolved_once = entities.events.count(f"resolve:{UBUNTU}:linux/amd64")

        result = await _submit(
            entities, jobs, build_set=_copy_set("ubuntu:24.04"), upstream={UBUNTU: UpstreamError("429 rate limited")}
        )

        assert entities.events.count(f"resolve:{UBUNTU}:linux/amd64") == resolved_once == 1
        upstream = result.images[0].provenance.upstream
        assert upstream is not None and upstream.manifest_digest == UBUNTU_AMD64

    @pytest.mark.asyncio
    async def test_a_retry_after_a_failed_job_creation_pins_the_digest_its_rows_hold(self) -> None:
        """A tag that moved between the two submits doesn't split the rows and the job across two digests."""
        entities = FakeEntityClient()

        class _Unavailable(FakeJobs):
            async def create_job(self, request: CreateHelixJobRequest) -> CreateHelixJobRequest:
                raise RuntimeError("jobs unavailable")

        with pytest.raises(RuntimeError):
            await _submit(
                entities,
                _Unavailable(entities.events),
                build_set=_copy_set("ubuntu:24.04"),
                upstream={UBUNTU: UBUNTU_AMD64},
            )
        jobs = FakeJobs(entities.events)
        await _submit(entities, jobs, build_set=_copy_set("ubuntu:24.04"), upstream={UBUNTU: "sha256:" + "e" * 64})

        fetch = FetchStepConfig.model_validate(next(iter(jobs.jobs.values())).platform_spec.steps[0].config)
        assert [i.ref for i in fetch.imports] == [f"docker.io/library/ubuntu@{UBUNTU_AMD64}"]

    @pytest.mark.asyncio
    async def test_a_copy_whose_name_another_request_holds_conflicts_without_asking_the_registry(self) -> None:
        """The request is a conflict whatever the registry says, so a registry failure mustn't hide it."""
        built_first = BuildSet(
            name="demo", revision=1, build_specs=[BuildSpec(name="copy0", source=FileSetSource(fileset="fs-a"))]
        )
        entities = FakeEntityClient()
        await _submit(entities, build_set=built_first)
        events_before = list(entities.events)

        with pytest.raises(BuildConflict, match="different request"):
            await _submit(
                entities, build_set=_copy_set("ubuntu:24.04"), upstream={UBUNTU: UpstreamError("429 rate limited")}
            )

        assert entities.events[len(events_before) :] == ["get:demo-1.copy0"]

    @pytest.mark.asyncio
    async def test_a_new_revision_resolves_its_copies_again(self) -> None:
        entities = FakeEntityClient()
        await _submit(entities, build_set=_copy_set("ubuntu:24.04"), upstream={UBUNTU: UBUNTU_AMD64})
        next_revision = _copy_set("ubuntu:24.04")
        next_revision.revision = 2

        result = await _submit(entities, build_set=next_revision, upstream={UBUNTU: "sha256:" + "e" * 64})

        upstream = result.images[0].provenance.upstream
        assert upstream is not None and upstream.manifest_digest == "sha256:" + "e" * 64

    @pytest.mark.asyncio
    async def test_a_set_of_copies_pulls_and_pushes_without_a_build_step(self) -> None:
        entities = FakeEntityClient()
        jobs = FakeJobs(entities.events)
        await _submit(entities, jobs, build_set=_copy_set("ubuntu:24.04"), upstream={UBUNTU: UBUNTU_AMD64})

        steps = next(iter(jobs.jobs.values())).platform_spec.steps
        assert [step.name for step in steps] == ["fetch", "push"]
        fetch = FetchStepConfig.model_validate(steps[0].config)
        assert fetch.sources == []
        assert [(i.image, i.ref, i.platform) for i in fetch.imports] == [
            ("demo-1.copy0", f"docker.io/library/ubuntu@{UBUNTU_AMD64}", "linux/amd64")
        ]
        assert PushStepConfig.model_validate(steps[1].config).images[0].expected_digest == UBUNTU_AMD64

    @pytest.mark.asyncio
    async def test_a_set_mixing_copies_and_builds_builds_only_the_builds(self) -> None:
        entities = FakeEntityClient()
        jobs = FakeJobs(entities.events)
        await _submit(entities, jobs, build_set=_copy_set("ubuntu:24.04", build=True), upstream={UBUNTU: UBUNTU_AMD64})

        steps = next(iter(jobs.jobs.values())).platform_spec.steps
        assert [step.name for step in steps] == ["fetch", "build", "push"]
        groups = SuperviseStepConfig.model_validate(steps[1].config).groups
        assert [image.image for group in groups for image in group.images] == ["demo-1.built"]
        pushed = PushStepConfig.model_validate(steps[2].config).images
        assert [(image.image, image.expected_digest) for image in pushed] == [
            ("demo-1.copy0", UBUNTU_AMD64),
            ("demo-1.built", None),
        ]

    @pytest.mark.asyncio
    async def test_a_copy_from_a_registry_the_deployment_does_not_allow_writes_nothing(self) -> None:
        entities = FakeEntityClient()
        with pytest.raises(BackendRejectedError, match="registry.example.org"):
            await _submit(entities, build_set=_copy_set("registry.example.org/app:1"))
        assert entities.events == []

    @pytest.mark.asyncio
    async def test_an_empty_allowlist_refuses_every_copy(self) -> None:
        with pytest.raises(BackendRejectedError, match="allowed: none"):
            await _submit(FakeEntityClient(), build_set=_copy_set("ubuntu:24.04"), config=_config(import_registries=[]))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("failure", "error"),
        [
            (UpstreamNotFound("no linux/amd64 image"), InvalidBuildRequest),
            (UpstreamError("timed out"), UpstreamUnavailable),
        ],
    )
    async def test_an_image_that_does_not_resolve_writes_nothing(
        self, failure: Exception, error: type[Exception]
    ) -> None:
        entities = FakeEntityClient()
        with pytest.raises(error, match="demo-1.copy0"):
            await _submit(entities, build_set=_copy_set("ubuntu:24.04"), upstream={UBUNTU: failure})
        assert entities.events == ["get:demo-1.copy0", f"resolve:{UBUNTU}:linux/amd64"]


class TestPlatforms:
    """A spec that names no platform is built for the deployment's `default_platform`."""

    @staticmethod
    def _supervise_platforms(jobs: FakeJobs) -> list[str]:
        steps = next(iter(jobs.jobs.values())).platform_spec.steps
        supervise = next(step for step in steps if step.name == "build")
        return [i.platform for g in SuperviseStepConfig.model_validate(supervise.config).groups for i in g.images]

    @pytest.mark.asyncio
    async def test_the_default_platform_builds_a_spec_that_names_none(self) -> None:
        entities = FakeEntityClient()
        jobs = FakeJobs(entities.events)
        await _submit(entities, jobs, build_set=_set(1), config=_config(default_platform="linux/arm64"))

        assert self._supervise_platforms(jobs) == ["linux/arm64"]
        assert next(iter(jobs.jobs.values())).spec["build_specs"][0]["platform"] == "linux/arm64"

    @pytest.mark.asyncio
    async def test_a_spec_that_names_a_platform_keeps_it(self) -> None:
        entities = FakeEntityClient()
        jobs = FakeJobs(entities.events)
        build_set = _set(1)
        build_set.build_specs[0].platform = "linux/amd64"
        await _submit(entities, jobs, build_set=build_set, config=_config(default_platform="linux/arm64"))

        assert self._supervise_platforms(jobs) == ["linux/amd64"]

    @pytest.mark.asyncio
    async def test_a_copy_resolves_the_default_platform(self) -> None:
        entities = FakeEntityClient()
        await _submit(
            entities,
            build_set=_copy_set("ubuntu:24.04"),
            config=_config(default_platform="linux/arm64"),
            upstream={UBUNTU: UBUNTU_AMD64},
        )

        assert entities.events[:2] == ["get:demo-1.copy0", f"resolve:{UBUNTU}:linux/arm64"]

    @pytest.mark.asyncio
    async def test_naming_the_default_is_the_same_request_as_naming_nothing(self) -> None:
        """Requests from before the default was configurable, which left it out, keep their digest."""
        explicit = _set(1)
        explicit.build_specs[0].platform = "linux/amd64"
        result = await _submit(FakeEntityClient(), build_set=_set(1))

        assert result.images[0].provenance.request_digest == request_digest(explicit)
