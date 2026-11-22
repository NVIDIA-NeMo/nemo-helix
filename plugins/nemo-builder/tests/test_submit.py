# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submit ordering, against fakes: what every backend's submit keeps, shown for each backend, and the rest
for a build in this cluster."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import httpx
import pytest
from nemo_builder_plugin.backend import Backend, BackendRejectedError
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.identity import ImageReference, compose_system_tag
from nemo_builder_plugin.remote import RemoteBackend
from nemo_builder_plugin.schema import BuildMount, BuildOutput, BuildSet, BuildSpec, FileSetSource, ImageSource
from nemo_builder_plugin.steps import FetchStepConfig, PushStepConfig, SuperviseStepConfig
from nemo_builder_plugin.submit import (
    JOB_SOURCE,
    BuildConflict,
    CreatedJob,
    ExistingJob,
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

#: The one profile the step of a remote build service runs on.
REMOTE_PROFILES = [
    KubernetesJobExecutionProfile.model_validate({"profile": "build-remote", "config": {"namespace": "nhx-remote"}})
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

    async def create_job(self, request: CreateHelixJobRequest) -> CreatedJob:
        assert request.name is not None
        if request.name in self.jobs:
            self.events.append("job_conflict")
            raise ConflictError(
                httpx.Response(409, json={"detail": "exists"}, request=httpx.Request("POST", "http://j"))
            )
        self.events.append("create_job")
        self.jobs[request.name] = request
        # As Jobs answers a create: with the job's own ID, and its first attempt's.
        return CreatedJob(uid=f"platform-job-{request.name}", attempt_id=f"attempt-of-{request.name}")

    async def get_job(self, name: str) -> ExistingJob:
        self.events.append(f"get_job:{name}")
        return ExistingJob(uid=f"platform-job-{name}", custom_fields=self.jobs[name].custom_fields or {})


def _config(**overrides: object) -> BuilderConfig:
    settings: dict[str, object] = {
        "registry": "reg.example.com",
        "sandbox": {"image": "kaniko.example.com/executor:debug"},
    }
    return BuilderConfig.model_validate(settings | overrides)


def _remote_config() -> BuilderConfig:
    """A deployment whose images a remote build service builds, two to a set at most."""
    return BuilderConfig.model_validate(
        {
            "backend": "remote",
            "remote": {
                "registry": "registry.example.com",
                "repository": "team/images",
                "image": "steps.example.com/remote-build:1",
                "max_specs_per_set": 2,
            },
        }
    )


def _set(n: int = 2, *, tag: str | None = "v1", fileset: str = "fs-a", names: list[str] | None = None) -> BuildSet:
    """With no ``tag``, no spec names an output: a remote build service publishes every image to its one
    repository, and its backend refuses a spec that names an output at all."""
    return BuildSet(
        name="demo",
        revision=1,
        build_specs=[
            BuildSpec(
                name=name,
                source=FileSetSource(fileset=fileset),
                output=BuildOutput(repository=f"team/{name}", tag=tag) if tag else None,
            )
            for name in names or [f"img{i}" for i in range(n)]
        ],
    )


def _with_a_helper(build_set: BuildSet) -> BuildSet:
    """``build_set`` with its last image marked a helper, and nothing else of it changed."""
    *others, last = build_set.build_specs
    return build_set.model_copy(update={"build_specs": [*others, last.model_copy(update={"role": "helper"})]})


def _digest_of_the_record(job: CreateHelixJobRequest) -> str:
    """sha256 of the canonical JSON of the request as ``job`` records it, in its ``spec``."""
    canonical = json.dumps(job.spec, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    """One backend, as the tests of what every backend's submit keeps need it: requests it takes, and what a
    submit does with its rows and its job that the other backend's does not."""

    backend: Backend
    #: A request the backend takes, for two images: ``img0`` and ``img1``.
    build_set: BuildSet
    #: Another request for the same two images, at the same revision.
    different: BuildSet
    #: A request the backend takes, for the images so named.
    naming: Callable[[list[str]], BuildSet]
    #: What a submit of ``build_set`` writes once it has created its job, and what it reads once it has found
    #: the job already there. Nothing, for a build in this cluster. A row a build service builds records its
    #: job, and a job found under the build's name is the build's own only if the rows record it.
    written_after_the_job: list[str]
    read_after_the_job: list[str]
    #: The job, and the attempt of it, that each row then records.
    job_recorded: tuple[str | None, str | None]


def _in_cluster() -> Case:
    return Case(
        backend=ExecutionBackend(_config(), PROFILES),
        build_set=_set(),
        different=_set(tag="v2"),
        naming=lambda names: _set(names=names),
        written_after_the_job=[],
        read_after_the_job=[],
        job_recorded=(None, None),
    )


def _remote() -> Case:
    return Case(
        backend=RemoteBackend(_remote_config(), REMOTE_PROFILES),
        build_set=_set(tag=None),
        # No output to name another tag with: it builds from another fileset.
        different=_set(tag=None, fileset="fs-b"),
        naming=lambda names: _set(tag=None, names=names),
        written_after_the_job=["update:demo-1.img0:pending", "update:demo-1.img1:pending"],
        read_after_the_job=["get:demo-1.img0", "get:demo-1.img1"],
        # As `FakeJobs` answers the create.
        job_recorded=("platform-job-builder-demo-1", "attempt-of-builder-demo-1"),
    )


@pytest.fixture(params=[_in_cluster, _remote], ids=["execution", "remote"])
def case(request: pytest.FixtureRequest) -> Case:
    """Each backend in turn. A test that takes it shows a rule of the generic submit path, for both."""
    return request.param()


async def _submit(
    entity_client: FakeEntityClient,
    jobs: FakeJobs | None = None,
    build_set: BuildSet | None = None,
    config: BuilderConfig | None = None,
    unreadable: frozenset[str] = frozenset(),
    upstream: Mapping[str, str | Exception] | None = None,
    backend: Backend | None = None,
) -> SubmitResult:
    """Submit to ``backend``: unless a test names one, a deployment that builds in this cluster, as ``config``
    has it."""
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
        backend=backend or ExecutionBackend(config or _config(), PROFILES),
        workspace="default",
        entity_client=entity_client,
        create_job=jobs.create_job,
        get_job=jobs.get_job,
        secret_readable=secret_readable,
        resolve_upstream=resolve_upstream,
        default_platform=(config or _config()).default_platform,
    )


class TestOrdering:
    """The tests that take a ``case`` hold whichever backend builds, and run for each."""

    @pytest.mark.asyncio
    async def test_rows_are_created_before_the_job(self, case: Case) -> None:
        client = FakeEntityClient()
        await _submit(client, build_set=case.build_set, backend=case.backend)
        assert client.events == [
            "create:demo-1.img0",
            "create:demo-1.img1",
            "create_job",
            *case.written_after_the_job,
        ]

    @pytest.mark.asyncio
    async def test_rows_are_created_pending_with_nothing_observed(self, case: Case) -> None:
        client = FakeEntityClient()
        result = await _submit(client, build_set=case.build_set, backend=case.backend)
        assert [i.status for i in result.images] == ["pending", "pending"]
        assert all(i.digest is None for i in result.images)

    @pytest.mark.asyncio
    async def test_every_row_records_the_request_that_wrote_it(self, case: Case) -> None:
        client = FakeEntityClient()
        # Taken before the submit: the digest of the request as it was sent, whatever a submit does with it.
        expected = request_digest(case.build_set.with_default_platform("linux/amd64"))
        result = await _submit(client, build_set=case.build_set, backend=case.backend)
        assert all(image.provenance.request_digest == expected for image in result.images)

    @pytest.mark.asyncio
    async def test_every_row_names_its_job_and_spec(self, case: Case) -> None:
        client = FakeEntityClient()
        result = await _submit(client, build_set=case.build_set, backend=case.backend)
        provenance = result.images[0].provenance
        assert (provenance.job, provenance.spec) == ("default/builder-demo-1", "img0")

    @pytest.mark.asyncio
    async def test_a_set_that_marks_an_image_a_helper_is_submitted_as_any_other(self, case: Case) -> None:
        """Either backend takes a role, and the job a submit creates records the request with it. What the
        job's steps are told of one is the backend's own, and shown with each: for a build in this cluster
        below, with the job's request, and for a build service with the remote backend's submits."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        helper = _with_a_helper(case.build_set)
        expected = request_digest(helper.with_default_platform("linux/amd64"))
        result = await _submit(client, jobs, helper, backend=case.backend)
        assert client.events == [
            "create:demo-1.img0",
            "create:demo-1.img1",
            "create_job",
            *case.written_after_the_job,
        ]
        assert all(image.provenance.request_digest == expected for image in result.images)
        job = jobs.jobs["builder-demo-1"]
        plain, marked = job.spec["build_specs"]
        assert (plain["name"], "role" in plain) == ("img0", False)
        assert (marked["name"], marked["role"]) == ("img1", "helper")
        assert _digest_of_the_record(job) == expected

    @pytest.mark.asyncio
    async def test_a_row_built_in_this_cluster_is_written_once_and_records_no_more_of_its_job(self) -> None:
        """Only a row that is completed on its caller's word records which job its submit created. Jobs answers
        this submit's create with an ID and an attempt too, and the row is as it was before a row could hold
        them: not written again, and stored without them."""
        client = FakeEntityClient()
        result = await _submit(client)
        assert not [event for event in client.events if event.startswith("update")]
        for row in (*result.images, *client.rows.values()):
            assert (row.provenance.job_uid, row.provenance.job_attempt_id) == (None, None)
            assert set(row._get_data_fields()["provenance"]) == {
                "build_set",
                "revision",
                "spec",
                "job",
                "request_digest",
                "upstream",
                "backend",
            }


class TestResubmission:
    """Settled by the unique-name index, not a read-then-write, and by the request digest.

    The tests that take a ``case`` hold whichever backend builds, and run for each. The others show what a
    build in this cluster does with a job it finds under its name: it takes or refuses the job by the digest
    the job carries. A backend whose rows record their job goes by that record instead, which the tests of
    the remote backend show.
    """

    @pytest.mark.asyncio
    async def test_the_same_request_again_adopts_the_rows_and_the_job(self, case: Case) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        # An equal request of its own, copied before the first is submitted: not the object that submit was given.
        again = case.build_set.model_copy(deep=True)
        first = await _submit(client, jobs, case.build_set, backend=case.backend)
        client.events.clear()

        second = await _submit(client, jobs, again, backend=case.backend)
        # Nothing is created, and nothing written.
        assert client.events == [
            "conflict:demo-1.img0",
            "get:demo-1.img0",
            "conflict:demo-1.img1",
            "get:demo-1.img1",
            "job_conflict",
            "get_job:builder-demo-1",
            *case.read_after_the_job,
        ]
        assert [i.name for i in second.images] == [i.name for i in first.images]
        assert len(client.rows) == 2 and len(jobs.jobs) == 1
        # The rows are returned as they are stored. Where a row records its job, the job under the build's name
        # is the one it records: the one the first submit created.
        recorded = [(i.provenance.job_uid, i.provenance.job_attempt_id) for i in second.images]
        assert recorded == [case.job_recorded, case.job_recorded]

    @pytest.mark.asyncio
    async def test_a_job_under_the_builds_name_is_its_own_by_the_digest_it_carries(self) -> None:
        """As it always was for a build in this cluster, whose rows record no more of their job than its name:
        a job that says it is of this request is taken for the build's, whoever created it, and no row is
        read again or written to find out."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        jobs.jobs["builder-demo-1"] = CreateHelixJobRequest.model_construct(
            name="builder-demo-1",
            custom_fields={"request_digest": request_digest(_set().with_default_platform("linux/amd64"))},
        )
        result = await _submit(client, jobs)
        assert client.events == [
            "create:demo-1.img0",
            "create:demo-1.img1",
            "job_conflict",
            "get_job:builder-demo-1",
        ]
        assert [image.status for image in result.images] == ["pending", "pending"]

    @pytest.mark.asyncio
    async def test_a_different_request_at_the_same_revision_is_refused_and_adopts_nothing(self, case: Case) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        await _submit(client, jobs, case.build_set, backend=case.backend)
        client.events.clear()

        with pytest.raises(BuildConflict, match="new revision"):
            await _submit(client, jobs, case.different, backend=case.backend)
        assert client.events == ["conflict:demo-1.img0", "get:demo-1.img0"]

    @pytest.mark.asyncio
    async def test_the_same_set_with_and_without_a_helper_are_two_requests_at_one_revision(self, case: Case) -> None:
        """An image's role is part of the request, on either backend: a caller that starts to mark a helper in
        a set it has already submitted needs the next revision, and so does one that stops."""
        plain, helper = case.build_set, _with_a_helper(case.build_set)
        for first, second in ((plain, helper), (helper, plain)):
            client = FakeEntityClient()
            jobs = FakeJobs(client.events)
            await _submit(client, jobs, first, backend=case.backend)
            client.events.clear()

            with pytest.raises(BuildConflict, match="new revision"):
                await _submit(client, jobs, second, backend=case.backend)
            assert client.events == ["conflict:demo-1.img0", "get:demo-1.img0"]

    @pytest.mark.asyncio
    async def test_a_set_with_a_helper_sent_again_is_the_same_request(self, case: Case) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        helper = _with_a_helper(case.build_set)
        await _submit(client, jobs, helper, backend=case.backend)
        client.events.clear()

        again = await _submit(client, jobs, _with_a_helper(case.build_set), backend=case.backend)
        assert client.events == [
            "conflict:demo-1.img0",
            "get:demo-1.img0",
            "conflict:demo-1.img1",
            "get:demo-1.img1",
            "job_conflict",
            "get_job:builder-demo-1",
            *case.read_after_the_job,
        ]
        assert [image.status for image in again.images] == ["pending", "pending"]

    @pytest.mark.asyncio
    async def test_rows_written_before_a_collision_are_failed(self, case: Case) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        await _submit(client, jobs, case.naming(["img0"]), backend=case.backend)
        client.events.clear()

        with pytest.raises(BuildConflict, match="new revision"):
            await _submit(client, jobs, case.naming(["extra", "img0"]), backend=case.backend)
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
    async def test_the_job_records_the_request_its_digest_is_taken_over(self, case: Case) -> None:
        """A job's `spec` is the request as it is built, and the digest the job and its rows carry is the sha256
        of that record's canonical JSON. A request that names no role is recorded with none, null or
        otherwise: the record a builder from before roles wrote of the same request."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        result = await _submit(client, jobs, case.build_set, backend=case.backend)
        job = jobs.jobs["builder-demo-1"]
        assert [spec["name"] for spec in job.spec["build_specs"]] == ["img0", "img1"]
        assert not any("role" in spec for spec in job.spec["build_specs"])
        assert job.custom_fields is not None
        assert _digest_of_the_record(job) == job.custom_fields["request_digest"]
        assert all(image.provenance.request_digest == _digest_of_the_record(job) for image in result.images)

    @pytest.mark.asyncio
    async def test_a_helper_built_in_this_cluster_is_the_job_it_is_without_the_role(self) -> None:
        """From the request to the job a submit creates for it. No step of a build in this cluster is told of
        a role, so the steps the job is given are the ones the same set is given unmarked, as Jobs is sent
        them."""
        sent = []
        for build_set in (_set(), _with_a_helper(_set())):
            client = FakeEntityClient()
            jobs = FakeJobs(client.events)
            await _submit(client, jobs, build_set)
            sent.append(jobs.jobs["builder-demo-1"].platform_spec.model_dump(mode="json", exclude_unset=True))
        unmarked, marked = sent
        assert [step["name"] for step in marked["steps"]] == ["fetch", "build", "push"]
        assert marked == unmarked

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
            async def create_job(self, request: CreateHelixJobRequest) -> CreatedJob:
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


class TestTheRequestDigest:
    """What a row records of the request that wrote it, and what a submit of its revision is compared with."""

    @pytest.mark.parametrize(
        ("as_sent", "spelt_out", "digest"),
        [
            pytest.param(
                BuildSet(
                    name="demo",
                    revision=1,
                    build_specs=[
                        BuildSpec(
                            name="img0",
                            source=FileSetSource(fileset="fs-a"),
                            output=BuildOutput(repository="team/img0", tag="v1"),
                            mounts=[BuildMount(source=FileSetSource(fileset="fs-tools"), target="/mnt/tools")],
                        )
                    ],
                ),
                BuildSet(
                    name="demo",
                    revision=1,
                    build_specs=[
                        BuildSpec(
                            name="img0",
                            source=FileSetSource(fileset="fs-a", archive=None, context_path=None),
                            output=BuildOutput(repository="team/img0", tag="v1"),
                            dockerfile="Dockerfile",
                            mounts=[
                                BuildMount(
                                    source=FileSetSource(fileset="fs-tools", archive=None, context_path=None),
                                    target="/mnt/tools",
                                )
                            ],
                            platform="linux/amd64",
                        )
                    ],
                ),
                "sha256:290ca8914aa9b5fda58ba891643c8060fbeb2c4f2e7dd391914db321b65546fb",
                id="a-build-from-a-fileset",
            ),
            pytest.param(
                BuildSet(name="demo", revision=1, build_specs=[BuildSpec(source=ImageSource(image="ubuntu:24.04"))]),
                BuildSet(
                    name="demo",
                    revision=1,
                    build_specs=[
                        # All but `dockerfile`, which a copy may not name: its digest covers the default all the same.
                        BuildSpec(
                            name=None,
                            source=ImageSource(type="image", image="ubuntu:24.04"),
                            output=None,
                            mounts=[],
                            platform="linux/amd64",
                        )
                    ],
                ),
                "sha256:2e300f4aed45e158f0951621e148c03b8e7a15fd6c663204fc91bcd273267fbf",
                id="a-copy",
            ),
        ],
    )
    def test_a_request_keeps_the_digest_its_rows_record(
        self, as_sent: BuildSet, spelt_out: BuildSet, digest: str
    ) -> None:
        """Each digest is written out here, as it was when this test was written, because nothing else holds it.

        A row records the digest of the request that wrote it, and a submit of its revision is the same request
        only at that digest. Every other test computes the digest it expects with ``request_digest`` itself, so
        all of them would pass if the digest of every request changed, and every revision submitted before the
        change would then be refused as a different request when it is submitted again. A change that moves
        one of these on purpose strands the rows that already record it.

        Each request is written twice. Once with every field its digest covers, so what the digest is taken
        over can be read here. And once as short as a caller may send it, so a default that changes fails here
        too. Between the two requests every field of a request is covered, and every default is left out once.
        A field a request gains later has to leave all of them as they are.
        """
        assert request_digest(spelt_out) == digest
        # As a submit computes it: over the request with the deployment's default platform filled in.
        assert request_digest(as_sent.with_default_platform("linux/amd64")) == digest

    def test_a_role_given_as_null_is_the_request_that_names_none(self) -> None:
        """A spec's role is left out of the digest where it has no value, which is what keeps the digests written
        out above where they were when a spec gained the field. Left out by its value, and not by whether the
        request named the field: null is no role, and the two spellings are one request."""
        left_out = BuildSet(name="demo", revision=1, build_specs=[BuildSpec(source=FileSetSource(fileset="fs-a"))])
        as_null = BuildSet(
            name="demo", revision=1, build_specs=[BuildSpec(source=FileSetSource(fileset="fs-a"), role=None)]
        )
        assert "role" in as_null.build_specs[0].model_fields_set
        assert "role" not in left_out.build_specs[0].model_fields_set
        assert request_digest(as_null) == request_digest(left_out)
        # As a submit computes it.
        filled = as_null.with_default_platform("linux/amd64")
        assert "role" in filled.build_specs[0].model_fields_set
        assert request_digest(filled) == request_digest(left_out.with_default_platform("linux/amd64"))

    def test_a_set_of_two_images_keeps_the_digest_it_had_before_a_spec_could_name_a_role(self) -> None:
        """Each request written out above is for one image. This one is for two, and neither names a role.
        Beside its digest is the text the digest is taken over, as it was before a spec could name a role: no
        `role` key, in either spec. A role left out of one spec's dump and written as null in the next would
        leave the digests above where they are, and move that of every request for more than one image."""
        two = BuildSet(
            name="demo",
            revision=1,
            build_specs=[
                BuildSpec(name="main", source=FileSetSource(fileset="fs-a")),
                BuildSpec(name="sidecar", source=FileSetSource(fileset="fs-b")),
            ],
        )
        canonical = (
            '{"build_specs":['
            '{"dockerfile":"Dockerfile","mounts":[],"name":"main","output":null,"platform":"linux/amd64",'
            '"source":{"archive":null,"context_path":null,"fileset":"fs-a"}},'
            '{"dockerfile":"Dockerfile","mounts":[],"name":"sidecar","output":null,"platform":"linux/amd64",'
            '"source":{"archive":null,"context_path":null,"fileset":"fs-b"}}'
            '],"name":"demo","revision":1}'
        )
        digest = "sha256:a9e8005e65bb34ac00b822602ab036f7cc88a7073e9192ce6001fda0eb176487"
        assert "sha256:" + hashlib.sha256(canonical.encode()).hexdigest() == digest
        assert request_digest(two.with_default_platform("linux/amd64")) == digest

    def test_a_helper_is_another_request_than_the_same_set_without_the_role(self) -> None:
        """Which image is the helper is part of the request too."""

        def two(main: Literal["helper"] | None = None, sidecar: Literal["helper"] | None = None) -> BuildSet:
            specs = [
                BuildSpec(name="main", source=FileSetSource(fileset="fs-a"), role=main),
                BuildSpec(name="sidecar", source=FileSetSource(fileset="fs-b"), role=sidecar),
            ]
            return BuildSet(name="demo", revision=1, build_specs=specs)

        digests = [request_digest(build_set) for build_set in (two(), two(sidecar="helper"), two(main="helper"))]
        assert len(set(digests)) == 3

    def test_a_helper_keeps_the_digest_its_rows_record(self) -> None:
        """Written out as the two above are, and for the same reason: nothing else holds the digest of a request
        that marks a helper, and a row records it. Beside it, the text the digest is taken over, so that what
        a role adds to it can be read here: its own key in its spec, as any other field."""
        helper = BuildSet(
            name="demo",
            revision=1,
            build_specs=[BuildSpec(name="sidecar", source=FileSetSource(fileset="fs-a"), role="helper")],
        )
        canonical = (
            '{"build_specs":[{"dockerfile":"Dockerfile","mounts":[],"name":"sidecar","output":null,'
            '"platform":"linux/amd64","role":"helper",'
            '"source":{"archive":null,"context_path":null,"fileset":"fs-a"}}],"name":"demo","revision":1}'
        )
        digest = "sha256:aac161d7feeb9d016bd812eea42b4733be0719d66614139f4f453e6d03962dd3"
        assert "sha256:" + hashlib.sha256(canonical.encode()).hexdigest() == digest
        assert request_digest(helper.with_default_platform("linux/amd64")) == digest
