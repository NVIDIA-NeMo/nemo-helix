# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote backend: how it is selected, what it refuses, where it places an image, the job it compiles, and the
submit path with it."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from nemo_builder_plugin import remote
from nemo_builder_plugin.backend import Backend, BackendRejectedError
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.completion import CompleteRequest, CompletionConflict, complete
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, ExternalImage, Provenance
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.identity import ImageReference
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.remote import CANCEL_MARGIN_SECONDS, STEP_NAME, RemoteBackend
from nemo_builder_plugin.schema import BuildMount, BuildOutput, BuildSet, BuildSpec, FileSetSource, ImageSource
from nemo_builder_plugin.steps import REMOTE_CREDENTIAL_ENV, RemoteBuildStepConfig
from nemo_builder_plugin.submit import (
    BackendChanged,
    BuildConflict,
    MissingSecrets,
    SubmitResult,
    referenced_secrets,
    submit_build_set,
)
from nemo_helix_plugin.auth.workload_identity import WorkloadIdentityConfigError
from nemo_helix_plugin.client.errors import ConflictError
from nemo_helix_plugin.entities import EntityConflictError, EntityNotFoundError
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfile
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobSpec
from nemo_helix_plugin.jobs.types import CreateHelixJobRequest
from pydantic import ValidationError

STEP_IMAGE = "steps.example.com/remote-build:1"


def _profile(name: str = "build-remote", **config: object) -> KubernetesJobExecutionProfile:
    return KubernetesJobExecutionProfile.model_validate({"profile": name, "config": {"namespace": "builds"} | config})


PROFILES: list[ExecutionProfile] = [_profile()]


def _config(default_platform: str = "linux/amd64", **remote: object) -> BuilderConfig:
    settings: dict[str, object] = {
        "registry": "registry.example.com",
        "repository": "team/images",
        "image": STEP_IMAGE,
    }
    return BuilderConfig.model_validate(
        {"backend": "remote", "default_platform": default_platform, "remote": settings | remote}
    )


def _spec(**fields: Any) -> BuildSpec:
    return BuildSpec.model_validate({"source": {"fileset": "fs-a"}} | fields)


def _plan(*specs: BuildSpec, workspace: str = "default") -> BuildPlan:
    build_set = BuildSet(name="demo", revision=1, build_specs=list(specs) or [_spec()])
    return BuildPlan.resolve(build_set.with_default_platform("linux/amd64"), workspace=workspace)


class TestSelection:
    def test_a_deployment_that_sets_nothing_builds_in_cluster(self) -> None:
        backend: Backend = load_backend(BuilderConfig.model_validate({}), [])
        assert isinstance(backend, ExecutionBackend)
        assert backend.name == "execution"

    def test_remote_is_selected_by_name(self) -> None:
        backend: Backend = load_backend(_config(), PROFILES)
        assert isinstance(backend, RemoteBackend)
        assert backend.name == "remote"

    def test_an_unknown_name_refuses_rather_than_build_elsewhere(self) -> None:
        config = BuilderConfig.model_validate({"backend": "remot"})
        with pytest.raises(BackendRejectedError, match="'remot', which is no backend.*execution, remote"):
            load_backend(config, PROFILES)

    def test_a_misspelt_remote_setting_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="repositry"):
            BuilderConfig.model_validate({"backend": "remote", "remote": {"repositry": "team/images"}})

    def test_the_repository_is_a_path_and_the_registry_a_host(self) -> None:
        with pytest.raises(ValidationError, match="remote.registry must be a host"):
            _config(registry="registry.example.com/team")
        assert _config(repository="/team/images/").remote.repository == "team/images"

    def test_a_misspelt_top_level_setting_is_warned_of(self, caplog: pytest.LogCaptureFixture) -> None:
        """The section takes keys it doesn't know: `bakend: remote` would build in this cluster, and say nothing."""
        with caplog.at_level(logging.WARNING):
            config = BuilderConfig.model_validate({"bakend": "remote"})
        assert config.backend == "execution"
        assert "these settings are not known, and are ignored: bakend" in caplog.text

    def test_what_a_deployment_gets_without_saying(self) -> None:
        defaults = BuilderConfig.model_validate({"backend": "remote"}).remote
        assert (defaults.profile, defaults.command, defaults.image) == ("build-remote", ["remote-build"], None)
        assert (defaults.platforms, defaults.max_specs_per_set) == (["linux/amd64"], 1)
        assert (defaults.build_timeout_seconds, defaults.credential_secret) == (3600, None)
        assert defaults.allow_dockerfile_path is True


class TestWhatItRefuses:
    @pytest.mark.parametrize(
        ("spec", "named"),
        [
            pytest.param(_spec(output={"repository": "team/app", "tag": "v1"}), "demo-1: output:", id="output"),
            pytest.param(
                _spec(mounts=[{"source": {"fileset": "fs-b"}, "target": "/mnt/agent"}]), "demo-1: mounts:", id="mounts"
            ),
            pytest.param(
                BuildSpec(source=ImageSource(image="docker.io/library/ubuntu:24.04")),
                "demo-1: source.image:",
                id="copy",
            ),
            pytest.param(
                _spec(platform="linux/arm64"), "demo-1: platform:.*linux/amd64, not linux/arm64", id="platform"
            ),
        ],
    )
    def test_a_request_no_such_service_builds(self, spec: BuildSpec, named: str) -> None:
        with pytest.raises(BackendRejectedError, match=named):
            RemoteBackend(_config(), PROFILES).check(_plan(spec))

    def test_more_images_than_the_service_takes_in_one_set(self) -> None:
        with pytest.raises(BackendRejectedError, match="build_specs: 2 images"):
            RemoteBackend(_config(), PROFILES).check(_plan(_spec(name="a"), _spec(name="b")))
        RemoteBackend(_config(max_specs_per_set=2), PROFILES).check(_plan(_spec(name="a"), _spec(name="b")))

    def test_a_dockerfile_off_the_root_is_taken_unless_the_deployment_says_otherwise(self) -> None:
        spec = _spec(dockerfile="services/feed/Dockerfile")
        RemoteBackend(_config(), PROFILES).check(_plan(spec))
        with pytest.raises(BackendRejectedError, match="demo-1: dockerfile:"):
            RemoteBackend(_config(allow_dockerfile_path=False), PROFILES).check(_plan(spec))

    def test_the_root_dockerfile_is_the_root_dockerfile_however_it_is_spelt(self) -> None:
        backend = RemoteBackend(_config(allow_dockerfile_path=False), PROFILES)
        backend.check(_plan(_spec(dockerfile="./Dockerfile")))

    def test_what_the_models_declare_is_what_the_cases_above_use(self) -> None:
        # Guards the parametrized specs: each really is the shape it is named for.
        assert _spec(output={"repository": "team/app", "tag": "v1"}).output == BuildOutput(
            repository="team/app", tag="v1"
        )
        assert _spec(mounts=[{"source": {"fileset": "fs-b"}, "target": "/mnt/agent"}]).mounts == [
            BuildMount(source=FileSetSource(fileset="fs-b"), target="/mnt/agent")
        ]


class TestADeploymentThatCannotBuild:
    """Rows are written before the job is created, so a deployment whose step couldn't run refuses every submit."""

    def test_nowhere_to_publish(self) -> None:
        with pytest.raises(BackendRejectedError, match="builder.remote.registry and builder.remote.repository"):
            RemoteBackend(_config(repository=None), PROFILES).check(_plan())

    def test_no_profile_for_the_step(self) -> None:
        with pytest.raises(BackendRejectedError, match="no execution profile cpu/build-remote"):
            RemoteBackend(_config(), [_profile("build-fetch")]).check(_plan())

    def test_no_image_that_holds_the_step(self) -> None:
        with pytest.raises(BackendRejectedError, match="no image that holds the remote-build step"):
            RemoteBackend(_config(image=None), PROFILES).check(_plan())

    def test_the_profiles_own_image_will_do(self) -> None:
        profiles: list[ExecutionProfile] = [_profile(default_task_image=STEP_IMAGE)]
        RemoteBackend(_config(image=None), profiles).check(_plan())

    def test_a_profile_that_ends_a_step_before_its_builds_could_finish(self) -> None:
        """Jobs counts the limit from the job's creation, so it is the profile's to set, and must be long enough."""
        config = _config(build_timeout_seconds=900, max_specs_per_set=2)
        enough = 900 * 2 + CANCEL_MARGIN_SECONDS
        RemoteBackend(config, [_profile(ttl_seconds_active=enough)]).check(_plan())
        with pytest.raises(BackendRejectedError, match=f"after {enough - 1}s.*less than the {enough}s"):
            RemoteBackend(config, [_profile(ttl_seconds_active=enough - 1)]).check(_plan())

    def test_a_default_platform_the_service_does_not_build(self) -> None:
        """Named as the deployment's problem, not each task's: a request that names no platform gets the default."""
        with pytest.raises(BackendRejectedError, match="builder.default_platform is linux/arm64.*builder.remote"):
            RemoteBackend(_config(default_platform="linux/arm64"), PROFILES).check(_plan())
        RemoteBackend(_config(default_platform="linux/arm64", platforms=["linux/arm64"]), PROFILES)._check_deployment()

    def test_a_secret_jobs_could_not_fetch_for_the_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under workload token exchange the launcher has no identity to fetch a step's secrets with."""
        backend = RemoteBackend(_config(credential_secret="system/remote-credential"), PROFILES)
        monkeypatch.setattr(remote, "is_workload_identity_token_exchange_enabled", lambda: True)
        with pytest.raises(BackendRejectedError, match="credential_secret is set.*workload token exchange"):
            backend.check(_plan())
        # Mounted by the profile instead, there is nothing to fetch.
        RemoteBackend(_config(), PROFILES).check(_plan())

    def test_auth_config_that_cant_be_read_refuses_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def unreadable() -> bool:
            raise WorkloadIdentityConfigError("no auth config")

        monkeypatch.setattr(remote, "is_workload_identity_token_exchange_enabled", unreadable)
        RemoteBackend(_config(credential_secret="system/remote-credential"), PROFILES).check(_plan())


class TestDestination:
    def test_every_workspace_gets_the_services_one_repository(self) -> None:
        backend = RemoteBackend(_config(), PROFILES)
        placed = [
            backend.destination(_plan(workspace=workspace).images[0]) for workspace in ("default", "another-team")
        ]
        assert {(p.registry, p.repository) for p in placed} == {("registry.example.com", "team/images")}

    def test_a_workspace_whose_name_no_tag_could_hold_still_builds(self) -> None:
        """No tag is pushed, so nothing needs the workspace's name to fit in one."""
        backend = RemoteBackend(_config(), PROFILES)
        placed = backend.destination(_plan(workspace="alice@example.com").images[0])
        assert (placed.repository, placed.system_tag) == ("team/images", "demo-1")

    def test_a_spec_with_no_output_passes_the_generic_placement(self) -> None:
        # `with_destinations` refuses two images that publish one tag. With no output there is no tag to clash.
        backend = RemoteBackend(_config(max_specs_per_set=2), PROFILES)
        placed = _plan(_spec(name="a"), _spec(name="b")).with_destinations(backend.destination)
        assert [image.caller_ref for image in placed.images] == [None, None]


class TestTheJob:
    def _compile(self, config: BuilderConfig, *specs: BuildSpec) -> HelixJobSpec:
        backend = RemoteBackend(config, PROFILES)
        return backend.compile(_plan(*specs).with_destinations(backend.destination))

    def test_it_is_one_step_on_the_deployments_profile_image_and_command(self) -> None:
        spec = self._compile(_config(command=["my-step", "--verbose"]))
        assert [step.name for step in spec.steps] == [STEP_NAME]
        executor = spec.steps[0].executor
        assert isinstance(executor, CPUExecutionProvider)
        assert executor.profile == "build-remote"
        assert executor.container.image == STEP_IMAGE
        assert executor.container.command == ["my-step", "--verbose"]

    def test_the_step_is_told_everything_it_cannot_read_from_platform_config(self) -> None:
        spec = self._compile(
            _config(build_timeout_seconds=900),
            _spec(source={"fileset": "fs-a", "context_path": "tests"}, dockerfile="docker/Dockerfile"),
        )
        config = RemoteBuildStepConfig.model_validate(spec.steps[0].config)
        assert config.destination == "registry.example.com/team/images"
        assert config.build_timeout_seconds == 900
        assert config.credential_in_environment is False
        (image,) = config.images
        assert (image.image, image.dockerfile, image.platform) == ("demo-1", "docker/Dockerfile", "linux/amd64")
        # Qualified with the submitting workspace, as the fetch helpers need it.
        assert (image.source.fileset, image.source.context_path) == ("default/fs-a", "tests")

    def test_the_step_has_no_time_limit_of_its_own(self) -> None:
        """Jobs counts a step's limit from the job's creation: one sized to the build would end every build that
        had waited for a slot. The profile's limit applies, and the step enforces each image's timeout itself."""
        spec = self._compile(_config(build_timeout_seconds=900))
        assert spec.steps[0].lifecycle.ttl_seconds_active == 0
        assert "lifecycle" not in spec.model_dump(mode="json", exclude_unset=True)["steps"][0]

    def test_no_credential_is_declared_when_the_profile_mounts_it(self) -> None:
        spec = self._compile(_config())
        assert spec.steps[0].environment is None
        assert referenced_secrets(spec, "default") == []

    @pytest.mark.parametrize(
        ("secret", "read_from"),
        [
            ("remote-credential", ("default", "remote-credential")),
            ("system/remote-credential", ("system", "remote-credential")),
        ],
    )
    def test_a_platform_secret_reaches_the_step_and_the_submit_check(
        self, secret: str, read_from: tuple[str, str]
    ) -> None:
        spec = self._compile(_config(credential_secret=secret))
        (variable,) = spec.steps[0].environment or []
        assert variable.name == REMOTE_CREDENTIAL_ENV
        assert variable.from_secret is not None and variable.from_secret.name == secret
        assert referenced_secrets(spec, "default") == [read_from]
        assert RemoteBuildStepConfig.model_validate(spec.steps[0].config).credential_in_environment is True

    def test_the_spec_survives_the_jobs_clients_serialization(self) -> None:
        # The Jobs client sends `exclude_unset`, which drops whatever a model only defaulted.
        sent = self._compile(_config()).model_dump(mode="json", exclude_unset=True)
        received = HelixJobSpec.model_validate(sent)
        executor = received.steps[0].executor
        assert isinstance(executor, CPUExecutionProvider)
        assert (executor.profile, executor.container.image, executor.container.command) == (
            "build-remote",
            STEP_IMAGE,
            ["remote-build"],
        )
        assert RemoteBuildStepConfig.model_validate(received.steps[0].config).build_timeout_seconds == 3600

    def test_an_unset_image_is_left_for_jobs_to_resolve_from_the_profile(self) -> None:
        backend = RemoteBackend(_config(image=None), [_profile(default_task_image=STEP_IMAGE)])
        spec = backend.compile(_plan().with_destinations(backend.destination))
        sent = spec.model_dump(mode="json", exclude_unset=True)
        assert "image" not in sent["steps"][0]["executor"]["container"]


class FakeEntityClient:
    def __init__(self) -> None:
        self.rows: dict[str, ContainerImage] = {}
        self.events: list[str] = []

    async def create(self, entity: ContainerImage) -> ContainerImage:
        if entity.name in self.rows:
            raise EntityConflictError(f"{entity.name} exists")
        self.events.append(f"create:{entity.name}")
        self.rows[entity.name] = entity
        return entity

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage:
        if name not in self.rows:
            raise EntityNotFoundError(f"{name} not found")
        return self.rows[name]

    async def update(self, entity: ContainerImage) -> ContainerImage:
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
            raise ConflictError(
                httpx.Response(409, json={"detail": "exists"}, request=httpx.Request("POST", "http://j"))
            )
        self.events.append("create_job")
        self.jobs[request.name] = request
        return request

    async def get_job_fields(self, name: str) -> Mapping[str, Any]:
        return self.jobs[name].custom_fields or {}


async def _submit(
    entity_client: FakeEntityClient,
    jobs: FakeJobs,
    backend: Backend,
    *,
    unreadable: frozenset[str] = frozenset(),
) -> SubmitResult:
    async def secret_readable(workspace: str, name: str) -> bool:
        return f"{workspace}/{name}" not in unreadable

    async def resolve_upstream(reference: ImageReference, platform: str) -> str:
        raise AssertionError("the remote backend refuses copies before anything is resolved")

    return await submit_build_set(
        BuildSet(name="demo", revision=1, build_specs=[_spec()]),
        backend=backend,
        workspace="default",
        entity_client=entity_client,
        create_job=jobs.create_job,
        get_job_fields=jobs.get_job_fields,
        secret_readable=secret_readable,
        resolve_upstream=resolve_upstream,
        default_platform="linux/amd64",
    )


class TestSubmitting:
    """The generic submit path, unchanged, with the remote backend behind it."""

    async def test_the_row_is_written_at_the_services_repository_then_the_job_is_created(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        result = await _submit(client, jobs, RemoteBackend(_config(), PROFILES))
        assert client.events == ["create:demo-1", "create_job"]
        (row,) = result.images
        assert (row.status, row.registry, row.repository) == ("pending", "registry.example.com", "team/images")
        assert row.provenance.backend == "remote"
        assert row.external is None
        job = jobs.jobs["builder-demo-1"]
        assert job.platform_spec is not None
        assert [step.name for step in job.platform_spec.steps] == [STEP_NAME]

    async def test_the_same_request_again_adopts_the_row_and_the_job(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        first = await _submit(client, jobs, backend)
        again = await _submit(client, jobs, backend)
        assert [image.name for image in again.images] == [image.name for image in first.images]
        assert client.events.count("create_job") == 1

    async def test_a_row_another_backend_built_is_not_adopted(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        in_cluster_profiles: list[ExecutionProfile] = [
            _profile(name, storage={"pvc_name": "work"}) for name in ("build-fetch", "build-control", "build-push")
        ]
        in_cluster = BuilderConfig.model_validate(
            {"registry": "reg.example.com", "sandbox": {"image": "kaniko.example.com/executor:debug"}}
        )
        await _submit(client, jobs, ExecutionBackend(in_cluster, in_cluster_profiles))
        assert client.rows["demo-1"].provenance.backend == "execution"

        with pytest.raises(BackendChanged, match="already built by the 'execution' backend.*builds with 'remote'"):
            await _submit(client, jobs, RemoteBackend(_config(), PROFILES))
        # A revision conflict to a caller that doesn't know the difference, as before.
        assert issubclass(BackendChanged, BuildConflict)
        # The row the other backend built is left as it was.
        assert client.rows["demo-1"].registry == "reg.example.com"

    async def test_a_credential_the_submitter_cannot_read_is_refused_before_anything_is_written(self) -> None:
        client = FakeEntityClient()
        backend = RemoteBackend(_config(credential_secret="remote-credential"), PROFILES)
        with pytest.raises(MissingSecrets, match="the build job needs.*default/remote-credential$"):
            await _submit(client, FakeJobs(client.events), backend, unreadable=frozenset({"default/remote-credential"}))
        assert client.events == []


class TestTheRecord:
    def test_a_row_from_before_the_field_existed_was_built_in_cluster(self) -> None:
        provenance = Provenance.model_validate(
            {
                "build_set": "demo",
                "revision": 1,
                "job": "default/builder-demo-1",
                "request_digest": "sha256:" + "a" * 64,
            }
        )
        assert provenance.backend == "execution"

    def test_complete_takes_the_services_handle(self) -> None:
        request = CompleteRequest.model_validate(
            {"digest": "sha256:" + "b" * 64, "external": {"image_id": "img-1", "build_id": "build-1"}}
        )
        assert request.external == ExternalImage(image_id="img-1", build_id="build-1")
        with pytest.raises(ValidationError):
            CompleteRequest.model_validate({"digest": "sha256:" + "b" * 64, "signature": "x"})

    def test_external_holds_a_few_short_facts(self) -> None:
        with pytest.raises(ValidationError, match="too long"):
            ExternalImage(attributes={"log": "x" * 513})
        with pytest.raises(ValidationError):
            ExternalImage(attributes={f"k{i}": "v" for i in range(33)})
        with pytest.raises(ValidationError):
            ExternalImage(image_id="")

    def test_a_misspelt_handle_is_refused_not_dropped(self) -> None:
        """Dropped, the row would become ready with no handle, and nothing would say so."""
        with pytest.raises(ValidationError, match="unknown fields: imageid"):
            CompleteRequest.model_validate({"digest": "sha256:" + "b" * 64, "external": {"imageid": "img-1"}})
        # What is stored stays lenient, so a row a later builder wrote still reads.
        assert ExternalImage.model_validate({"image_id": "img-1", "later": "field"}).image_id == "img-1"

    async def test_an_image_this_cluster_builds_takes_no_handle(self) -> None:
        client = FakeEntityClient()
        in_cluster_profiles: list[ExecutionProfile] = [
            _profile(name, storage={"pvc_name": "work"}) for name in ("build-fetch", "build-control", "build-push")
        ]
        in_cluster = BuilderConfig.model_validate(
            {"registry": "reg.example.com", "sandbox": {"image": "kaniko.example.com/executor:debug"}}
        )
        await _submit(client, FakeJobs(client.events), ExecutionBackend(in_cluster, in_cluster_profiles))
        with pytest.raises(CompletionConflict, match="built in this cluster"):
            await complete(
                client,
                workspace="default",
                name="demo-1",
                digest="sha256:" + "c" * 64,
                external=ExternalImage(image_id="img-1"),
            )
        assert client.rows["demo-1"].status == "pending"

    async def test_completing_writes_the_handle_with_the_digest(self) -> None:
        client = FakeEntityClient()
        await _submit(client, FakeJobs(client.events), RemoteBackend(_config(), PROFILES))
        digest = "sha256:" + "c" * 64
        external = ExternalImage(image_id="img-1", build_id="build-1", attributes={"builder": "v1"})

        ready = await complete(client, workspace="default", name="demo-1", digest=digest, external=external)

        assert (ready.status, ready.external) == ("ready", external)
        assert ready.image_ref == f"registry.example.com/team/images@{digest}"
        # What the entity store keeps round-trips, handle included.
        stored = ready.model_dump(mode="json", exclude_computed_fields=True)
        assert ContainerImage.model_validate(stored).external == external
