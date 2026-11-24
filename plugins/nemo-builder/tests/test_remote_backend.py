# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote backend: what it refuses at submit, where it publishes, and the one step it compiles."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.backend import BackendRejectedError
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.identity import ImageReference, compose_system_tag
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.remote import RemoteBackend
from nemo_builder_plugin.schema import BuildMount, BuildOutput, BuildSet, BuildSpec, FileSetSource, ImageSource
from nemo_builder_plugin.steps import REMOTE_CREDENTIAL_ENV, RemoteBuildStepConfig
from nemo_builder_plugin.submit import MissingSecrets, submit_build_set
from nemo_helix_plugin.jobs.execution_profiles import KubernetesJobExecutionProfile
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider
from test_submit import FakeEntityClient, FakeJobs

PROFILES = [KubernetesJobExecutionProfile.model_validate({"profile": "build-remote", "config": {}})]
REMOTE = {
    "registry": "reg.example.com",
    "repository": "builds/images",
    "profile": "build-remote",
    "image": "client.example.com/remote-build:1",
    "command": ["remote-build"],
    "credential_secret": "build-service-token",
    "settings": {"endpoint": "https://builds.example.com"},
}


def _config(**remote: object) -> BuilderConfig:
    return BuilderConfig.model_validate({"backend": "remote", "remote": REMOTE | remote})


def _spec(name: str = "app", **fields: object) -> BuildSpec:
    return BuildSpec.model_validate({"name": name, "source": FileSetSource(fileset="ctx", context_path="app")} | fields)


def _plan(*specs: BuildSpec, backend: RemoteBackend | None = None) -> BuildPlan:
    build_set = BuildSet(name="demo", revision=1, build_specs=list(specs or [_spec()]))
    plan = BuildPlan.resolve(build_set.with_default_platform("linux/amd64"), workspace="default")
    return plan.with_destinations(backend.destination) if backend else plan


def test_the_configured_backend_is_loaded() -> None:
    assert isinstance(load_backend(_config(), PROFILES), RemoteBackend)
    assert isinstance(load_backend(BuilderConfig(), PROFILES), ExecutionBackend)


class TestCheck:
    def test_a_plain_build_passes(self) -> None:
        RemoteBackend(_config(), PROFILES).check(_plan(_spec(dockerfile="./Dockerfile")))

    def test_a_deployment_without_the_section_or_its_profile_is_refused(self) -> None:
        with pytest.raises(BackendRejectedError, match="builder.remote is not configured"):
            RemoteBackend(BuilderConfig(backend="remote"), PROFILES).check(_plan())
        with pytest.raises(BackendRejectedError, match="cpu/elsewhere"):
            RemoteBackend(_config(profile="elsewhere"), PROFILES).check(_plan())

    @pytest.mark.parametrize(
        ("spec", "field"),
        [
            (BuildSpec(name="copy", source=ImageSource(image="docker.io/library/alpine:3")), "source.image"),
            (_spec(mounts=[BuildMount(source=FileSetSource(fileset="tools"), target="/tools")]), "mounts"),
            (_spec(output=BuildOutput(repository="team/app", tag="v1")), "output"),
            (_spec(platform="linux/arm64"), "platform"),
            (_spec(dockerfile="docker/Dockerfile"), "dockerfile"),
        ],
    )
    def test_what_the_service_cannot_build_is_refused(self, spec: BuildSpec, field: str) -> None:
        with pytest.raises(BackendRejectedError, match=f": {field}:"):
            RemoteBackend(_config(), PROFILES).check(_plan(spec))


class TestCompile:
    def test_one_step_on_the_profile_with_every_image(self) -> None:
        backend = RemoteBackend(_config(), PROFILES)
        plan = _plan(_spec("a"), _spec("b"), backend=backend)
        [step] = backend.compile(plan).steps
        assert isinstance(step.executor, CPUExecutionProvider)
        assert step.executor.profile == "build-remote"
        assert (step.executor.container.image, step.executor.container.command) == (
            "client.example.com/remote-build:1",
            ["remote-build"],
        )
        config = RemoteBuildStepConfig.model_validate(step.config)
        assert config.settings == {"endpoint": "https://builds.example.com"}
        assert config.build_timeout_seconds == 1800
        image = config.images[0]
        assert (image.name, image.platform, image.registry, image.repository, image.tag) == (
            "demo-1.a",
            "linux/amd64",
            "reg.example.com",
            "builds/images/default/demo/a",
            compose_system_tag("default", "demo-1.a"),
        )
        assert (image.source.fileset, image.source.context_path) == ("default/ctx", "app")
        [variable] = step.environment or []
        assert variable.name == REMOTE_CREDENTIAL_ENV and variable.value is None
        assert variable.from_secret is not None and variable.from_secret.name == "build-service-token"
        assert RemoteBuildStepConfig.model_validate(step.config).uses_credential

    def test_a_service_that_publishes_to_one_repository_shares_it_across_workspaces(self) -> None:
        backend = RemoteBackend(_config(per_workspace=False), PROFILES)
        [image] = RemoteBuildStepConfig.model_validate(backend.compile(_plan(backend=backend)).steps[0].config).images
        assert (image.repository, image.tag) == ("builds/images", compose_system_tag("default", "demo-1.app"))

    def test_no_credential_reaches_a_step_that_uses_its_own_identity(self) -> None:
        backend = RemoteBackend(_config(credential_secret=None), PROFILES)
        [step] = backend.compile(_plan(backend=backend)).steps
        assert not step.environment
        assert not RemoteBuildStepConfig.model_validate(step.config).uses_credential


async def _submit(client: FakeEntityClient, unreadable: frozenset[str] = frozenset()) -> None:
    async def secret_readable(workspace: str, name: str) -> bool:
        return f"{workspace}/{name}" not in unreadable

    async def resolve_upstream(reference: ImageReference, platform: str) -> str:
        raise AssertionError("nothing is copied")

    jobs = FakeJobs(client.events)
    await submit_build_set(
        BuildSet(name="demo", revision=1, build_specs=[_spec()]),
        backend=RemoteBackend(_config(), PROFILES),
        workspace="default",
        entity_client=client,
        create_job=jobs.create_job,
        get_job_fields=jobs.get_job_fields,
        secret_readable=secret_readable,
        resolve_upstream=resolve_upstream,
        default_platform="linux/amd64",
    )


class TestSubmit:
    @pytest.mark.asyncio
    async def test_rows_are_written_pending_in_the_services_repository(self) -> None:
        client = FakeEntityClient()
        await _submit(client)
        assert client.events == ["create:demo-1.app", "create_job"]
        row = client.rows["demo-1.app"]
        assert (row.status, row.registry, row.repository) == (
            "pending",
            "reg.example.com",
            "builds/images/default/demo/app",
        )

    @pytest.mark.asyncio
    async def test_a_caller_who_cannot_read_the_credential_is_refused(self) -> None:
        client = FakeEntityClient()
        with pytest.raises(MissingSecrets, match="default/build-service-token$"):
            await _submit(client, unreadable=frozenset({"default/build-service-token"}))
        assert client.events == []
