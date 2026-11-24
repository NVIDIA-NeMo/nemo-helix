# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The remote backend: how it is selected, what it refuses, where it places an image, the job it compiles, and the
submit path with it."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from nemo_builder_plugin import remote, submit
from nemo_builder_plugin.backend import Backend, BackendRejectedError
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.completion import CompleteRequest, CompletionConflict, complete
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.entities import ContainerImage, ExternalImage, ImageRuntimeConfig, Provenance
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.identity import ImageReference
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.remote import CANCEL_MARGIN_SECONDS, STEP_BACKENDS, STEP_NAME, RemoteBackend
from nemo_builder_plugin.schema import BuildSet, BuildSpec, ImageSource
from nemo_builder_plugin.steps import REMOTE_CREDENTIAL_ENV, RemoteBuildStepConfig
from nemo_builder_plugin.submit import (
    BackendChanged,
    BuildConflict,
    CreatedJob,
    ExistingJob,
    InvalidBuildRequest,
    MissingSecrets,
    SubmitResult,
    referenced_secrets,
    request_digest,
    submit_build_set,
)
from nemo_helix_plugin.auth import workload_identity
from nemo_helix_plugin.auth.workload_identity import WorkloadIdentityConfigError
from nemo_helix_plugin.client.errors import ConflictError, NotFoundError
from nemo_helix_plugin.entities import EntityConflictError, EntityNotFoundError
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    KubernetesJobExecutionProfile,
    OpenShellJobExecutionProfile,
    SubprocessJobExecutionProfile,
)
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
        # Nothing the builder can't verify is taken unless the operator says so.
        assert defaults.allow_unverified_completion is False


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


class TestADeploymentThatCannotBuild:
    """Rows are written before the job is created, so a deployment whose step couldn't run refuses every submit."""

    def test_nowhere_to_publish(self) -> None:
        with pytest.raises(BackendRejectedError, match="builder.remote.registry and builder.remote.repository"):
            RemoteBackend(_config(repository=None), PROFILES).check(_plan())

    def test_no_profile_for_the_step(self) -> None:
        with pytest.raises(BackendRejectedError, match="no execution profile cpu/build-remote"):
            RemoteBackend(_config(), [_profile("build-fetch")]).check(_plan())

    @pytest.mark.parametrize(
        "kind", [OpenShellJobExecutionProfile, SubprocessJobExecutionProfile, DockerJobExecutionProfile]
    )
    def test_a_profile_on_a_backend_that_cant_run_the_step(self, kind: type[ExecutionProfile]) -> None:
        """One that runs no container, gives a step no job ID, can't put a secret in its environment, or names
        its container without the workspace. The step would fail, or collide, after the rows were written."""
        profile = kind.model_validate({"provider": "cpu", "profile": "build-remote", "config": {}})
        with pytest.raises(BackendRejectedError, match=f"runs on {profile.backend}, which can't run the remote-build"):
            RemoteBackend(_config(), [profile]).check(_plan())

    def test_a_step_image_is_kept_without_surrounding_spaces_and_a_blank_one_is_refused(self) -> None:
        """A templated value, say. The job would be created all the same, and its image never pulled."""
        assert _config(image="  steps.example.com/my-step:1 ").remote.image == "steps.example.com/my-step:1"
        with pytest.raises(ValueError, match="remote.image is blank"):
            _config(image="   ")
        with pytest.raises(ValueError, match="remote.command has a blank entry"):
            _config(command=["remote-build", " "])

    def test_a_padded_credential_secret_is_kept_without_the_padding(self) -> None:
        """Jobs trims the reference before it reads it. The builder's own check of the secret splits it as it
        is, and would look for the secret in a workspace named with a space."""
        assert _config(credential_secret=" platform/remote-credential ").remote.credential_secret == (
            "platform/remote-credential"
        )

    def test_the_profiles_limit_is_checked_for_as_many_images_as_a_request_can_name(self) -> None:
        """A deployment may allow more images in a set than any request can hold. No set is that large."""
        config = _config(build_timeout_seconds=60, max_specs_per_set=1000)
        enough = 60 * 100 + CANCEL_MARGIN_SECONDS
        RemoteBackend(config, [_profile(ttl_seconds_active=enough)]).check(_plan())
        # And for no fewer: a second less is refused, with the number of images it was checked for.
        with pytest.raises(BackendRejectedError, match=f"less than the {enough}s .* times 100, the images one set"):
            RemoteBackend(config, [_profile(ttl_seconds_active=enough - 1)]).check(_plan())

    def test_kubernetes_is_the_one_backend_the_step_runs_on(self) -> None:
        """An allow-list: a backend Jobs gains later is refused until it is known to run the step."""
        assert STEP_BACKENDS == (_profile().backend,) == ("kubernetes_job",)
        new = _profile().model_copy(update={"backend": "a_new_backend"})
        with pytest.raises(BackendRejectedError, match="runs on a_new_backend, which can't run the remote-build"):
            RemoteBackend(_config(), [new]).check(_plan())

    def test_the_profile_looked_up_is_the_one_the_deployment_names(self) -> None:
        config = _config(profile="builds-elsewhere")
        RemoteBackend(config, [_profile("builds-elsewhere")]).check(_plan())
        with pytest.raises(BackendRejectedError, match="no execution profile cpu/builds-elsewhere"):
            RemoteBackend(config, PROFILES).check(_plan())

    @pytest.mark.parametrize("reference", ["/remote-credential", "team/", "team/build/credential", " "])
    def test_a_credential_secret_that_is_no_reference_is_a_configuration_error(self, reference: str) -> None:
        """Jobs reads it as `<name>` or `<workspace>/<name>` when it creates the job, after the rows are
        written. Found here, it refuses the submit and writes none."""
        with pytest.raises(ValidationError, match="remote.credential_secret must be `<name>` or `<workspace>/<name>`"):
            _config(credential_secret=reference)

    @pytest.mark.parametrize("reference", ["remote-credential", "platform/remote-credential", "team+ml/credential"])
    def test_a_credential_secret_in_either_form_is_taken(self, reference: str) -> None:
        assert _config(credential_secret=reference).remote.credential_secret == reference

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

    def test_a_credential_secret_is_taken_under_workload_token_exchange(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Jobs' launcher fetches a step's secrets there too, with a token for the job's creator, so the backend
        doesn't refuse a credential for how the platform authenticates a step."""
        # The platform uses exchange, wherever that is asked: where the platform answers it, and under the name
        # the backend reads the answer by. It once refused a credential on it. With platform auth on or off.
        monkeypatch.setattr(workload_identity, "is_workload_identity_token_exchange_enabled", lambda: True)
        monkeypatch.setattr(remote, "is_workload_identity_token_exchange_enabled", lambda: True)
        monkeypatch.setattr(remote, "platform_auth_enabled", lambda: True)
        backend = RemoteBackend(_config(credential_secret="system/remote-credential"), PROFILES)
        plan = _plan()
        backend.check(plan)
        # And the step is still given the secret to read, as it is without exchange.
        assert referenced_secrets(backend.compile(plan), "default") == [("system", "remote-credential")]


def _cant_be_read() -> bool:
    raise WorkloadIdentityConfigError("Could not resolve auth config for workload identity token exchange")


def _platform(monkeypatch: pytest.MonkeyPatch, *, auth: bool, token_exchange: Callable[[], bool]) -> None:
    """How the platform authenticates a caller, as the backend asks it."""
    monkeypatch.setattr(remote, "platform_auth_enabled", lambda: auth)
    monkeypatch.setattr(remote, "is_workload_identity_token_exchange_enabled", token_exchange)


class TestAPlatformThatCannotShowWhoCompletesAnImage:
    """With platform auth on and workload token exchange off, a request's identity is whatever its headers say.
    A completion would be refused there, so the submit is: no build is made that could not be recorded."""

    async def test_every_submit_is_refused_and_nothing_is_written(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _platform(monkeypatch, auth=True, token_exchange=lambda: False)
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        with pytest.raises(BackendRejectedError) as refused:
            await _submit(client, jobs, RemoteBackend(_config(), PROFILES))
        # The sentence names the setting, and what turning it on accepts.
        assert "platform auth is on without workload token exchange" in str(refused.value)
        assert "builder.remote.allow_unverified_completion, which accepts that anything able to reach" in str(
            refused.value
        )
        assert client.events == [] and jobs.jobs == {}

    def test_the_operator_can_accept_completions_nothing_verifies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _platform(monkeypatch, auth=True, token_exchange=lambda: False)
        RemoteBackend(_config(allow_unverified_completion=True), PROFILES).check(_plan())

    @pytest.mark.parametrize(
        ("auth", "token_exchange"),
        [(False, lambda: False), (False, _cant_be_read), (True, lambda: True)],
        ids=["auth-off", "auth-off-and-no-readable-exchange-setting", "token-exchange"],
    )
    def test_a_platform_that_checks_no_caller_or_verifies_each_builds(
        self, auth: bool, token_exchange: Callable[[], bool], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With auth off no caller is checked at all, as the README says of that mode, and how workloads
        would be authenticated is not asked. With token exchange a step's token says which job it is of."""
        _platform(monkeypatch, auth=auth, token_exchange=token_exchange)
        RemoteBackend(_config(), PROFILES).check(_plan())

    @pytest.mark.parametrize("unverified", [False, True], ids=["by-default", "though-the-operator-accepts-unverified"])
    def test_a_platform_whose_auth_settings_cant_be_read_is_refused(
        self, unverified: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing then says a completion could be verified, and the submit doesn't end as a server error.
        Whatever the operator accepted: the completion route reads how the platform authenticates a workload
        before it reads that setting, and admits no one where it can't, so the build's step would be put off
        until it gave up. What the operator accepts is a platform known to verify no one."""
        _platform(monkeypatch, auth=True, token_exchange=_cant_be_read)
        with pytest.raises(BackendRejectedError, match="the platform's auth configuration can't be read"):
            RemoteBackend(_config(allow_unverified_completion=unverified), PROFILES).check(_plan())

    def test_the_setting_changes_nothing_where_a_step_is_verified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _platform(monkeypatch, auth=True, token_exchange=lambda: True)
        RemoteBackend(_config(allow_unverified_completion=True), PROFILES).check(_plan())

    def test_it_is_the_last_thing_a_deployment_is_refused_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A deployment with nowhere to publish is told that, in any mode."""
        _platform(monkeypatch, auth=True, token_exchange=lambda: False)
        with pytest.raises(BackendRejectedError, match="builder.remote.registry and builder.remote.repository"):
            RemoteBackend(_config(repository=None), PROFILES).check(_plan())


class TestWhereTheStepMayRun:
    """Under workload token exchange a completion's caller is the build's step by the platform's record of
    its pod. That record is as sure as the step's namespace is closed, so the check holds as far as nothing
    but Jobs creates pods there."""

    @pytest.mark.parametrize("namespace", ["builds", None], ids=["named", "left-to-jobs"])
    def test_under_token_exchange_the_step_may_not_run_where_an_in_cluster_build_step_creates_pods(
        self, namespace: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The build step of an in-cluster build may create pods in its own namespace, so the remote-build
        step may not share it."""
        _platform(monkeypatch, auth=True, token_exchange=lambda: True)
        shared: list[ExecutionProfile] = [_profile(namespace=namespace), _profile("build-control", namespace=namespace)]
        with pytest.raises(BackendRejectedError) as refused:
            RemoteBackend(_config(), shared).check(_plan())
        assert "cpu/build-remote runs the remote-build step in the namespace of cpu/build-control" in str(refused.value)
        assert "Run the remote-build step in a namespace of its own, where only Jobs creates pods" in str(refused.value)

        # In a namespace of its own it builds, where both profiles name theirs, and so it does where Jobs lists
        # no such profile.
        apart: list[ExecutionProfile] = [_profile(namespace="remote-builds"), _profile("build-control")]
        RemoteBackend(_config(), apart).check(_plan())
        RemoteBackend(_config(), shared[:1]).check(_plan())
        # The profile looked for is the one the deployment names for that step.
        renamed = BuilderConfig.model_validate(_config().model_dump() | {"control_profile": "sandboxes"})
        RemoteBackend(renamed, shared).check(_plan())
        with pytest.raises(BackendRejectedError, match="in the namespace of cpu/sandboxes"):
            RemoteBackend(renamed, [shared[0], _profile("sandboxes", namespace=namespace)]).check(_plan())

    @pytest.mark.parametrize(
        ("step_namespace", "control_namespace", "unnamed"),
        [
            ("remote-builds", None, "build-control"),
            ("remote-builds", "", "build-control"),
            (None, "builds", "build-remote"),
            ("", "builds", "build-remote"),
        ],
        ids=[
            "the-control-profile-names-none",
            "the-control-profile-names-an-empty-one",
            "the-step-names-none",
            "the-step-names-an-empty-one",
        ],
    )
    def test_under_token_exchange_a_namespace_left_to_jobs_is_not_known_to_be_apart_from_a_named_one(
        self,
        step_namespace: str | None,
        control_namespace: str | None,
        unnamed: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Jobs runs the steps of a profile that names no namespace, or an empty one, in the namespace of its
        own controller, which the builder can't read. That may be the namespace the other profile names, so
        the two are not known to be apart, and the submit is refused."""
        _platform(monkeypatch, auth=True, token_exchange=lambda: True)
        profiles: list[ExecutionProfile] = [
            _profile(namespace=step_namespace),
            _profile("build-control", namespace=control_namespace),
        ]
        with pytest.raises(BackendRejectedError) as refused:
            RemoteBackend(_config(), profiles).check(_plan())
        # The sentence says which profile to name, and what to do.
        assert f"cpu/{unnamed} names no namespace" in str(refused.value)
        assert "runs apart from the steps of cpu/build-control" in str(refused.value)
        assert "Name a namespace on both profiles" in str(refused.value)
        # Named on both, and not the same, the step builds.
        named: list[ExecutionProfile] = [
            _profile(namespace=step_namespace or "remote-builds"),
            _profile("build-control", namespace=control_namespace or "builds"),
        ]
        RemoteBackend(_config(), named).check(_plan())

    def test_under_token_exchange_an_empty_namespace_and_none_are_the_same_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Jobs reads an empty namespace as none: both profiles' steps then run in its controller's own."""
        _platform(monkeypatch, auth=True, token_exchange=lambda: True)
        shared: list[ExecutionProfile] = [_profile(namespace=None), _profile("build-control", namespace="")]
        with pytest.raises(BackendRejectedError, match="in the namespace of cpu/build-control"):
            RemoteBackend(_config(), shared).check(_plan())

    @pytest.mark.parametrize(
        ("auth", "unverified"), [(False, False), (True, True)], ids=["auth-off", "the-operator-accepts-unverified"]
    )
    def test_where_no_step_is_verified_the_steps_namespace_is_not_asked(
        self, auth: bool, unverified: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing there rests on which pod is calling."""
        _platform(monkeypatch, auth=auth, token_exchange=lambda: False)
        shared: list[ExecutionProfile] = [_profile(), _profile("build-control")]
        RemoteBackend(_config(allow_unverified_completion=unverified), shared).check(_plan())


class TestDestination:
    def test_every_workspace_gets_the_services_one_repository(self) -> None:
        backend = RemoteBackend(_config(), PROFILES)
        placed = [
            backend.destination(_plan(workspace=workspace).images[0]) for workspace in ("default", "another-team")
        ]
        assert {(p.registry, p.repository) for p in placed} == {("registry.example.com", "team/images")}

    def test_a_workspace_whose_name_no_repository_could_hold_is_placed_all_the_same(self) -> None:
        """Nothing is published under the workspace's name, so it needn't fit in a repository path or a tag."""
        backend = RemoteBackend(_config(), PROFILES)
        placed = backend.destination(_plan(workspace="Team_A").images[0])
        assert (placed.repository, placed.system_tag) == ("team/images", "demo-1")

    def test_a_spec_with_no_output_passes_the_generic_placement(self) -> None:
        # `with_destinations` refuses two images that publish one tag. With no output there is no tag to clash.
        backend = RemoteBackend(_config(max_specs_per_set=2), PROFILES)
        placed = _plan(_spec(name="a"), _spec(name="b")).with_destinations(backend.destination)
        assert [image.caller_ref for image in placed.images] == [None, None]
        # Both in the service's one repository, each under its own row's name.
        assert [(image.placed.repository, image.placed.system_tag) for image in placed.images] == [
            ("team/images", "demo-1.a"),
            ("team/images", "demo-1.b"),
        ]


class TestTheJob:
    def _compile(self, config: BuilderConfig, *specs: BuildSpec) -> HelixJobSpec:
        backend = RemoteBackend(config, PROFILES)
        return backend.compile(_plan(*specs).with_destinations(backend.destination))

    def test_it_is_one_step_on_the_deployments_profile_image_and_command(self) -> None:
        spec = self._compile(_config(profile="builds-elsewhere", command=["my-step", "--verbose"]))
        assert [step.name for step in spec.steps] == [STEP_NAME]
        executor = spec.steps[0].executor
        assert isinstance(executor, CPUExecutionProvider)
        assert executor.profile == "builds-elsewhere"
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
    """The entity store, as far as a submit and a completion meet it. A row is found by its name, and written
    by its ID on condition of the version it was read at: each create makes a row with an ID of its own."""

    def __init__(self) -> None:
        self.rows: dict[str, ContainerImage] = {}
        self.events: list[str] = []
        #: The names read, in order.
        self.reads: list[str] = []
        #: What happens elsewhere just before the next write is answered, or the nth read from now.
        self.before_write: Callable[[], object] | None = None
        self.before_read: dict[int, Callable[[], object]] = {}
        self._made = 0

    async def create(self, entity: ContainerImage) -> ContainerImage:
        if entity.name in self.rows:
            raise EntityConflictError(f"{entity.name} exists")
        self.events.append(f"create:{entity.name}")
        self._made += 1
        entity._id = f"container-image-{self._made}"
        self.rows[entity.name] = entity
        return entity

    async def get(self, entity_type: type[ContainerImage], *, name: str, workspace: str) -> ContainerImage:
        self.reads.append(name)
        if (happens := self.before_read.pop(len(self.reads), None)) is not None:
            happens()
        if name not in self.rows:
            raise EntityNotFoundError(f"{name} not found")
        return self.rows[name]

    async def update(self, entity: ContainerImage) -> ContainerImage:
        if self.before_write is not None:
            happens, self.before_write = self.before_write, None
            happens()
        stored = self.rows.get(entity.name)
        if stored is None or stored.id != entity.id:
            raise EntityNotFoundError(f"{entity.name} not found")
        if stored.db_version != entity.db_version:
            self.events.append(f"update_conflict:{entity.name}")
            raise EntityConflictError(f"{entity.name} changed")
        self.events.append(f"update:{entity.name}:{entity.status}")
        entity._db_version += 1
        self.rows[entity.name] = entity
        return entity

    def written_elsewhere(self, name: str, **changes: object) -> ContainerImage:
        """Any other write of the row: it is then at the next version, with ``changes``."""
        row = self.rows[name].model_copy(update=changes, deep=True)
        row._db_version += 1
        self.rows[name] = row
        return row

    def replaced(self, name: str) -> ContainerImage:
        """The row is deleted, and another is created under its name: a row with an ID of its own."""
        row = self.rows[name].model_copy(deep=True)
        self._made += 1
        row._id, row._db_version = f"container-image-{self._made}", 1
        self.rows[name] = row
        return row


class FakeJobs:
    """Jobs, as far as a submit meets it: a name holds one job at a time, and every create makes a new job,
    with an ID and a first attempt of its own, which it answers with."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.jobs: dict[str, CreateHelixJobRequest] = {}
        #: What Jobs answered the create of the job now under each name.
        self.answered: dict[str, CreatedJob] = {}
        self._made = 0

    def create(self, request: CreateHelixJobRequest) -> CreatedJob:
        """A create through Jobs' own API, by whoever may create a job in the workspace."""
        assert request.name is not None
        if request.name in self.jobs:
            raise ConflictError(
                httpx.Response(409, json={"detail": "exists"}, request=httpx.Request("POST", "http://j"))
            )
        self._made += 1
        self.jobs[request.name] = request
        self.answered[request.name] = CreatedJob(uid=f"platform-job-{self._made}", attempt_id=f"attempt-{self._made}")
        return self.answered[request.name]

    def delete(self, name: str) -> None:
        del self.jobs[name], self.answered[name]

    @classmethod
    def sharing(cls, other: FakeJobs) -> FakeJobs:
        """The same Jobs, as another request meets it: one that answers a create at once."""
        jobs = cls(other.events)
        jobs.jobs, jobs.answered = other.jobs, other.answered
        return jobs

    async def create_job(self, request: CreateHelixJobRequest) -> CreatedJob:
        """The create a submit makes."""
        created = self.create(request)
        self.events.append("create_job")
        return created

    async def get_job(self, name: str) -> ExistingJob:
        return ExistingJob(uid=self.answered[name].uid, custom_fields=self.jobs[name].custom_fields or {})


async def _submit(
    entity_client: FakeEntityClient,
    jobs: FakeJobs,
    backend: Backend,
    *,
    unreadable: frozenset[str] = frozenset(),
    workspace: str = "default",
    specs: Sequence[BuildSpec] = (),
) -> SubmitResult:
    async def secret_readable(workspace: str, name: str) -> bool:
        return f"{workspace}/{name}" not in unreadable

    async def resolve_upstream(reference: ImageReference, platform: str) -> str:
        raise AssertionError("the remote backend refuses copies before anything is resolved")

    return await submit_build_set(
        BuildSet(name="demo", revision=1, build_specs=list(specs) or [_spec()]),
        backend=backend,
        workspace=workspace,
        entity_client=entity_client,
        create_job=jobs.create_job,
        get_job=jobs.get_job,
        secret_readable=secret_readable,
        resolve_upstream=resolve_upstream,
        default_platform="linux/amd64",
    )


async def _submit_in_cluster(client: FakeEntityClient, jobs: FakeJobs | None = None) -> None:
    """Submit the set to a deployment that builds in this cluster."""
    profiles: list[ExecutionProfile] = [
        _profile(name, storage={"pvc_name": "work"}) for name in ("build-fetch", "build-control", "build-push")
    ]
    config = BuilderConfig.model_validate(
        {"registry": "reg.example.com", "sandbox": {"image": "kaniko.example.com/executor:debug"}}
    )
    await _submit(client, jobs or FakeJobs(client.events), ExecutionBackend(config, profiles))


class TestSubmitting:
    """The generic submit path, with the remote backend behind it."""

    async def test_a_workspace_whose_name_no_repository_could_hold_builds(self) -> None:
        client = FakeEntityClient()
        submitted = await _submit(
            client, FakeJobs(client.events), RemoteBackend(_config(), PROFILES), workspace="Team_A"
        )
        assert submitted.images[0].provenance.job == "Team_A/builder-demo-1"

    async def test_a_workspace_whose_name_a_row_cant_record_is_refused_before_anything_is_written(self) -> None:
        """The in-cluster backend refuses such a name sooner, as a repository path. Here nothing is published
        under it, so the row's reference to its job is the first thing that can't hold it."""
        client = FakeEntityClient()
        with pytest.raises(InvalidBuildRequest, match=r"in workspace 'alice@example.com' \(job: "):
            await _submit(
                client, FakeJobs(client.events), RemoteBackend(_config(), PROFILES), workspace="alice@example.com"
            )
        assert client.events == []

    async def test_the_row_is_written_at_the_services_repository_then_the_job_is_created(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        result = await _submit(client, jobs, RemoteBackend(_config(), PROFILES))
        # And the row is written once more, with the job that was created (below).
        assert client.events == ["create:demo-1", "create_job", "update:demo-1:pending"]
        (row,) = result.images
        assert (row.status, row.registry, row.repository) == ("pending", "registry.example.com", "team/images")
        assert row.provenance.backend == "remote"
        assert row.external is None
        job = jobs.jobs["builder-demo-1"]
        assert job.platform_spec is not None
        assert [step.name for step in job.platform_spec.steps] == [STEP_NAME]

    async def test_the_same_request_again_adopts_the_row_and_the_job(self) -> None:
        """The job under the build's name is the one the row records: the one the first submit created."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        first = await _submit(client, jobs, backend)
        events = list(client.events)
        again = await _submit(client, jobs, backend)
        assert [image.name for image in again.images] == [image.name for image in first.images]
        # Nothing is created, and nothing written: the row is returned as it is stored.
        assert client.events == events
        assert again.images[0].provenance.job_uid == jobs.answered["builder-demo-1"].uid

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

        with pytest.raises(BackendChanged, match="submitted under the 'execution' backend.*builds with 'remote'"):
            await _submit(client, jobs, RemoteBackend(_config(), PROFILES))
        # A revision conflict to a caller that doesn't know the difference, as before.
        assert issubclass(BackendChanged, BuildConflict)
        # The row the other backend built is left as it was: still waiting for its own build, and not written.
        row = client.rows["demo-1"]
        assert (row.registry, row.status, row.status_detail) == ("reg.example.com", "pending", None)
        assert [event for event in client.events if event.startswith("update:")] == []

    async def test_a_row_written_before_another_backends_row_is_met_says_why_it_was_not_built(self) -> None:
        """A set of two, built in the cluster. One row is later removed, and the set is asked for again from a
        deployment that builds remotely: the row is written anew, then the other backend's row is met."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        in_cluster_profiles: list[ExecutionProfile] = [
            _profile(name, storage={"pvc_name": "work"}) for name in ("build-fetch", "build-control", "build-push")
        ]
        in_cluster = BuilderConfig.model_validate(
            {"registry": "reg.example.com", "sandbox": {"image": "kaniko.example.com/executor:debug"}}
        )
        specs = [_spec(name="a"), _spec(name="b")]
        await _submit(client, jobs, ExecutionBackend(in_cluster, in_cluster_profiles), specs=specs)
        del client.rows["demo-1.a"]

        with pytest.raises(BackendChanged):
            await _submit(client, jobs, RemoteBackend(_config(max_specs_per_set=2), PROFILES), specs=specs)
        written, met = client.rows["demo-1.a"], client.rows["demo-1.b"]
        assert (written.status, written.status_detail) == (
            "failed",
            "not built: demo revision 1 was already submitted under another backend",
        )
        assert (met.status, met.status_detail, met.provenance.backend) == ("pending", None, "execution")
        # Written to the store, not only on the row the submit holds.
        assert "update:demo-1.a:failed" in client.events

    async def test_a_credential_the_submitter_cannot_read_is_refused_before_anything_is_written(self) -> None:
        client = FakeEntityClient()
        backend = RemoteBackend(_config(credential_secret="remote-credential"), PROFILES)
        with pytest.raises(MissingSecrets, match="the build job needs.*default/remote-credential$"):
            await _submit(client, FakeJobs(client.events), backend, unreadable=frozenset({"default/remote-credential"}))
        assert client.events == []


JOB = "builder-demo-1"
#: What a submit is answered when the job under its build's name is not one its rows record.
NOT_RECORDED = f"a job named {JOB!r} already exists, and this request's images do not record it as their build job"


def _recorded(row: ContainerImage) -> tuple[str | None, str | None]:
    """The job, and the attempt of it, that ``row`` says its submit created."""
    return row.provenance.job_uid, row.provenance.job_attempt_id


def _carrying_the_requests_digest(*specs: BuildSpec) -> CreateHelixJobRequest:
    """A job created through Jobs' own API under the name a submit of the request gives its job, with the
    request's digest in metadata its creator writes: anyone who knows the request can work that out."""
    build_set = BuildSet(name="demo", revision=1, build_specs=list(specs) or [_spec()])
    digest = request_digest(build_set.with_default_platform("linux/amd64"))
    return CreateHelixJobRequest.model_construct(
        name=JOB, source="my-own-tool", custom_fields={"request_digest": digest}
    )


class _Unavailable(FakeJobs):
    async def create_job(self, request: CreateHelixJobRequest) -> CreatedJob:
        raise RuntimeError("jobs unavailable")


class _SlowToAnswer(FakeJobs):
    """Jobs takes the name with its first write, and answers the create only once ``returns`` is set."""

    def __init__(self, events: list[str], returns: asyncio.Event) -> None:
        super().__init__(events)
        self.returns = returns

    async def create_job(self, request: CreateHelixJobRequest) -> CreatedJob:
        created = self.create(request)
        self.events.append("create_job")
        await self.returns.wait()
        return created


#: How long a submit run on its own is given to reach Jobs. It takes a few turns of the event loop: the wait
#: is this long only for a submit that never gets there, which then fails its test.
SUBMIT_REACHES_JOBS_SECONDS = 5.0


@asynccontextmanager
async def _submitting(
    client: FakeEntityClient, jobs: FakeJobs, backend: Backend
) -> AsyncGenerator[asyncio.Task[SubmitResult]]:
    """A submit run on its own, handed over once Jobs holds its job: with a Jobs that is slow to answer, it
    is by then waiting for the answer.

    A submit that ends before it creates a job never creates one. What it raised is raised here: a wait for
    the job alone would never end, and would hide why. A submit still running when the test leaves, as it
    is when an assertion fails first, is stopped.
    """
    submitting = asyncio.create_task(_submit(client, jobs, backend))
    try:
        async with asyncio.timeout(SUBMIT_REACHES_JOBS_SECONDS):
            while JOB not in jobs.jobs:
                if submitting.done():
                    submitting.result()
                    raise AssertionError("the submit answered without creating its job")
                await asyncio.sleep(0)
        yield submitting
    finally:
        submitting.cancel()
        # Waited for, so nothing of it runs on into another test, and an error of its own is not left unread.
        await asyncio.gather(submitting, return_exceptions=True)


class TestTheJobARowRecords:
    """A row a build service builds is completed on its caller's word, so it records which job its submit
    created: by the ID Jobs gave that job and its first attempt, which no other job, and no rerun, is given."""

    async def test_each_row_records_the_job_its_submit_created(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        result = await _submit(
            client,
            jobs,
            RemoteBackend(_config(max_specs_per_set=2), PROFILES),
            specs=[_spec(name="a"), _spec(name="b")],
        )
        created = jobs.answered[JOB]
        assert client.events == [
            "create:demo-1.a",
            "create:demo-1.b",
            "create_job",
            "update:demo-1.a:pending",
            "update:demo-1.b:pending",
        ]
        # What the submit answers with, and what is stored.
        for row in (*result.images, *client.rows.values()):
            assert _recorded(row) == (created.uid, created.attempt_id)
            assert row.provenance.job == f"default/{JOB}"
        stored = client.rows["demo-1.a"]._get_data_fields()["provenance"]
        assert (stored["job_uid"], stored["job_attempt_id"]) == (created.uid, created.attempt_id)

    async def test_a_job_put_under_the_builds_name_beforehand_is_not_the_builds_and_is_never_recorded(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Whoever may create a job in the workspace can make one under the name a build's job will have,
        carrying the request's digest. The submit does not take it for its own on that, as it once did: the
        job's creator wrote the digest. And nothing of a job found under a name is recorded on a row: only
        what Jobs answered a create this submit made.

        The row is left waiting, and records no job. Nothing tells this job from one that another submit of
        the request has just created and is about to record, so the row is not failed for it: no step that
        is verified completes a row that records no job, and the row fails once the job has ended."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        beforehand = jobs.create(_carrying_the_requests_digest())
        with caplog.at_level(logging.INFO), pytest.raises(BuildConflict, match=NOT_RECORDED) as refused:
            await _submit(client, jobs, RemoteBackend(_config(), PROFILES))
        # The answer says what to do: the same request once more, in case it is another submit's, then the next.
        assert "submit the request again in a moment. If the answer is still this, submit the next revision" in str(
            refused.value
        )
        row = client.rows["demo-1"]
        assert (row.status, row.status_detail, _recorded(row)) == ("pending", None, (None, None))
        assert beforehand.uid not in row.model_dump_json()
        assert "adopting it" not in caplog.text
        # The job is still the one made beforehand: the submit created none, and wrote nothing but its row.
        assert jobs.answered == {JOB: beforehand} and client.events == ["create:demo-1"]

    async def test_a_job_put_under_the_name_once_the_builds_own_is_deleted_is_not_adopted(self) -> None:
        """The row records the job its submit created. Another under that name, though it says it is of the
        same request, is another job, and the submit is answered a conflict.

        The row is left waiting, and still records its own job. Nothing tells the job found from one that
        another submit of the request has just created, and is about to record in place of the one that
        is gone, so the row is not failed for it: no step that is verified completes a row that records
        a job it is no step of, and the row fails once the job under the name has ended."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        await _submit(client, jobs, backend)
        own = jobs.answered[JOB]
        jobs.delete(JOB)
        later = jobs.create(_carrying_the_requests_digest())
        assert later.uid != own.uid
        events = list(client.events)

        with pytest.raises(BuildConflict, match=NOT_RECORDED) as refused:
            await _submit(client, jobs, backend)
        row = client.rows["demo-1"]
        # Not failed, and not written: it waits as it did, still recording its own job.
        assert (row.status, row.status_detail, _recorded(row)) == ("pending", None, (own.uid, own.attempt_id))
        assert later.uid not in row.model_dump_json()
        assert client.events == events
        # Read a few times first: a submit that created the job found would record it on the row.
        assert client.reads == ["demo-1"] * (1 + submit.ADOPT_READS)
        # The answer says what to do, as it does for a row that records no job.
        assert "submit the request again in a moment. If the answer is still this, submit the next revision" in str(
            refused.value
        )

    async def test_a_job_is_the_builds_own_only_if_every_row_still_waiting_records_it(self) -> None:
        """A set of two, submitted again. One row records the job under the build's name. The other records a
        job that held the name before it: the job found is not the one that image's row says builds it, and
        the submit is answered a conflict. Neither row is written. The one that records the job is left to
        it: that job is its build, and is running. The other is left for a submit that may be about to
        record the job on it, and is failed once the job has ended if none does."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(max_specs_per_set=2), PROFILES)
        specs = [_spec(name="a"), _spec(name="b")]
        await _submit(client, jobs, backend, specs=specs)
        earlier = {"job_uid": "platform-job-earlier", "job_attempt_id": "attempt-earlier"}
        held = client.rows["demo-1.b"]
        client.written_elsewhere("demo-1.b", provenance=held.provenance.model_copy(update=earlier))
        events = list(client.events)

        with pytest.raises(BuildConflict, match=NOT_RECORDED + ". Another submit of the same request may be"):
            await _submit(client, jobs, backend, specs=specs)
        assert [row.status for row in client.rows.values()] == ["pending", "pending"]
        assert _recorded(client.rows["demo-1.a"]) == (jobs.answered[JOB].uid, jobs.answered[JOB].attempt_id)
        assert _recorded(client.rows["demo-1.b"]) == ("platform-job-earlier", "attempt-earlier")
        assert client.events == events

        # Once that row has settled, the same request finds nothing waiting on another job, and the one image
        # still building: a row that has settled is not asked which job it records.
        client.written_elsewhere("demo-1.b", status="failed", status_detail="abandoned")
        again = await _submit(client, jobs, backend, specs=specs)
        assert [row.status for row in again.images] == ["pending", "failed"]
        assert client.events == events

    async def test_a_resubmit_that_creates_the_job_anew_records_the_new_one(self) -> None:
        """The build's job was deleted before it finished, and nothing is under its name. The same request
        again creates a job, and the row, still waiting, is completed by that job's steps and no other's."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        await _submit(client, jobs, backend)
        first = jobs.answered[JOB]
        jobs.delete(JOB)

        (row,) = (await _submit(client, jobs, backend)).images
        second = jobs.answered[JOB]
        assert second != first
        assert _recorded(row) == _recorded(client.rows["demo-1"]) == (second.uid, second.attempt_id)

    async def test_only_a_row_still_waiting_is_written(self) -> None:
        """A row that has settled is never written again, and no step completes it now."""
        client = FakeEntityClient()
        backend = RemoteBackend(_config(max_specs_per_set=2), PROFILES)
        specs = [_spec(name="a"), _spec(name="b")]
        with pytest.raises(RuntimeError, match="jobs unavailable"):
            await _submit(client, _Unavailable(client.events), backend, specs=specs)
        # Neither row records a job: none was created.
        assert [_recorded(row) for row in client.rows.values()] == [(None, None), (None, None)]
        client.rows["demo-1.a"].status = "failed"
        client.events.clear()

        jobs = FakeJobs(client.events)
        result = await _submit(client, jobs, backend, specs=specs)
        assert client.events == ["create_job", "update:demo-1.b:pending"]
        created = jobs.answered[JOB]
        assert [_recorded(row) for row in result.images] == [(None, None), (created.uid, created.attempt_id)]

    async def test_a_row_that_changed_meanwhile_is_read_once_more_and_written(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        client.before_write = lambda: client.written_elsewhere("demo-1")
        (row,) = (await _submit(client, jobs, RemoteBackend(_config(), PROFILES))).images
        assert client.events == ["create:demo-1", "create_job", "update_conflict:demo-1", "update:demo-1:pending"]
        assert client.reads == ["demo-1"]
        created = jobs.answered[JOB]
        assert _recorded(row) == _recorded(client.rows["demo-1"]) == (created.uid, created.attempt_id)

    async def test_a_row_that_settled_meanwhile_is_left_as_it_settled(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        client.before_write = lambda: client.written_elsewhere("demo-1", status="failed", status_detail="abandoned")
        (row,) = (await _submit(client, jobs, RemoteBackend(_config(), PROFILES))).images
        assert client.events == ["create:demo-1", "create_job", "update_conflict:demo-1"]
        assert (row.status, row.status_detail, _recorded(row)) == ("failed", "abandoned", (None, None))

    async def test_a_resubmit_whose_row_only_changed_meanwhile_still_records_the_new_job(self) -> None:
        """The row records the job its first submit created, deleted since, and the same request again
        creates another. Before that submit writes its job on the row, something else writes the row and
        leaves what it says of its job alone, so the write is refused for the row's version. Read again,
        the row says of its job what it said when the submit read it: no other create was recorded on it,
        and the new job is written in the earlier one's place. Left recording the earlier job, the row
        would refuse the new job's step."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        await _submit(client, jobs, backend)
        earlier = jobs.answered[JOB]
        jobs.delete(JOB)
        client.events.clear()

        client.before_write = lambda: client.written_elsewhere("demo-1")
        (row,) = (await _submit(client, jobs, backend)).images
        created = jobs.answered[JOB]
        assert created.uid != earlier.uid
        assert client.events == ["create_job", "update_conflict:demo-1", "update:demo-1:pending"]
        # Read to adopt the row, and once more after the refused write.
        assert client.reads == ["demo-1"] * 2
        assert _recorded(row) == _recorded(client.rows["demo-1"]) == (created.uid, created.attempt_id)

    async def test_a_resubmit_whose_row_came_to_record_another_job_meanwhile_leaves_it_to_that_job(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The same, but what was written meanwhile is another job: another submit of the request created
        one since, and recorded it. The row no longer says of its job what it said when this submit read
        it, and is left to the job it records now."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        await _submit(client, jobs, backend)
        jobs.delete(JOB)
        client.events.clear()
        later = {"job_uid": "platform-job-later", "job_attempt_id": "attempt-later"}

        def another_submit_records_its_job() -> None:
            held = client.rows["demo-1"]
            client.written_elsewhere("demo-1", provenance=held.provenance.model_copy(update=later))

        client.before_write = another_submit_records_its_job
        with caplog.at_level(logging.WARNING):
            result = await _submit(client, jobs, backend)
        assert result.job == JOB
        stored = client.rows["demo-1"]
        assert _recorded(stored) == ("platform-job-later", "attempt-later")
        assert jobs.answered[JOB].uid not in stored.model_dump_json()
        assert client.events == ["create_job", "update_conflict:demo-1"]
        assert "image default/demo-1 changed while its build job was being recorded on it" in caplog.text

    async def test_a_resubmit_whose_row_settled_meanwhile_returns_it_as_it_settled(self) -> None:
        """The row records an earlier job, and is failed, as abandoned, before the resubmit writes the job
        it created. It is returned as it settled, as a row that recorded no job is, and not as the submit
        read it: no step completes it now."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        await _submit(client, jobs, backend)
        earlier = jobs.answered[JOB]
        jobs.delete(JOB)
        client.events.clear()

        client.before_write = lambda: client.written_elsewhere("demo-1", status="failed", status_detail="abandoned")
        (row,) = (await _submit(client, jobs, backend)).images
        assert client.events == ["create_job", "update_conflict:demo-1"]
        assert (row.status, row.status_detail) == ("failed", "abandoned")
        assert _recorded(row) == (earlier.uid, earlier.attempt_id)

    async def test_a_write_that_was_made_though_its_answer_was_lost_is_the_submits_own(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The store wrote the job on the row and its answer was lost, so the write is sent again and refused
        for the row's version. Read again, the row records the very job this submit created, whose ID no one
        else was given: the write is the submit's own, and the row is returned as it is stored."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)

        def written_and_not_answered() -> None:
            created = jobs.answered[JOB]
            recorded = {"job_uid": created.uid, "job_attempt_id": created.attempt_id}
            held = client.rows["demo-1"]
            client.written_elsewhere("demo-1", provenance=held.provenance.model_copy(update=recorded))

        client.before_write = written_and_not_answered
        with caplog.at_level(logging.WARNING):
            (row,) = (await _submit(client, jobs, RemoteBackend(_config(), PROFILES))).images
        created = jobs.answered[JOB]
        assert row is client.rows["demo-1"]
        assert _recorded(row) == (created.uid, created.attempt_id)
        assert client.events == ["create:demo-1", "create_job", "update_conflict:demo-1"]
        assert "changed while its build job was being recorded" not in caplog.text

    @pytest.mark.parametrize(
        "meanwhile",
        ["replaced", "changed-then-replaced", "recorded-by-a-later-create", "keeps-changing", "the-store-fails"],
    )
    async def test_a_row_that_cant_be_written_is_logged_by_name_and_the_submit_still_answers(
        self, meanwhile: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The log says what then becomes of the image, which goes by how the platform authenticates a
        caller: a step that is verified is not let complete a row that doesn't record its job, and where no
        caller is verified the step completes it all the same."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        later = {"job_uid": "platform-job-later", "job_attempt_id": "attempt-later"}

        def happens() -> None:
            if meanwhile == "replaced":
                client.replaced("demo-1")
            elif meanwhile == "changed-then-replaced":
                # The write is refused for the row's version. Read again, the name holds another row.
                client.written_elsewhere("demo-1")
                client.before_read[1] = lambda: client.replaced("demo-1")
            elif meanwhile == "recorded-by-a-later-create":
                held = client.rows["demo-1"]
                client.written_elsewhere("demo-1", provenance=held.provenance.model_copy(update=later))
            elif meanwhile == "keeps-changing":
                client.written_elsewhere("demo-1")
                client.before_write = lambda: client.written_elsewhere("demo-1")
            else:
                raise RuntimeError("entity store down")

        client.before_write = happens
        with caplog.at_level(logging.WARNING):
            result = await _submit(client, jobs, RemoteBackend(_config(), PROFILES))
        assert result.job == JOB
        assert "image default/demo-1" in caplog.text
        # A row that changed is left as it is, and may record another job by then. One that couldn't be
        # written records what it did: no job, or an earlier one, and the log says how each then ends.
        said = (
            "no step of this job completes it;"
            if meanwhile in ("recorded-by-a-later-create", "keeps-changing", "changed-then-replaced")
            else (
                "no step of this job completes it, and the image then fails: its completion is put off until "
                "its step gives up, or refused at once where the row records an earlier job;"
            )
        )
        assert f"under workload token exchange {said}" in caplog.text
        assert "where no caller is verified, the step completes it all the same" in caplog.text
        stored = client.rows["demo-1"]
        created = jobs.answered[JOB]
        assert _recorded(stored) == (
            ("platform-job-later", "attempt-later") if "recorded" in meanwhile else (None, None)
        )
        assert created.uid not in stored.model_dump_json()
        assert not [event for event in client.events if event.startswith("update:")]

    async def test_a_create_that_names_no_attempt_is_not_recorded(self, caplog: pytest.LogCaptureFixture) -> None:
        """A row records both or neither: a row that named a job and no attempt would wait on one for good."""

        class _NoAttempt(FakeJobs):
            async def create_job(self, request: CreateHelixJobRequest) -> CreatedJob:
                return CreatedJob(uid=(await super().create_job(request)).uid, attempt_id=None)

        client = FakeEntityClient()
        with caplog.at_level(logging.ERROR):
            (row,) = (await _submit(client, _NoAttempt(client.events), RemoteBackend(_config(), PROFILES))).images
        assert _recorded(row) == (None, None)
        assert client.events == ["create:demo-1", "create_job"]
        assert f"Jobs created job {JOB} and named no ID, or no attempt, for it" in caplog.text
        assert "its completion is put off until its step gives up" in caplog.text

    async def test_a_submit_made_at_the_same_moment_adopts_the_job_once_the_other_has_recorded_it(self) -> None:
        """Two submits of one request, at once. One creates the job; the other finds it under the build's
        name a moment before the first has written it on the row. It reads the row again, and adopts the job
        once the row records it, where failing the row would have failed the first submit's build."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        # The first submit, up to its create: the row is written, and the job exists.
        with pytest.raises(RuntimeError, match="jobs unavailable"):
            await _submit(client, _Unavailable(client.events), backend)
        created = jobs.create(_carrying_the_requests_digest())
        held = client.rows["demo-1"]

        def the_first_submit_records_its_job() -> None:
            recorded = {"job_uid": created.uid, "job_attempt_id": created.attempt_id}
            client.written_elsewhere("demo-1", provenance=held.provenance.model_copy(update=recorded))

        # The second reads the row to adopt it, then twice to see whether it records the job.
        client.before_read[3] = the_first_submit_records_its_job
        (row,) = (await _submit(client, jobs, backend)).images
        assert (row.status, _recorded(row)) == ("pending", (created.uid, created.attempt_id))
        assert client.reads == ["demo-1"] * 3
        assert not [event for event in client.events if event.startswith("update")]

    async def test_a_job_no_row_comes_to_record_is_refused_after_a_few_reads_and_the_row_is_left(self) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        with pytest.raises(RuntimeError, match="jobs unavailable"):
            await _submit(client, _Unavailable(client.events), backend)
        jobs.create(_carrying_the_requests_digest())
        events = list(client.events)

        with pytest.raises(BuildConflict, match=NOT_RECORDED + ". Another submit of the same request may be"):
            await _submit(client, jobs, backend)
        assert client.reads == ["demo-1"] * (1 + submit.ADOPT_READS)
        # Not failed: the job may be one that a submit of this request created, and has yet to record.
        assert client.rows["demo-1"].status == "pending" and client.events == events

    async def test_a_job_whose_record_failed_is_not_recorded_by_the_same_request_again(self) -> None:
        """Jobs created the job, and the store failed the write that records it on the row. The submit
        answers with the row as it was. The same request again finds that job under the build's name, on a
        row that records none: nothing tells it from a job someone else put there, so nothing of it is
        written, the answer is a conflict, and the row is left for settling. Once it has settled, the same
        request returns it."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)

        def the_store_fails() -> None:
            raise RuntimeError("entity store down")

        client.before_write = the_store_fails
        (row,) = (await _submit(client, jobs, backend)).images
        created = jobs.answered[JOB]
        assert (row.status, _recorded(row)) == ("pending", (None, None))
        events = list(client.events)
        assert events == ["create:demo-1", "create_job"]

        # The store answers again. The job found is the one that submit created, and nothing shows it.
        with pytest.raises(BuildConflict, match=NOT_RECORDED + ". Another submit of the same request may be"):
            await _submit(client, jobs, backend)
        stored = client.rows["demo-1"]
        assert (stored.status, stored.status_detail, _recorded(stored)) == ("pending", None, (None, None))
        assert created.uid not in stored.model_dump_json()
        assert client.events == events and jobs.answered == {JOB: created}

        # Its job ends without completing it, and the row is failed. The same request then returns it.
        client.written_elsewhere("demo-1", status="failed", status_detail="abandoned")
        (again,) = (await _submit(client, jobs, backend)).images
        assert (again.status, again.status_detail, _recorded(again)) == ("failed", "abandoned", (None, None))
        assert client.events == events

    async def test_a_submit_that_records_its_job_after_another_found_it_still_builds(self) -> None:
        """Two submits of one request: a client that sends it again after a timeout, say. The first is
        still inside Jobs' create when the second finds the job under the build's name, on a row that
        records nothing. The second waits a few reads, answers a conflict, and leaves the row. The first's
        create then returns, and it records its job on the row, which a step of that job completes.

        Failed by the second, the row would have been found failed by the first, and by the first's job:
        the build was lost to a retry of its own request."""
        client = FakeEntityClient()
        backend = RemoteBackend(_config(), PROFILES)
        returns = asyncio.Event()
        jobs = _SlowToAnswer(client.events, returns)
        async with _submitting(client, jobs, backend) as first:
            with pytest.raises(BuildConflict, match=NOT_RECORDED + ". Another submit of the same request may be"):
                await _submit(client, FakeJobs.sharing(jobs), backend)
            assert (client.rows["demo-1"].status, _recorded(client.rows["demo-1"])) == ("pending", (None, None))

            returns.set()
            (row,) = (await first).images
        created = jobs.answered[JOB]
        assert (row.status, _recorded(row)) == ("pending", (created.uid, created.attempt_id))
        assert client.events == ["create:demo-1", "create_job", "update:demo-1:pending"]
        # And the same request once more now finds the job its row records.
        (again,) = (await _submit(client, FakeJobs.sharing(jobs), backend)).images
        assert _recorded(again) == (created.uid, created.attempt_id)

    async def test_a_resubmit_that_records_its_job_over_an_earlier_one_after_another_found_it_still_builds(
        self,
    ) -> None:
        """The row records the job its first submit created, which was cancelled and deleted before the row
        was failed for it. The same request again finds the name free and creates a job, and is still
        inside Jobs' create when a retry of it finds that job under the build's name, on a row that
        records the earlier one. The retry waits a few reads, answers a conflict, and leaves the row. The
        resubmit's create then returns, and it records its job on the row in place of the earlier one,
        which a step of that job completes.

        Failed by the retry, the row would have been found failed by the resubmit, and by the resubmit's
        job: the build was lost to a retry of its own request, as one whose row records no job was."""
        client = FakeEntityClient()
        backend = RemoteBackend(_config(), PROFILES)
        returns = asyncio.Event()
        returns.set()
        jobs = _SlowToAnswer(client.events, returns)
        await _submit(client, jobs, backend)
        earlier = jobs.answered[JOB]
        jobs.delete(JOB)
        client.events.clear()

        returns.clear()
        async with _submitting(client, jobs, backend) as resubmit:
            assert jobs.answered[JOB].uid != earlier.uid

            with pytest.raises(BuildConflict, match=NOT_RECORDED) as refused:
                await _submit(client, FakeJobs.sharing(jobs), backend)
            row = client.rows["demo-1"]
            assert (row.status, row.status_detail, _recorded(row)) == (
                "pending",
                None,
                (earlier.uid, earlier.attempt_id),
            )
            assert client.reads == ["demo-1"] * (2 + submit.ADOPT_READS)
            assert "Another submit of the same request may be recording it now" in str(refused.value)

            returns.set()
            (row,) = (await resubmit).images
        created = jobs.answered[JOB]
        assert (row.status, _recorded(row)) == ("pending", (created.uid, created.attempt_id))
        assert client.events == ["create_job", "update:demo-1:pending"]
        # The row is there for the new job's step to complete: failed, it would have refused the completion.
        built = await complete(
            client,
            workspace="default",
            name="demo-1",
            digest="sha256:" + "c" * 64,
            external=ExternalImage(build_id="build-1"),
        )
        assert (built.status, _recorded(built)) == ("ready", (created.uid, created.attempt_id))

    async def test_a_submit_that_ends_before_jobs_holds_its_job_is_not_waited_for(self) -> None:
        """The two tests above wait for the job a submit creates while it runs on its own. A submit that ends
        first creates none: what it raised is what the wait raises, where it would otherwise never end."""
        client = FakeEntityClient()
        with pytest.raises(RuntimeError, match="jobs unavailable"):
            async with _submitting(client, _Unavailable(client.events), RemoteBackend(_config(), PROFILES)):
                raise AssertionError("a submit that created no job was handed over")
        # It got as far as its row, and no further.
        assert client.events == ["create:demo-1"]

    async def test_a_submit_adopts_the_job_once_another_has_recorded_it_over_an_earlier_one(self) -> None:
        """The row records an earlier job of the request, deleted since, and a resubmit has created another
        under the build's name. A retry of the resubmit finds that job a moment before the resubmit writes
        it on the row. It reads the row again, and adopts the job once the row records it."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        await _submit(client, jobs, backend)
        earlier = jobs.answered[JOB]
        jobs.delete(JOB)
        # The resubmit, up to its create: the job exists, and the row still records the earlier one.
        created = jobs.create(_carrying_the_requests_digest())
        held = client.rows["demo-1"]
        assert _recorded(held) == (earlier.uid, earlier.attempt_id) != (created.uid, created.attempt_id)
        events = list(client.events)

        def the_resubmit_records_its_job() -> None:
            recorded = {"job_uid": created.uid, "job_attempt_id": created.attempt_id}
            client.written_elsewhere("demo-1", provenance=held.provenance.model_copy(update=recorded))

        # The retry reads the row to adopt it, then twice to see whether it records the job.
        client.before_read[3] = the_resubmit_records_its_job
        (row,) = (await _submit(client, jobs, backend)).images
        assert (row.status, _recorded(row)) == ("pending", (created.uid, created.attempt_id))
        assert client.reads == ["demo-1"] * 3
        # The retry wrote nothing, and created nothing.
        assert client.events == events

    @pytest.mark.parametrize("meanwhile", ["replaced", "deleted"])
    async def test_a_row_that_will_never_record_the_job_is_not_waited_for(self, meanwhile: str) -> None:
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        with pytest.raises(RuntimeError, match="jobs unavailable"):
            await _submit(client, _Unavailable(client.events), backend)
        jobs.create(_carrying_the_requests_digest())

        def happens() -> None:
            if meanwhile == "replaced":
                client.replaced("demo-1")
            else:
                del client.rows["demo-1"]

        client.before_read[2] = happens
        with pytest.raises(BuildConflict, match=NOT_RECORDED + "$"):
            await _submit(client, jobs, backend)
        assert client.reads == ["demo-1"] * 2
        # A row that is no longer the one this submit read is not written: it is someone else's now.
        assert not [event for event in client.events if event.startswith("update:")]

    @pytest.mark.parametrize("settled", ["ready", "failed"])
    @pytest.mark.parametrize("records", ["a-job-since-deleted", "no-job"])
    async def test_rows_that_have_all_settled_are_returned_whichever_job_holds_the_name(
        self, settled: str, records: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A finished build, submitted again. Its job was deleted, which Jobs allows once a job has ended,
        and another job has its name: one a resubmit created, which wrote nothing on a row that had settled,
        or anyone's. The row records the first job, or none: it settled before one was recorded, or was
        written before rows recorded their job.

        No step completes a settled row, so whose job holds the name decides nothing. The row is returned as
        it is, as a build in this cluster returns its rows: answered with a conflict, a caller would build
        an image that is already ready under the next revision."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        if records == "no-job":
            with pytest.raises(RuntimeError, match="jobs unavailable"):
                await _submit(client, _Unavailable(client.events), backend)
        else:
            await _submit(client, jobs, backend)
            jobs.delete(JOB)
        row = client.rows["demo-1"]
        recorded = _recorded(row)
        if settled == "ready":
            row.status, row.digest, row.external = "ready", "sha256:" + "c" * 64, ExternalImage(build_id="build-1")
        else:
            row.status, row.status_detail = "failed", "its job ended without completing it"
        jobs.create(_carrying_the_requests_digest())
        events = list(client.events)

        with caplog.at_level(logging.INFO):
            result = await _submit(client, jobs, backend)
        assert result.job == JOB
        assert [(image.name, image.status, _recorded(image)) for image in result.images] == [
            ("demo-1", settled, recorded)
        ]
        # Nothing is written, nothing is created, and the job is not said to be the build's.
        assert client.events == events
        assert "no image of this request still waits on a job" in caplog.text and "adopting it" not in caplog.text

    async def test_a_finished_build_is_returned_however_often_it_is_submitted_again(self) -> None:
        """The row is ready, and its job is deleted. The same request again finds the name free and creates
        a job, which writes nothing on a row that has settled: the row still records the first. Once more,
        and the job found is one the row doesn't record. Both are answered with the ready row."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        await _submit(client, jobs, backend)
        first = jobs.answered[JOB]
        await complete(
            client,
            workspace="default",
            name="demo-1",
            digest="sha256:" + "c" * 64,
            external=ExternalImage(build_id="build-1"),
        )
        jobs.delete(JOB)
        client.events.clear()

        for created in (["create_job"], []):
            (row,) = (await _submit(client, jobs, backend)).images
            assert (row.status, _recorded(row)) == ("ready", (first.uid, first.attempt_id))
            assert client.events == created
            client.events.clear()
        assert jobs.answered[JOB] != first

    async def test_a_row_that_settled_while_the_rows_were_read_again_ends_the_wait(self) -> None:
        """The one row still waiting, on a job it doesn't record, fails meanwhile: abandoned, say. Nothing
        is left for a job to complete, and the row is returned as it settled."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        backend = RemoteBackend(_config(), PROFILES)
        with pytest.raises(RuntimeError, match="jobs unavailable"):
            await _submit(client, _Unavailable(client.events), backend)
        jobs.create(_carrying_the_requests_digest())
        client.before_read[3] = lambda: client.written_elsewhere("demo-1", status="failed", status_detail="abandoned")

        (row,) = (await _submit(client, jobs, backend)).images
        assert (row.status, row.status_detail) == ("failed", "abandoned")
        assert client.reads == ["demo-1"] * 3

    async def test_a_create_refused_for_a_job_that_cant_then_be_read_ends_as_jobs_ended_it(self) -> None:
        """A 409 from Jobs is not proof that a job of that name exists: it refuses a create for other reasons
        too. The submit ends with Jobs' own error, and the row is left waiting, for settling to fail."""

        class _RefusedForAnotherReason(FakeJobs):
            async def create_job(self, request: CreateHelixJobRequest) -> CreatedJob:
                raise ConflictError(httpx.Response(409, request=httpx.Request("POST", "http://j")))

            async def get_job(self, name: str) -> ExistingJob:
                raise NotFoundError(httpx.Response(404, request=httpx.Request("GET", "http://j")))

        client = FakeEntityClient()
        with pytest.raises(NotFoundError):
            await _submit(client, _RefusedForAnotherReason(client.events), RemoteBackend(_config(), PROFILES))
        assert client.events == ["create:demo-1"]
        assert (client.rows["demo-1"].status, _recorded(client.rows["demo-1"])) == ("pending", (None, None))

    async def test_a_backend_says_itself_whether_its_rows_record_their_job(self) -> None:
        """The submit asks the backend, not its name: a row completed on its caller's word records its job."""
        assert RemoteBackend.binds_rows_to_job is True
        assert ExecutionBackend.binds_rows_to_job is False

        class Renamed(RemoteBackend):
            name = "another-service"

        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        (row,) = (await _submit(client, jobs, Renamed(_config(), PROFILES))).images
        assert row.provenance.backend == "another-service"
        assert _recorded(row) == (jobs.answered[JOB].uid, jobs.answered[JOB].attempt_id)

    async def test_a_build_in_this_cluster_takes_the_job_by_its_digest_and_records_nothing(self) -> None:
        """The same job made beforehand, met by a build in this cluster: taken for the build's own, as it
        always was, and the row is not written again."""
        client = FakeEntityClient()
        jobs = FakeJobs(client.events)
        jobs.create(_carrying_the_requests_digest())
        await _submit_in_cluster(client, jobs)
        row = client.rows["demo-1"]
        assert (row.status, row.provenance.backend, _recorded(row)) == ("pending", "execution", (None, None))
        assert client.events == ["create:demo-1"]
        assert client.reads == []


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

    async def test_a_remote_row_is_not_completed_without_the_services_name_for_the_build(self) -> None:
        """A row a build service built is ready only with the name of the build that made it."""
        client = FakeEntityClient()
        await _submit(client, FakeJobs(client.events), RemoteBackend(_config(), PROFILES))
        digest = "sha256:" + "c" * 64
        for external in (None, ExternalImage(image_id="img-1")):
            with pytest.raises(CompletionConflict, match="its completion must name the build"):
                await complete(client, workspace="default", name="demo-1", digest=digest, external=external)
        assert client.rows["demo-1"].status == "pending"

    async def test_a_name_that_is_only_spaces_names_no_build(self) -> None:
        """The remote-build step refuses a build its service's client left unnamed, a blank name included. Any
        other caller is held to the same: a row is not made ready under a name no operator can look up."""
        client = FakeEntityClient()
        await _submit(client, FakeJobs(client.events), RemoteBackend(_config(), PROFILES))
        digest = "sha256:" + "c" * 64
        for blank in (" ", "\t\n", "   "):
            with pytest.raises(CompletionConflict, match="its completion must name the build"):
                await complete(
                    client,
                    workspace="default",
                    name="demo-1",
                    digest=digest,
                    external=ExternalImage(image_id="img-1", build_id=blank),
                )
        assert client.rows["demo-1"].status == "pending"
        # A name is otherwise the service's own, and is recorded as it was sent: nothing is trimmed from it.
        named = ExternalImage(build_id=" build 1 ")
        ready = await complete(client, workspace="default", name="demo-1", digest=digest, external=named)
        assert (ready.status, ready.external) == ("ready", named)

    async def test_completing_again_must_name_the_same_build(self) -> None:
        client = FakeEntityClient()
        await _submit(client, FakeJobs(client.events), RemoteBackend(_config(), PROFILES))
        digest = "sha256:" + "c" * 64
        external = ExternalImage(image_id="img-1", build_id="build-1")
        first = await complete(client, workspace="default", name="demo-1", digest=digest, external=external)

        # The same completion again, as a retried call makes it, returns the row: a handle read anew from the
        # request, equal to the row's and not the same object.
        retried = ExternalImage.model_validate(external.model_dump(mode="json"))
        assert retried is not first.external
        again = await complete(client, workspace="default", name="demo-1", digest=digest, external=retried)
        assert again.external == external
        # Anything else is not swallowed, whichever of the three differs: the row keeps the first, and the
        # caller is told.
        for other in (
            ExternalImage(image_id="img-2", build_id="build-1"),
            ExternalImage(image_id="img-1", build_id="build-2"),
            ExternalImage(image_id="img-1", build_id="build-1", attributes={"builder": "v2"}),
            ExternalImage(build_id="build-1"),
        ):
            with pytest.raises(CompletionConflict, match="another build's image"):
                await complete(client, workspace="default", name="demo-1", digest=digest, external=other)
        assert client.rows["demo-1"].external == external

    async def test_completing_again_must_say_the_image_starts_the_same_way(self) -> None:
        """The same digest and the same build with another runtime configuration is not the same completion.
        Returned as done, the row would keep the first one, and the caller would take it for what it sent."""
        client = FakeEntityClient()
        await _submit(client, FakeJobs(client.events), RemoteBackend(_config(), PROFILES))
        digest = "sha256:" + "c" * 64
        external = ExternalImage(build_id="build-1")
        starts = ImageRuntimeConfig(entrypoint=["/entrypoint.sh"], cmd=["serve"], working_dir="/srv", user="")

        async def again(runtime_config: ImageRuntimeConfig | None) -> ContainerImage:
            return await complete(
                client,
                workspace="default",
                name="demo-1",
                digest=digest,
                runtime_config=runtime_config,
                external=external,
            )

        await again(starts)
        # The same one, read anew from a retried request, returns the row.
        assert (await again(ImageRuntimeConfig.model_validate(starts.model_dump()))).runtime_config == starts
        for other in (
            starts.model_copy(update={"entrypoint": ["/altered"]}),
            starts.model_copy(update={"cmd": None}),
            starts.model_copy(update={"user": None}),
            None,
        ):
            with pytest.raises(CompletionConflict, match="another runtime configuration"):
                await again(other)
        assert client.rows["demo-1"].runtime_config == starts

    async def test_a_remote_row_completed_with_none_is_not_completed_again_with_one(self) -> None:
        """A build service's step sends the same on every try, so a row that records none was sent none."""
        client = FakeEntityClient()
        await _submit(client, FakeJobs(client.events), RemoteBackend(_config(), PROFILES))
        digest = "sha256:" + "c" * 64
        external = ExternalImage(build_id="build-1")
        await complete(client, workspace="default", name="demo-1", digest=digest, external=external)
        with pytest.raises(CompletionConflict, match="another runtime configuration"):
            await complete(
                client,
                workspace="default",
                name="demo-1",
                digest=digest,
                runtime_config=ImageRuntimeConfig(entrypoint=["/entrypoint.sh"]),
                external=external,
            )
        assert client.rows["demo-1"].runtime_config is None

    async def test_an_image_built_here_takes_again_what_its_row_does_not_record(self) -> None:
        """A row an earlier builder completed records no runtime configuration, or not the image's user. A push
        step whose completion is repeated across the upgrade sends both, for an image that is published."""
        client = FakeEntityClient()
        await _submit_in_cluster(client)
        digest = "sha256:" + "c" * 64
        starts = ImageRuntimeConfig(entrypoint=["/bin/app"], cmd=["serve"], working_dir="/app", user="agent")

        await complete(client, workspace="default", name="demo-1", digest=digest)
        repeated = await complete(client, workspace="default", name="demo-1", digest=digest, runtime_config=starts)
        # Returned as it is: a ready row is never written again.
        assert (repeated.status, repeated.runtime_config) == ("ready", None)

        client.rows["demo-1"].runtime_config = starts.model_copy(update={"user": None})
        repeated = await complete(client, workspace="default", name="demo-1", digest=digest, runtime_config=starts)
        assert repeated.runtime_config == starts.model_copy(update={"user": None})
        assert [event for event in client.events if event.startswith("update:")] == ["update:demo-1:ready"]

    async def test_an_image_built_here_is_not_completed_again_another_way(self) -> None:
        """What its row does record, a repeat that carries one must match, as for a row a build service built."""
        client = FakeEntityClient()
        await _submit_in_cluster(client)
        digest = "sha256:" + "c" * 64
        starts = ImageRuntimeConfig(entrypoint=["/bin/app"], cmd=["serve"], working_dir="/app", user="agent")
        await complete(client, workspace="default", name="demo-1", digest=digest, runtime_config=starts)

        assert (await complete(client, workspace="default", name="demo-1", digest=digest, runtime_config=starts)) is (
            client.rows["demo-1"]
        )
        for other in (
            starts.model_copy(update={"entrypoint": ["/altered"]}),
            starts.model_copy(update={"user": "root"}),
        ):
            with pytest.raises(CompletionConflict, match="another runtime configuration"):
                await complete(client, workspace="default", name="demo-1", digest=digest, runtime_config=other)
        assert client.rows["demo-1"].runtime_config == starts

    async def test_an_image_built_here_is_returned_to_a_repeat_that_carries_none(self) -> None:
        """A repeat that says nothing of how the image starts says nothing the row could differ from: the
        image's config is fixed by its digest. A client that sends only the digest is returned the row, as it
        was before a repeat was compared with it."""
        client = FakeEntityClient()
        await _submit_in_cluster(client)
        digest = "sha256:" + "c" * 64
        starts = ImageRuntimeConfig(entrypoint=["/bin/app"], cmd=["serve"], working_dir="/app", user="agent")
        await complete(client, workspace="default", name="demo-1", digest=digest, runtime_config=starts)

        repeated = await complete(client, workspace="default", name="demo-1", digest=digest)
        # Returned as it is, with what it records: a ready row is never written again.
        assert repeated is client.rows["demo-1"]
        assert (repeated.status, repeated.runtime_config) == ("ready", starts)
        assert [event for event in client.events if event.startswith("update:")] == ["update:demo-1:ready"]
        # The digest is still compared first.
        with pytest.raises(CompletionConflict, match="already ready as"):
            await complete(client, workspace="default", name="demo-1", digest="sha256:" + "e" * 64)

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
