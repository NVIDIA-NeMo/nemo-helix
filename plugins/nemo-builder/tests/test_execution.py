# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution backend's declarations: what it refuses, and where it publishes."""

from __future__ import annotations

import pytest
from nemo_builder_plugin.backend import Backend, BackendRejectedError
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_helix_plugin.jobs.endpoints import ExecutionProfile
from nemo_helix_plugin.jobs.execution_profiles import DockerJobExecutionProfile, KubernetesJobExecutionProfile


def _config(**overrides: object) -> BuilderConfig:
    settings: dict[str, object] = {
        "registry": "reg.example.com",
        "sandbox": {"image": "kaniko.example.com/executor:debug"},
    }
    return BuilderConfig.model_validate(settings | overrides)


def _profile(name: str, **config: object) -> KubernetesJobExecutionProfile:
    settings: dict[str, object] = {"namespace": "nhx-builds", "storage": {"pvc_name": "nhx-build-work"}}
    return KubernetesJobExecutionProfile.model_validate({"profile": name, "config": settings | config})


#: The build's Jobs execution profiles, as the quickstart has them.
PROFILES = [_profile(name) for name in ("build-fetch", "build-control", "build-push")]


def _spec(name: str | None = "main", *, repository: str | None = "team/app") -> BuildSpec:
    output = BuildOutput(repository=repository, tag="v1") if repository else None
    return BuildSpec(name=name, source=FileSetSource(fileset="fs-a"), output=output)


def _plan(*specs: BuildSpec, workspace: str = "default") -> BuildPlan:
    build_set = BuildSet(name="demo", revision=2, build_specs=list(specs) or [_spec()])
    return BuildPlan.resolve(build_set, workspace=workspace)


def test_the_deployment_runs_the_execution_backend() -> None:
    backend: Backend = load_backend(_config(), PROFILES)
    assert isinstance(backend, ExecutionBackend)


class TestADeploymentThatCannotBuild:
    @pytest.mark.parametrize(
        ("unset", "named"),
        [
            ({"registry": None}, "builder.registry"),
            ({"sandbox": {"image": None}}, "builder.sandbox.image"),
        ],
    )
    def test_a_setting_nothing_can_default(self, unset: dict[str, object], named: str) -> None:
        with pytest.raises(BackendRejectedError, match=named):
            ExecutionBackend(_config(**unset), PROFILES).check(_plan())

    def test_the_opensandbox_provider_without_its_server_is_refused(self) -> None:
        config = _config(sandbox={"image": "kaniko.example.com/executor:debug", "provider": "opensandbox"})
        with pytest.raises(BackendRejectedError, match="builder.sandbox.opensandbox"):
            ExecutionBackend(config, PROFILES).check(_plan())

    def test_the_opensandbox_provider_with_its_server_is_accepted(self) -> None:
        sandbox = {
            "image": "kaniko.example.com/executor:debug",
            "provider": "opensandbox",
            "opensandbox": {"domain": "opensandbox-server.opensandbox-system.svc.cluster.local"},
        }
        ExecutionBackend(_config(sandbox=sandbox), PROFILES).check(_plan())

    def test_a_configured_deployment_honors_every_request(self) -> None:
        ExecutionBackend(_config(), PROFILES).check(_plan(_spec("a"), _spec("b", repository=None)))


class TestTheBuildProfiles:
    """The sandbox mounts the fetch step's work volume in the build step's namespace, so the profiles must agree."""

    @pytest.mark.parametrize(
        ("profiles", "named"),
        [
            pytest.param(PROFILES[:2], "no execution profile cpu/build-push", id="missing"),
            pytest.param(
                [DockerJobExecutionProfile.model_validate({"profile": "build-fetch", "config": {}}), *PROFILES[1:]],
                "runs on docker",
                id="not-kubernetes",
            ),
            pytest.param(
                [_profile("build-fetch", storage={"pvc_name": ""}), *PROFILES[1:]],
                "names no work volume",
                id="no-work-volume",
            ),
            pytest.param(
                [PROFILES[0], _profile("build-control", namespace="elsewhere"), PROFILES[2]],
                "one namespace",
                id="two-namespaces",
            ),
            pytest.param(
                [*PROFILES[:2], _profile("build-push", storage={"pvc_name": "other"})],
                "one work volume",
                id="two-work-volumes",
            ),
        ],
    )
    def test_profiles_the_sandbox_cannot_follow_are_refused(self, profiles: list[ExecutionProfile], named: str) -> None:
        with pytest.raises(BackendRejectedError, match=named):
            ExecutionBackend(_config(), profiles).check(_plan())

    def test_profiles_that_all_leave_the_namespace_to_jobs_agree(self) -> None:
        profiles = [_profile(name, namespace=None) for name in ("build-fetch", "build-control", "build-push")]
        ExecutionBackend(_config(), profiles).check(_plan())


class TestDestination:
    def test_a_named_output_lands_under_the_workspace(self) -> None:
        backend = ExecutionBackend(_config(), PROFILES)
        placed = _plan().with_destinations(backend.destination).images[0].placed
        assert (placed.registry, placed.repository, placed.system_tag) == (
            "reg.example.com",
            "default/team/app",
            "default--demo-2.main",
        )

    def test_an_unnamed_output_lands_at_set_and_spec(self) -> None:
        backend = ExecutionBackend(_config(), PROFILES)
        placed = _plan(_spec("verifier", repository=None)).with_destinations(backend.destination).images[0].placed
        assert placed.repository == "default/demo/verifier"

    def test_an_unnamed_spec_without_an_output_lands_at_the_set(self) -> None:
        backend = ExecutionBackend(_config(), PROFILES)
        image = _plan(_spec(None, repository=None)).with_destinations(backend.destination).images[0]
        assert (image.name, image.placed.repository, image.placed.system_tag) == (
            "demo-2",
            "default/demo",
            "default--demo-2",
        )

    def test_the_prefix_goes_ahead_of_the_workspace(self) -> None:
        backend = ExecutionBackend(_config(repository_prefix="proj/artifacts"), PROFILES)
        placed = _plan().with_destinations(backend.destination).images[0].placed
        assert placed.repository == "proj/artifacts/default/team/app"

    def test_two_workspaces_asking_for_one_repository_get_two(self) -> None:
        backend = ExecutionBackend(_config(), PROFILES)
        a = _plan(workspace="team-a").with_destinations(backend.destination).images[0].placed
        b = _plan(workspace="team-b").with_destinations(backend.destination).images[0].placed
        assert a.repository != b.repository

    def test_a_workspace_that_cannot_be_a_repository_component_is_refused(self) -> None:
        backend = ExecutionBackend(_config(), PROFILES)
        with pytest.raises(ValueError, match="repository"):
            _plan(workspace="Team@A").with_destinations(backend.destination)

    def test_every_image_has_its_own_system_tag(self) -> None:
        backend = ExecutionBackend(_config(), PROFILES)
        plan = _plan(_spec("a", repository=None), _spec("b", repository=None)).with_destinations(backend.destination)
        assert [image.placed.system_tag for image in plan.images] == ["default--demo-2.a", "default--demo-2.b"]
