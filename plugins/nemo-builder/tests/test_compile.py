# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the compiler must produce, asserted without a cluster."""

from __future__ import annotations

from nemo_builder_plugin.compile import WORK_MOUNT
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.steps import (
    REGISTRY_PASSWORD_ENV,
    REGISTRY_USERNAME_ENV,
    SIGNING_KEY_ENV,
    ContextSource,
    PushStepConfig,
    SuperviseStepConfig,
)
from nemo_helix_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_helix_plugin.jobs.providers import CPUExecutionProvider
from nemo_helix_plugin.jobs.spec import HelixJobSpec

WORKSPACE = "default"
SYSTEM_TAG = "default--demo-1.main"
SANDBOX_IMAGE = "kaniko.example.com/executor:debug"


def _compile(build_set: BuildSet, *, config: BuilderConfig) -> HelixJobSpec:
    backend = ExecutionBackend(config)
    plan = BuildPlan.resolve(build_set, workspace=WORKSPACE)
    backend.check(plan)
    return backend.compile(plan.with_destinations(backend.destination))


def _config(*, registry: str = "reg.example.com", sandbox_cpu: str = "2", sandbox_memory: str = "8Gi") -> BuilderConfig:
    return BuilderConfig(
        registry=registry,
        sandbox_image=SANDBOX_IMAGE,
        sandbox_cpu=sandbox_cpu,
        sandbox_memory=sandbox_memory,
    )


def _spec(name: str, *, fileset: str = "fs-a", context_path: str | None = None) -> BuildSpec:
    return BuildSpec(
        name=name,
        source=FileSetSource(fileset=fileset, context_path=context_path),
        output=BuildOutput(repository=f"team/{name}", tag="v1"),
    )


def _set(*specs: BuildSpec) -> BuildSet:
    return BuildSet(name="demo", revision=1, build_specs=list(specs) or [_spec("main")])


class TestShape:
    def test_three_steps_in_order(self) -> None:
        spec = _compile(_set(), config=_config())
        assert [s.name for s in spec.steps] == ["fetch", "build", "push"]

    def test_each_step_names_a_different_profile(self) -> None:
        spec = _compile(_set(), config=_config())
        profiles = [s.executor.profile for s in spec.steps]
        assert profiles == ["build-fetch", "build-control", "build-push"]
        assert len(set(profiles)) == 3

    def test_each_step_runs_its_own_binary(self) -> None:
        spec = _compile(_set(), config=_config())
        executors = [s.executor for s in spec.steps]
        # Narrowed, not suppressed: a step on `Provider`'s subprocess arm would have no container.
        assert all(isinstance(e, CPUExecutionProvider) for e in executors)
        assert [e.container.command for e in executors if isinstance(e, CPUExecutionProvider)] == [
            ["nhx-build", "fetch"],
            ["nhx-build", "supervise"],
            ["nhx-build", "push"],
        ]


class TestItSurvivesTheWire:
    def test_the_compiled_spec_survives_exclude_unset(self) -> None:
        """The Jobs client serializes with `exclude_unset`, which drops a defaulted discriminator."""
        spec = _compile(_set(), config=_config())
        on_the_wire = spec.model_dump(exclude_unset=True)

        for step in on_the_wire["steps"]:
            assert step["executor"].get("provider") == "cpu", (
                f"step {step['name']!r} lost its executor discriminator when serialized"
            )

        HelixJobSpec.model_validate(on_the_wire)


class TestOnlyPushGetsACredential:
    def test_push_reads_the_workspaces_credential_and_key_as_platform_secrets(self) -> None:
        """Bare names, which Jobs reads in the job's workspace, as the submitter."""
        push = _compile(_set(), config=_config()).steps[2]
        secrets = {env.name: env.from_secret.name for env in push.environment or [] if env.from_secret is not None}
        assert secrets == {
            REGISTRY_USERNAME_ENV: "builder-registry-username",
            REGISTRY_PASSWORD_ENV: "builder-registry-password",
            SIGNING_KEY_ENV: "builder-signing-key",
        }

    def test_the_deployment_names_the_secrets(self) -> None:
        config = _config().model_copy(update={"signing_key_secret": "release-signing-key"})
        push = _compile(_set(), config=config).steps[2]
        assert [
            env.from_secret.name for env in push.environment or [] if env.name == SIGNING_KEY_ENV and env.from_secret
        ] == ["release-signing-key"]

    def test_fetch_and_build_read_no_secret(self) -> None:
        spec = _compile(_set(), config=_config())
        assert not spec.secrets
        for step in spec.steps[:2]:
            assert not [env for env in step.environment or [] if env.from_secret is not None]

    def test_no_step_config_carries_a_secret(self) -> None:
        for step in _compile(_set(), config=_config()).steps:
            serialized = str(step.config).lower()
            assert "secret" not in serialized and "key" not in serialized


class TestTheBuildStepCannotReachTheWorkVolume:
    def test_fetch_and_push_request_the_work_volume_and_build_does_not(self) -> None:
        spec = _compile(_set(), config=_config())
        requested = [s.name for s in spec.steps if s.requires_persistent_storage]
        assert requested == ["fetch", "push"]

    def test_the_mount_path_is_the_env_vars_value(self) -> None:
        spec = _compile(_set(), config=_config())
        fetch = spec.steps[0]
        env = {e.name: e.value for e in fetch.environment or []}
        assert env[PERSISTENT_JOB_STORAGE_PATH_ENVVAR] == WORK_MOUNT


class TestTheSandboxIsToldNothingAboutPublishing:
    def test_no_registry_tag_or_secret_reaches_the_supervise_config(self) -> None:
        spec = _compile(_set(), config=_config())
        serialized = str(spec.steps[1].config)
        assert "reg.example.com" not in serialized
        assert SYSTEM_TAG not in serialized
        assert "cosign" not in serialized

    def test_the_sandbox_runs_the_configured_image(self) -> None:
        spec = _compile(_set(), config=_config())
        assert SuperviseStepConfig.model_validate(spec.steps[1].config).sandbox.image == SANDBOX_IMAGE

    def test_the_sandbox_is_sized_by_the_deployment_not_the_request(self) -> None:
        config = _config(sandbox_cpu="1", sandbox_memory="2Gi")
        spec = _compile(_set(), config=config)
        sandbox = SuperviseStepConfig.model_validate(spec.steps[1].config).sandbox
        assert (sandbox.cpu, sandbox.memory) == ("1", "2Gi")

    def test_the_sandbox_uses_public_dns_not_cluster_dns(self) -> None:
        spec = _compile(_set(), config=_config())
        sandbox = SuperviseStepConfig.model_validate(spec.steps[1].config).sandbox
        assert sandbox.dns_nameservers == ["8.8.8.8", "1.1.1.1"]


class TestGrouping:
    def test_specs_sharing_a_source_get_one_sandbox(self) -> None:
        spec = _compile(_set(_spec("main"), _spec("verifier")), config=_config())
        groups = SuperviseStepConfig.model_validate(spec.steps[1].config).groups
        assert len(groups) == 1
        assert [i.image for i in groups[0].images] == ["demo-1.main", "demo-1.verifier"]

    def test_distinct_sources_get_a_sandbox_each_mounting_only_its_own(self) -> None:
        spec = _compile(
            _set(_spec("a", context_path="environment"), _spec("b", context_path="tests")), config=_config()
        )
        groups = SuperviseStepConfig.model_validate(spec.steps[1].config).groups
        assert [g.source for g in groups] == [
            ContextSource(fileset="default/fs-a", context_path="environment"),
            ContextSource(fileset="default/fs-a", context_path="tests"),
        ]

    def test_fetch_downloads_a_shared_source_once(self) -> None:
        spec = _compile(_set(_spec("main"), _spec("verifier")), config=_config())
        assert len(spec.steps[0].config["sources"]) == 1

    def test_fetch_downloads_each_distinct_subtree(self) -> None:
        spec = _compile(
            _set(_spec("a", context_path="environment"), _spec("b", context_path="tests")), config=_config()
        )
        assert [s["context_path"] for s in spec.steps[0].config["sources"]] == ["environment", "tests"]

    def test_a_root_request_absorbs_subtree_requests_for_the_same_fileset(self) -> None:
        spec = _compile(_set(_spec("whole"), _spec("subtree", context_path="tests")), config=_config())
        sources = spec.steps[0].config["sources"]
        assert sources == [{"fileset": "default/fs-a", "context_path": None}]

        groups = SuperviseStepConfig.model_validate(spec.steps[1].config).groups
        assert [g.source.context_path for g in groups] == [None, "tests"]

    def test_a_root_request_on_a_different_fileset_absorbs_nothing(self) -> None:
        spec = _compile(
            _set(_spec("whole", fileset="fs-a"), _spec("subtree", fileset="fs-b", context_path="tests")),
            config=_config(),
        )
        sources = spec.steps[0].config["sources"]
        assert sources == [
            {"fileset": "default/fs-a", "context_path": None},
            {"fileset": "default/fs-b", "context_path": "tests"},
        ]


class TestEachImageHasItsOwnIdentity:
    def test_specs_sharing_a_repository_push_distinct_system_tags(self) -> None:
        staging = BuildSpec(
            name="staging",
            source=FileSetSource(fileset="fs-a"),
            output=BuildOutput(repository="team/app", tag="staging"),
        )
        prod = BuildSpec(
            name="prod",
            source=FileSetSource(fileset="fs-b"),
            output=BuildOutput(repository="team/app", tag="prod"),
        )
        spec = _compile(_set(staging, prod), config=_config())
        push = PushStepConfig.model_validate(spec.steps[2].config)
        targets = [set(image.refs) for image in push.images]
        assert not (targets[0] & targets[1]), f"two images share a push target: {targets[0] & targets[1]}"


class TestPublishing:
    def test_every_image_gets_the_callers_tag_then_the_system_tag(self) -> None:
        spec = _compile(_set(), config=_config())
        push = PushStepConfig.model_validate(spec.steps[2].config)
        assert push.images[0].refs == [
            "reg.example.com/default/team/main:v1",
            f"reg.example.com/default/team/main:{SYSTEM_TAG}",
        ]

    def test_a_spec_that_names_no_output_pushes_the_system_tag_only(self) -> None:
        spec = _compile(_set(BuildSpec(name="main", source=FileSetSource(fileset="fs-a"))), config=_config())
        push = PushStepConfig.model_validate(spec.steps[2].config)
        assert push.images[0].refs == [f"reg.example.com/default/demo/main:{SYSTEM_TAG}"]

    def test_every_image_goes_to_the_deployments_registry_under_its_workspace(self) -> None:
        spec = _compile(_set(_spec("main"), _spec("verifier")), config=_config())
        push = PushStepConfig.model_validate(spec.steps[2].config)
        assert push.registry == "reg.example.com"
        assert all(ref.startswith("reg.example.com/default/") for image in push.images for ref in image.refs)

    def test_the_registry_is_reached_over_https_unless_configured_as_http(self) -> None:
        https = _compile(_set(), config=_config())
        assert PushStepConfig.model_validate(https.steps[2].config).plain_http is False
        plain = _compile(_set(), config=_config(registry="http://registry.local:5000"))
        push = PushStepConfig.model_validate(plain.steps[2].config)
        assert (push.registry, push.plain_http) == ("registry.local:5000", True)


class TestNoConfigCarriesAPath:
    def test_no_step_config_mentions_the_work_mount_or_a_sandbox_path(self) -> None:
        spec = _compile(_set(_spec("main", context_path="tests")), config=_config())
        for step in spec.steps:
            serialized = str(step.config)
            assert WORK_MOUNT not in serialized, f"{step.name} config carries the work mount"
            assert "/out" not in serialized and "context/" not in serialized, f"{step.name} config carries a path"
