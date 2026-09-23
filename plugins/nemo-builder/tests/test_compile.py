# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the compiler must produce, asserted without a cluster.

The compiler is a pure function precisely so this file can exist. Everything the security
argument rests on is a property of the document it emits -- which step holds a credential, which
step can reach the work volume, what the sandbox is told -- and all of it is checkable here, in
milliseconds, rather than by reading pod specs off a cluster.
"""

from __future__ import annotations

import pytest
from nemo_builder_plugin.compile import (
    PUSH_CREDENTIAL_ENVVAR,
    WORK_MOUNT,
    BuildCompileError,
    compile_build_set,
    image_name_for,
    job_name_for,
    resolve_destinations,
)
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.steps import PushStepConfig, SuperviseStepConfig
from nemo_builder_plugin.submit import system_tags_for
from nemo_platform_plugin.jobs.constants import PERSISTENT_JOB_STORAGE_PATH_ENVVAR
from nemo_platform_plugin.jobs.providers import CPUExecutionProvider
from nemo_platform_plugin.jobs.spec import PlatformJobSpec
from pydantic import ValidationError

WORKSPACE = "default"
#: The system tag of the first image of the default `_set()`. Tags are per IMAGE, not per set.
SYSTEM_TAG = "default--demo-1-0"


def _tags(build_set: BuildSet) -> list[str]:
    return system_tags_for(build_set, WORKSPACE)


def _compile(build_set: BuildSet, *, config: BuilderConfig) -> PlatformJobSpec:
    return compile_build_set(build_set, config=config, system_tags=_tags(build_set))


def _config(
    *,
    default_registry: str | None = "reg.example.com",
    repository_prefix: str = "",
    sandbox_cpu: str = "2",
    sandbox_memory: str = "8Gi",
    push_secret: str | None = "my-reg-secret",
    signing_key: str | None = "k8s://nmp-builds/cosign-key",
    execution_enabled: bool = True,
) -> BuilderConfig:
    return BuilderConfig(
        default_registry=default_registry,
        repository_prefix=repository_prefix,
        sandbox_cpu=sandbox_cpu,
        sandbox_memory=sandbox_memory,
        push_secret=push_secret,
        signing_key=signing_key,
        execution_enabled=execution_enabled,
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
        """The trust split is three strings; each resolves to a different ServiceAccount."""
        spec = _compile(_set(), config=_config())
        profiles = [s.executor.profile for s in spec.steps]
        assert profiles == ["build-fetch", "build-control", "build-push"]
        assert len(set(profiles)) == 3

    def test_each_step_runs_its_own_binary(self) -> None:
        spec = _compile(_set(), config=_config())
        executors = [s.executor for s in spec.steps]
        # Narrowed rather than suppressed: `Provider` is a union whose subprocess arm has no
        # container at all, and a build step landing on that arm would be a real bug.
        assert all(isinstance(e, CPUExecutionProvider) for e in executors)
        assert [e.container.command for e in executors if isinstance(e, CPUExecutionProvider)] == [
            ["nmp-build", "fetch"],
            ["nmp-build", "supervise"],
            ["nmp-build", "push"],
        ]


class TestItSurvivesTheWire:
    """The compiled spec has to validate after the client serializes it, not just in memory."""

    def test_the_compiled_spec_survives_exclude_unset(self) -> None:
        """The Jobs client serializes with `exclude_unset`, which DROPS defaulted fields.

        `CPUExecutionProvider.provider` is `Literal["cpu"] = "cpu"` -- the discriminator of the
        executor union. Leaving it defaulted means it never reaches the wire, and the server
        answers 422 "Unable to extract tag using discriminator 'provider'". That is exactly what
        the first real submit returned, and an in-memory assertion cannot see it.
        """
        spec = _compile(_set(), config=_config())
        on_the_wire = spec.model_dump(exclude_unset=True)

        for step in on_the_wire["steps"]:
            assert step["executor"].get("provider") == "cpu", (
                f"step {step['name']!r} lost its executor discriminator when serialized"
            )

        # And it round-trips back into the type the Jobs API validates against.
        PlatformJobSpec.model_validate(on_the_wire)


class TestTheCredentialAppearsOnce:
    """Requirement 7, as a property of the emitted document."""

    def test_only_the_push_step_receives_the_credential(self) -> None:
        spec = _compile(_set(), config=_config())
        fetch, build, push = spec.steps

        holders = [step.name for step in spec.steps for env in (step.environment or []) if env.from_secret is not None]
        assert holders == ["push"]

        assert any(e.name == PUSH_CREDENTIAL_ENVVAR for e in push.environment or [])
        assert not any(e.from_secret for e in fetch.environment or [])
        assert build.environment is None or not any(e.from_secret for e in build.environment)

    def test_the_secret_value_never_enters_the_spec(self) -> None:
        """`from_secret` is a NAME. The launcher resolves it in-pod, as the submitter."""
        spec = _compile(_set(), config=_config())
        assert spec.secrets is not None
        assert [s.name for s in spec.secrets] == ["my-reg-secret"]
        assert all(s.value is None for s in spec.secrets)


class TestTheBuildStepCannotReachTheWorkVolume:
    """Absence is the control, so absence is what gets asserted."""

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
        """The sandbox does not push, so its orchestrator is not told where anything goes."""
        spec = _compile(_set(), config=_config())
        serialized = str(spec.steps[1].config)
        assert "reg.example.com" not in serialized
        assert "my-reg-secret" not in serialized
        assert SYSTEM_TAG not in serialized
        assert "cosign" not in serialized

    def test_the_sandbox_carries_no_service_account(self) -> None:
        spec = _compile(_set(), config=_config())
        sandbox = SuperviseStepConfig.model_validate(spec.steps[1].config).sandbox
        assert sandbox.service_account == ""

    def test_the_sandbox_is_sized_by_the_deployment_not_the_request(self) -> None:
        """A caller cannot make its build bigger by asking; there is no field for it to ask with."""
        config = _config(sandbox_cpu="1", sandbox_memory="2Gi")
        spec = _compile(_set(), config=config)
        sandbox = SuperviseStepConfig.model_validate(spec.steps[1].config).sandbox
        assert (sandbox.cpu, sandbox.memory) == ("1", "2Gi")

    def test_the_sandbox_uses_public_dns_not_cluster_dns(self) -> None:
        """Measured: cluster DNS here is link-local, and re-allowing it reopens the Pod CIDR."""
        spec = _compile(_set(), config=_config())
        sandbox = SuperviseStepConfig.model_validate(spec.steps[1].config).sandbox
        assert sandbox.dns_nameservers == ["8.8.8.8", "1.1.1.1"]


class TestGrouping:
    def test_specs_sharing_a_source_get_one_sandbox(self) -> None:
        spec = _compile(_set(_spec("main"), _spec("verifier")), config=_config())
        groups = SuperviseStepConfig.model_validate(spec.steps[1].config).groups
        assert len(groups) == 1
        assert [i.image for i in groups[0].images] == ["demo-1-0", "demo-1-1"]

    def test_distinct_sources_get_a_sandbox_each_mounting_only_its_own(self) -> None:
        spec = _compile(
            _set(_spec("a", context_path="environment"), _spec("b", context_path="tests")), config=_config()
        )
        groups = SuperviseStepConfig.model_validate(spec.steps[1].config).groups
        assert len(groups) == 2
        assert groups[0].context_sub_path == "context/fs-a/environment"
        assert groups[1].context_sub_path == "context/fs-a/tests"

    def test_fetch_downloads_a_shared_source_once(self) -> None:
        spec = _compile(_set(_spec("main"), _spec("verifier")), config=_config())
        assert len(spec.steps[0].config["sources"]) == 1

    def test_fetch_downloads_each_distinct_subtree(self) -> None:
        spec = _compile(
            _set(_spec("a", context_path="environment"), _spec("b", context_path="tests")), config=_config()
        )
        assert [s["context_path"] for s in spec.steps[0].config["sources"]] == ["environment", "tests"]

    def test_a_root_request_absorbs_subtree_requests_for_the_same_fileset(self) -> None:
        """`context_path` selects a subtree, so the root download already contains it.

        Emitting both would fetch the overlap twice and write it twice into one directory. The
        sandbox mounts still differ -- each group gets its own subPath -- so collapsing the
        download changes nothing about what a Dockerfile can see.
        """
        spec = _compile(_set(_spec("whole"), _spec("subtree", context_path="tests")), config=_config())
        sources = spec.steps[0].config["sources"]
        assert sources == [{"fileset": "fs-a", "context_path": None}]

        # ...and the two groups still mount different subPaths.
        groups = SuperviseStepConfig.model_validate(spec.steps[1].config).groups
        assert [g.context_sub_path for g in groups] == ["context/fs-a", "context/fs-a/tests"]

    def test_a_root_request_on_a_different_fileset_absorbs_nothing(self) -> None:
        spec = _compile(
            _set(_spec("whole", fileset="fs-a"), _spec("subtree", fileset="fs-b", context_path="tests")),
            config=_config(),
        )
        sources = spec.steps[0].config["sources"]
        assert sources == [
            {"fileset": "fs-a", "context_path": None},
            {"fileset": "fs-b", "context_path": "tests"},
        ]


class TestRegistryIsAHostNotAPath:
    """A registry host and a repository path are different things.

    Conflating them reads fine in a log and produces `https://host/project/repo/v2/...` the
    moment a registry client tries to resolve the result -- which is exactly how this was found,
    on the first real end-to-end submit.
    """

    def test_a_registry_with_a_path_is_rejected_at_config_time(self) -> None:
        with pytest.raises(ValidationError, match="without a path"):
            _config(default_registry="us-central1-docker.pkg.dev/proj/repo")

    def test_the_prefix_goes_on_the_repository(self) -> None:
        config = _config(repository_prefix="proj/artifacts")
        registry, repository = resolve_destinations(_set(), config)[0]
        assert registry == "reg.example.com"
        assert repository == "proj/artifacts/team/main"

    def test_a_spec_naming_its_own_registry_gets_no_prefix(self) -> None:
        """It named a full destination; prepending a deployment default would corrupt it."""
        spec = _spec("main")
        spec.output.registry = "other.example.com"
        registry, repository = resolve_destinations(_set(spec), _config(repository_prefix="proj/artifacts"))[0]
        assert (registry, repository) == ("other.example.com", "team/main")


class TestEachImageHasItsOwnIdentity:
    """The collision the adversarial review found, as a property of the compiled document."""

    def test_specs_sharing_a_repository_push_distinct_system_tags(self) -> None:
        """Two Dockerfiles, one repository, different caller tags -- `staging` and `prod`.

        With a per-set system tag both images pushed `team/app:<system-tag>`, the second push
        replaced the first, and both rows resolved one digest.
        """
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
        targets = [set(image.tags) for image in push.images]
        assert not (targets[0] & targets[1]), f"two images share a push target: {targets[0] & targets[1]}"

    def test_two_specs_publishing_the_same_reference_are_rejected(self) -> None:
        """The caller's own tag could point at only one of them. A 400 at submit, not a 409."""
        a = BuildSpec(name="a", source=FileSetSource(fileset="fs-a"), output=BuildOutput(repository="team/x", tag="v1"))
        b = BuildSpec(name="b", source=FileSetSource(fileset="fs-b"), output=BuildOutput(repository="team/x", tag="v1"))
        with pytest.raises(ValueError, match="both publish") as caught:
            _compile(_set(a, b), config=_config())
        assert not isinstance(caught.value, BuildCompileError)

    def test_an_explicit_default_registry_and_an_omitted_one_are_the_same_destination(self) -> None:
        a = BuildSpec(name="a", source=FileSetSource(fileset="fs-a"), output=BuildOutput(repository="team/x", tag="v1"))
        b = BuildSpec(
            name="b",
            source=FileSetSource(fileset="fs-b"),
            output=BuildOutput(registry="reg.example.com", repository="team/x", tag="v1"),
        )
        with pytest.raises(ValueError, match="both publish"):
            _compile(_set(a, b), config=_config())

    def test_the_compiler_refuses_duplicate_or_missing_system_tags(self) -> None:
        """Checked rather than trusted: the failure it prevents -- two rows, one digest -- is silent."""
        build_set = _set(_spec("a"), _spec("b"))
        with pytest.raises(ValueError, match="distinct"):
            compile_build_set(build_set, config=_config(), system_tags=["t", "t"])
        with pytest.raises(ValueError, match="expected 2"):
            compile_build_set(build_set, config=_config(), system_tags=["t"])


class TestPublishing:
    def test_every_image_gets_the_callers_tag_and_the_system_tag(self) -> None:
        spec = _compile(_set(), config=_config())
        push = PushStepConfig.model_validate(spec.steps[2].config)
        assert push.images[0].tags == [
            "reg.example.com/team/main:v1",
            f"reg.example.com/team/main:{SYSTEM_TAG}",
        ]

    def test_a_spec_may_override_the_default_registry(self) -> None:
        spec_with_registry = _spec("main")
        spec_with_registry.output.registry = "other.example.com"
        spec = _compile(_set(spec_with_registry), config=_config())
        push = PushStepConfig.model_validate(spec.steps[2].config)
        assert all(t.startswith("other.example.com/") for t in push.images[0].tags)

    def test_the_credential_is_bound_to_the_operators_registry_not_a_callers(self) -> None:
        """A spec naming its own registry changes where bytes go, never where the credential goes.
        Deriving the credential's host from destinations would send it to any host a caller names."""
        spec_with_registry = _spec("main")
        spec_with_registry.output.registry = "attacker.example.com"
        spec = _compile(_set(spec_with_registry), config=_config())
        push = PushStepConfig.model_validate(spec.steps[2].config)
        assert push.credential_registry == "reg.example.com"

    def test_layout_paths_line_up_with_what_the_sandbox_writes(self) -> None:
        """`supervise`'s output IS `push`'s input; a mismatch here is silent until runtime."""
        spec = _compile(_set(_spec("main"), _spec("v")), config=_config())
        groups = SuperviseStepConfig.model_validate(spec.steps[1].config).groups
        push = PushStepConfig.model_validate(spec.steps[2].config)
        written = {i.image: i.layout.removeprefix("/out/") for g in groups for i in g.images}
        read = {i.image: i.layout.removeprefix(f"{WORK_MOUNT}/out/") for i in push.images}
        assert written == read


class TestUnconfiguredDeploymentFailsTheCompile:
    """At compile time, not twenty minutes later in a pod."""

    def test_missing_signing_key(self) -> None:
        with pytest.raises(BuildCompileError, match="signing_key"):
            _compile(_set(), config=_config(signing_key=None))

    def test_missing_push_secret(self) -> None:
        with pytest.raises(BuildCompileError, match="push_secret"):
            _compile(_set(), config=_config(push_secret=None))

    def test_no_registry_anywhere(self) -> None:
        with pytest.raises(BuildCompileError, match="registry"):
            _compile(_set(), config=_config(default_registry=None))

    def test_the_kill_switch_refuses_the_compile(self) -> None:
        with pytest.raises(BuildCompileError, match="execution_enabled"):
            _compile(_set(), config=_config(execution_enabled=False))


class TestNaming:
    def test_names_are_deterministic_from_the_request(self) -> None:
        """Which is what lets a losing racer adopt the winner's rows instead of duplicating them."""
        build_set = _set(_spec("main"), _spec("verifier"))
        assert job_name_for(build_set) == "demo-1"
        assert image_name_for("demo-1", 1) == "demo-1-1"
