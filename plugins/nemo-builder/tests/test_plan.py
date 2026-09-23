# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolving a request against a deployment: every check that can reject a submit lives here.

The compiler and the row builder both read the plan these tests produce, so a property asserted
here holds for the job and the rows at once.
"""

from __future__ import annotations

import pytest
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.plan import BuildCompileError, BuildPlan
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.steps import ContextSource
from pydantic import ValidationError

WORKSPACE = "default"


def _config(**overrides: object) -> BuilderConfig:
    settings: dict[str, object] = {
        "registry": "reg.example.com",
        "push_credential_secret": "registry-push-credential",
        "signing_key": "k8s://nhx-builds/cosign-key",
    }
    settings.update(overrides)
    return BuilderConfig.model_validate(settings)


def _spec(
    name: str,
    *,
    fileset: str = "fs-a",
    context_path: str | None = None,
    repository: str | None = None,
    tag: str = "v1",
) -> BuildSpec:
    return BuildSpec(
        name=name,
        source=FileSetSource(fileset=fileset, context_path=context_path),
        output=BuildOutput(repository=repository or f"team/{name}", tag=tag),
    )


def _set(*specs: BuildSpec) -> BuildSet:
    return BuildSet(name="demo", revision=1, build_specs=list(specs) or [_spec("main")])


def _plan(build_set: BuildSet | None = None, **config: object) -> BuildPlan:
    return BuildPlan.resolve(build_set or _set(), config=_config(**config), workspace=WORKSPACE)


class TestIdentity:
    def test_names_are_deterministic_from_the_request(self) -> None:
        """Which is what lets a losing racer adopt the winner's rows instead of duplicating them."""
        plan = _plan(_set(_spec("main"), _spec("verifier")))
        assert plan.job_name == "demo-1"
        assert [i.name for i in plan.images] == ["demo-1-0", "demo-1-1"]

    def test_every_image_has_its_own_system_tag(self) -> None:
        """Per image, not per set: two images in one repository must not push one tag."""
        plan = _plan(
            _set(
                _spec("staging", repository="team/app", tag="staging"), _spec("prod", repository="team/app", tag="prod")
            )
        )
        assert [i.system_tag for i in plan.images] == ["default--demo-1-0", "default--demo-1-1"]
        assert plan.images[0].system_ref != plan.images[1].system_ref


class TestDestinations:
    """One registry, and a path the compiler composes rather than one a caller chooses."""

    def test_every_image_goes_to_the_deployments_registry_under_its_workspace(self) -> None:
        image = _plan().images[0]
        assert (image.registry, image.repository) == ("reg.example.com", "default/team/main")

    def test_the_prefix_goes_ahead_of_the_workspace(self) -> None:
        image = _plan(repository_prefix="proj/artifacts").images[0]
        assert image.repository == "proj/artifacts/default/team/main"

    def test_two_workspaces_asking_for_one_repository_get_two(self) -> None:
        """The isolation between tenants. One credential writes the whole registry, so the path
        is the only thing keeping one workspace from moving another's tags."""
        ours = BuildPlan.resolve(_set(), config=_config(), workspace="team-a").images[0]
        theirs = BuildPlan.resolve(_set(), config=_config(), workspace="team-b").images[0]
        assert (ours.repository, theirs.repository) == ("team-a/team/main", "team-b/team/main")

    def test_a_workspace_that_cannot_be_a_repository_component_is_refused(self) -> None:
        """`NAME_PATTERN` admits `team..a`; a repository component does not. A 400, not a 409."""
        with pytest.raises(ValueError, match="repository") as caught:
            BuildPlan.resolve(_set(), config=_config(), workspace="team..a")
        assert not isinstance(caught.value, BuildCompileError)


class TestConfiguration:
    def test_a_registry_with_a_path_is_rejected(self) -> None:
        """A host and a repository path are different things. Conflating them reads fine in a log
        and produces `https://host/project/repo/v2/...` the moment a registry client resolves it."""
        with pytest.raises(ValidationError, match="without a path"):
            _config(registry="us-central1-docker.pkg.dev/proj/repo")

    def test_the_prefix_is_normalized_and_checked(self) -> None:
        assert _config(repository_prefix="/proj/artifacts/").repository_prefix == "proj/artifacts"
        with pytest.raises(ValidationError, match="repository"):
            _config(repository_prefix="proj/../elsewhere")

    def test_the_reconcilers_credential_can_come_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """How deploy/local/platform.yaml supplies it from a Kubernetes Secret, keeping it out of
        the ConfigMap -- and a guard that `registry` and `registry_username` do not collide
        under the `_` nested-env delimiter."""
        monkeypatch.setenv("NEMO_BUILDER_REGISTRY_USERNAME", "reader")
        monkeypatch.setenv("NEMO_BUILDER_REGISTRY_PASSWORD", "s3cret")
        config = BuilderConfig(registry="reg.example.com")
        assert (config.registry, config.registry_username, config.registry_password) == (
            "reg.example.com",
            "reader",
            "s3cret",
        )


class TestContradictoryRequests:
    def test_two_specs_publishing_the_same_reference_are_rejected(self) -> None:
        """The caller's own tag could point at only one of them. A 400 at submit, not a 409."""
        build_set = _set(
            _spec("a", fileset="fs-a", repository="team/x"), _spec("b", fileset="fs-b", repository="team/x")
        )
        with pytest.raises(ValueError, match="both publish") as caught:
            _plan(build_set)
        assert not isinstance(caught.value, BuildCompileError)


class TestADeploymentThatCannotBuild:
    """At submit, not twenty minutes later in a pod."""

    def test_missing_signing_key(self) -> None:
        with pytest.raises(BuildCompileError, match="signing_key"):
            _plan(signing_key=None)

    def test_missing_registry(self) -> None:
        with pytest.raises(BuildCompileError, match="builder.registry"):
            _plan(registry=None)

    def test_missing_push_credential_secret(self) -> None:
        with pytest.raises(BuildCompileError, match="push_credential_secret"):
            _plan(push_credential_secret=None)

    def test_the_kill_switch_refuses(self) -> None:
        with pytest.raises(BuildCompileError, match="execution_enabled"):
            _plan(execution_enabled=False)

    def test_the_kill_switch_is_checked_before_the_request(self) -> None:
        """A disabled deployment says so, whatever was asked of it -- even a malformed request."""
        duplicate = _set(_spec("a", repository="team/x"), _spec("b", fileset="fs-b", repository="team/x"))
        with pytest.raises(BuildCompileError, match="execution_enabled"):
            _plan(duplicate, execution_enabled=False)


class TestGrouping:
    """One sandbox per distinct source, so a Dockerfile never sees another source's context."""

    def test_images_sharing_a_source_share_a_group(self) -> None:
        plan = _plan(_set(_spec("main"), _spec("verifier"), _spec("other", fileset="fs-b")))
        groups = plan.groups()
        assert [source for source, _ in groups] == [ContextSource(fileset="fs-a"), ContextSource(fileset="fs-b")]
        assert [[i.name for i in images] for _, images in groups] == [["demo-1-0", "demo-1-1"], ["demo-1-2"]]

    def test_context_path_splits_a_shared_fileset(self) -> None:
        """The key is (fileset, context_path), not the fileset alone -- so a Dockerfile under
        `tests/` never sees `environment/`, even though both came from one upload."""
        plan = _plan(_set(_spec("env", context_path="environment"), _spec("tests", context_path="tests")))
        assert len(plan.groups()) == 2

    def test_an_empty_context_path_is_the_whole_fileset(self) -> None:
        plan = _plan(_set(_spec("a", context_path=""), _spec("b")))
        assert [source for source, _ in plan.groups()] == [ContextSource(fileset="fs-a", context_path=None)]

    def test_group_order_follows_first_appearance(self) -> None:
        plan = _plan(_set(_spec("b", fileset="fs-b"), _spec("a", fileset="fs-a"), _spec("b2", fileset="fs-b")))
        assert [source.fileset for source, _ in plan.groups()] == ["fs-b", "fs-a"]
