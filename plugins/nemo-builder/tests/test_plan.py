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
        "default_registry": "reg.example.com",
        "push_secret": "my-reg-secret",
        "signing_key": "k8s://nmp-builds/cosign-key",
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
    registry: str | None = None,
) -> BuildSpec:
    return BuildSpec(
        name=name,
        source=FileSetSource(fileset=fileset, context_path=context_path),
        output=BuildOutput(registry=registry, repository=repository or f"team/{name}", tag=tag),
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
    """A registry host and a repository path are different things.

    Conflating them reads fine in a log and produces `https://host/project/repo/v2/...` the
    moment a registry client tries to resolve the result.
    """

    def test_a_registry_with_a_path_is_rejected_at_config_time(self) -> None:
        with pytest.raises(ValidationError, match="without a path"):
            _config(default_registry="us-central1-docker.pkg.dev/proj/repo")

    def test_the_prefix_goes_on_the_repository(self) -> None:
        image = _plan(repository_prefix="proj/artifacts").images[0]
        assert (image.registry, image.repository) == ("reg.example.com", "proj/artifacts/team/main")

    def test_a_spec_naming_its_own_registry_gets_no_prefix(self) -> None:
        """It named a full destination; prepending a deployment default would corrupt it."""
        image = _plan(_set(_spec("main", registry="other.example.com")), repository_prefix="proj/artifacts").images[0]
        assert (image.registry, image.repository) == ("other.example.com", "team/main")

    def test_no_registry_anywhere_is_the_deployments_problem(self) -> None:
        with pytest.raises(BuildCompileError, match="registry"):
            _plan(default_registry=None)


class TestContradictoryRequests:
    def test_two_specs_publishing_the_same_reference_are_rejected(self) -> None:
        """The caller's own tag could point at only one of them. A 400 at submit, not a 409."""
        build_set = _set(
            _spec("a", fileset="fs-a", repository="team/x"), _spec("b", fileset="fs-b", repository="team/x")
        )
        with pytest.raises(ValueError, match="both publish") as caught:
            _plan(build_set)
        assert not isinstance(caught.value, BuildCompileError)

    def test_an_explicit_default_registry_and_an_omitted_one_are_the_same_destination(self) -> None:
        build_set = _set(
            _spec("a", repository="team/x"),
            _spec("b", repository="team/x", registry="reg.example.com"),
        )
        with pytest.raises(ValueError, match="both publish"):
            _plan(build_set)


class TestADeploymentThatCannotBuild:
    """At submit, not twenty minutes later in a pod."""

    def test_missing_signing_key(self) -> None:
        with pytest.raises(BuildCompileError, match="signing_key"):
            _plan(signing_key=None)

    def test_missing_push_secret(self) -> None:
        with pytest.raises(BuildCompileError, match="push_secret"):
            _plan(push_secret=None)

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
