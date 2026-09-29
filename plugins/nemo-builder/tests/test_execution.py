# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution backend's declarations: what it refuses, and where it publishes."""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from nemo_builder_plugin.backend import Backend, BackendRefused
from nemo_builder_plugin.backends import load_backend
from nemo_builder_plugin.config import BuilderConfig
from nemo_builder_plugin.execution import ExecutionBackend
from nemo_builder_plugin.plan import BuildPlan
from nemo_builder_plugin.schema import BuildOutput, BuildSet, BuildSpec, FileSetSource
from nemo_builder_plugin.signing import key_fingerprint

KEY = ec.generate_private_key(ec.SECP256R1())
PUBLIC_KEY = KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def _config(**overrides: object) -> BuilderConfig:
    settings: dict[str, object] = {
        "registry": "reg.example.com",
        "sandbox_image": "kaniko.example.com/executor:debug",
        "credential_broker": "http://broker.example:8080",
        "signing_public_key": PUBLIC_KEY.decode(),
    }
    return BuilderConfig.model_validate(settings | overrides)


def _spec(name: str = "main", *, repository: str | None = "team/app") -> BuildSpec:
    output = BuildOutput(repository=repository, tag="v1") if repository else None
    return BuildSpec(name=name, source=FileSetSource(fileset="fs-a"), output=output)


def _plan(*specs: BuildSpec, workspace: str = "default") -> BuildPlan:
    build_set = BuildSet(name="demo", revision=2, build_specs=list(specs) or [_spec()])
    return BuildPlan.resolve(build_set, workspace=workspace)


def test_the_deployment_runs_the_execution_backend() -> None:
    backend: Backend = load_backend(_config())
    assert isinstance(backend, ExecutionBackend)


class TestADeploymentThatCannotBuild:
    """Refused at submit, not twenty minutes later in a pod."""

    @pytest.mark.parametrize(
        ("missing", "named"),
        [
            ("registry", "builder.registry"),
            ("sandbox_image", "sandbox_image"),
            ("credential_broker", "credential_broker"),
            ("signing_public_key", "signing_public_key"),
        ],
    )
    def test_a_setting_nothing_can_default(self, missing: str, named: str) -> None:
        with pytest.raises(BackendRefused, match=named):
            ExecutionBackend(_config(**{missing: None})).check(_plan())

    def test_a_configured_deployment_honors_every_request(self) -> None:
        ExecutionBackend(_config()).check(_plan(_spec("a"), _spec("b", repository=None)))


class TestDestination:
    def test_a_named_output_lands_under_the_workspace(self) -> None:
        backend = ExecutionBackend(_config())
        placed = _plan().with_destinations(backend.destination).images[0].placed
        assert (placed.registry, placed.repository, placed.system_tag) == (
            "reg.example.com",
            "default/team/app",
            "default--demo-2-0",
        )

    def test_an_unnamed_output_lands_at_set_and_spec(self) -> None:
        backend = ExecutionBackend(_config())
        placed = _plan(_spec("verifier", repository=None)).with_destinations(backend.destination).images[0].placed
        assert placed.repository == "default/demo/verifier"

    def test_the_prefix_goes_ahead_of_the_workspace(self) -> None:
        backend = ExecutionBackend(_config(repository_prefix="proj/artifacts"))
        placed = _plan().with_destinations(backend.destination).images[0].placed
        assert placed.repository == "proj/artifacts/default/team/app"

    def test_two_workspaces_asking_for_one_repository_get_two(self) -> None:
        """The workspace component is the isolation: the broker grants only under it."""
        backend = ExecutionBackend(_config())
        a = _plan(workspace="team-a").with_destinations(backend.destination).images[0].placed
        b = _plan(workspace="team-b").with_destinations(backend.destination).images[0].placed
        assert a.repository != b.repository

    def test_a_workspace_that_cannot_be_a_repository_component_is_refused(self) -> None:
        """NAME_PATTERN admits spellings a repository path does not. A 400, before any row."""
        backend = ExecutionBackend(_config())
        with pytest.raises(ValueError, match="repository"):
            _plan(workspace="Team@A").with_destinations(backend.destination)

    def test_every_image_has_its_own_system_tag(self) -> None:
        backend = ExecutionBackend(_config())
        plan = _plan(_spec("a", repository=None), _spec("b", repository=None)).with_destinations(backend.destination)
        assert [image.placed.system_tag for image in plan.images] == ["default--demo-2-0", "default--demo-2-1"]


class TestSignaturePolicy:
    def test_signatures_are_verified_against_the_deployments_key(self) -> None:
        assert ExecutionBackend(_config()).signature_policy().trust_root == key_fingerprint(KEY.public_key())


class TestEntitlements:
    def test_only_the_push_step_is_entitled_to_anything(self) -> None:
        """fetch reads Files as the submitter; build orchestrates a sandbox that must reach nothing."""
        entitlements = ExecutionBackend(_config()).entitlements()
        assert list(entitlements) == ["push"]
        push = entitlements["push"]
        assert (push.step, push.profile, push.service_account) == ("push", "build-push", "nhx-build-push")
        assert push.destinations == ("pull", "push") and push.sign

    def test_the_push_identity_follows_the_config(self) -> None:
        push = ExecutionBackend(_config(push_profile="p", push_service_account="sa")).entitlements()["push"]
        assert (push.profile, push.service_account) == ("p", "sa")
