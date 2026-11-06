# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from config import Step
from job_profile import ProfileCandidate, ProfileError, ProfileFacts, choose_profile, runnable_steps, validate_profile


def _facts(backend: str, image: str | None, active: int) -> ProfileFacts:
    return ProfileFacts(
        name="default",
        provider="cpu",
        backend=backend,
        task_image=image,
        ttl_seconds_before_active=60,
        ttl_seconds_active=active,
    )


def _candidate(
    name: str,
    provider: str,
    backend: str,
    image: str | None = None,
    active: int = 3600,
) -> ProfileCandidate:
    return ProfileCandidate(
        name=name,
        provider=provider,
        backend=backend,
        task_image=image,
        ttl_seconds_before_active=60,
        ttl_seconds_active=active,
    )


def test_kubernetes_profile_with_stock_image_is_accepted() -> None:
    step = Step("idle-01", "idle", 1, 300, None, None)
    validate_profile(_facts("kubernetes_job", None, 3600), [step])
    validate_profile(_facts("kubernetes_job", "registry.example/nhx-tasks:latest", 3600), [step])
    validate_profile(_facts("docker", None, 3600), [step])


def test_other_backends_are_rejected() -> None:
    step = Step("exit-01", "exit", 1, None, None, None)
    with pytest.raises(ProfileError, match="kubernetes_job or docker"):
        validate_profile(_facts("volcano_job", None, 3600), [step])


def test_custom_image_is_rejected() -> None:
    step = Step("exit-01", "exit", 1, None, None, None)
    with pytest.raises(ProfileError, match="nhx-tasks"):
        validate_profile(_facts("kubernetes_job", "python:3.12", 3600), [step])


def test_duration_beyond_active_ttl_is_rejected() -> None:
    step = Step("idle-01", "idle", 1, 300, None, None)
    with pytest.raises(ProfileError, match="ttl_seconds_active"):
        validate_profile(_facts("kubernetes_job", None, 60), [step])


def test_default_cpu_profile_is_selected_ahead_of_gpu() -> None:
    facts = choose_profile(
        [
            _candidate("default", "gpu", "kubernetes_job"),
            _candidate("default", "cpu", "kubernetes_job"),
            _candidate("default", "gpu_distributed", "volcano_job"),
        ],
        "default",
    )
    assert (facts.provider, facts.backend) == ("cpu", "kubernetes_job")


def test_docker_default_is_accepted_when_it_is_the_cpu_profile() -> None:
    facts = choose_profile(
        [
            _candidate("default", "cpu", "docker"),
            _candidate("default", "gpu", "docker"),
        ],
        "default",
    )
    assert (facts.provider, facts.backend) == ("cpu", "docker")


def test_subprocess_default_is_rejected_and_lists_other_profiles() -> None:
    with pytest.raises(ProfileError, match=r"cpu/gpu \(docker\)") as caught:
        choose_profile(
            [
                _candidate("default", "subprocess", "subprocess"),
                _candidate("default", "cpu", "docker"),
                _candidate("gpu", "cpu", "docker"),
            ],
            "default",
        )
    assert "subprocess" in str(caught.value)
    assert "--job-profile" in str(caught.value)


def test_provider_and_profile_selects_that_pair() -> None:
    candidates = [
        _candidate("gpu", "cpu", "docker"),
        _candidate("gpu", "gpu", "docker"),
        _candidate("default", "subprocess", "subprocess"),
    ]
    facts = choose_profile(candidates, "cpu/gpu")
    assert (facts.provider, facts.name, facts.backend) == ("cpu", "gpu", "docker")
    gpu = choose_profile(candidates, "gpu/gpu")
    assert (gpu.provider, gpu.name) == ("gpu", "gpu")


def test_unknown_provider_profile_lists_registered_names() -> None:
    with pytest.raises(ProfileError, match="cpu/default"):
        choose_profile([_candidate("default", "cpu", "kubernetes_job")], "gpu")


def test_override_selects_the_named_docker_profile() -> None:
    facts = choose_profile(
        [
            _candidate("default", "subprocess", "subprocess"),
            _candidate("gpu", "cpu", "docker", "registry.example/nhx-tasks:latest"),
        ],
        "gpu",
    )
    assert facts.name == "gpu"
    assert facts.backend == "docker"


def test_non_stock_image_on_the_chosen_profile_is_rejected() -> None:
    with pytest.raises(ProfileError, match="python:3.12"):
        choose_profile([_candidate("gpu", "cpu", "docker", "python:3.12")], "gpu")


def test_auth_steps_are_skipped_when_auth_is_disabled() -> None:
    exit_step = Step("exit-01", "exit", 1, None, None, None)
    auth_step = Step("auth-01", "hammer", 1, 60, "auth", 5)
    kept, skipped = runnable_steps([(exit_step, False), (auth_step, False)], auth_enabled=False, only=None)
    assert kept == [(exit_step, False)]
    assert skipped == ["auth-01"]


def test_explicit_auth_step_fails_when_auth_is_disabled() -> None:
    auth_step = Step("auth-01", "hammer", 1, 60, "auth", 5)
    with pytest.raises(ProfileError, match="auth-01"):
        runnable_steps([(auth_step, True)], auth_enabled=False, only="auth-01")


def test_auth_steps_run_when_auth_is_enabled() -> None:
    auth_step = Step("auth-01", "hammer", 1, 60, "auth", 5)
    kept, skipped = runnable_steps([(auth_step, False)], auth_enabled=True, only=None)
    assert kept == [(auth_step, False)]
    assert skipped == []
