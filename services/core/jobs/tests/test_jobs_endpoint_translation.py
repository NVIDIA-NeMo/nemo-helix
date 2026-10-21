# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi import HTTPException
from nhx.core.jobs.api.v2.jobs.endpoints import translate_cpu_container_steps_to_subprocess, validate_job_spec
from nhx.core.jobs.app.profiles import ExecutionProfileT
from nhx.core.jobs.app.providers import ContainerSpec, CPUExecutionProvider, SubprocessExecutionProvider
from nhx.core.jobs.app.schemas import HelixJobSpec, HelixJobStepSpec
from nhx.core.jobs.controllers.backends.docker import DockerJobExecutionProfile, DockerJobExecutionProfileConfig
from nhx.core.jobs.controllers.backends.kubernetes import KubernetesJobExecutionProfile


def _cpu_step(name: str, profile: str = "default") -> HelixJobStepSpec:
    return HelixJobStepSpec(
        name=name,
        executor=CPUExecutionProvider(
            provider="cpu",
            profile=profile,
            container=ContainerSpec(image="image", entrypoint=["python", "-m"], command=["task"]),
        ),
    )


def test_translate_cpu_container_steps_to_subprocess_uses_explicit_compat_profiles() -> None:
    spec = HelixJobSpec(steps=[_cpu_step("local-step"), _cpu_step("docker-step", profile="docker")])

    translated = translate_cpu_container_steps_to_subprocess(spec, {"default"})

    assert isinstance(translated.steps[0].executor, SubprocessExecutionProvider)
    assert translated.steps[0].executor.command == ["python", "-m", "task"]
    assert isinstance(translated.steps[1].executor, CPUExecutionProvider)
    assert isinstance(spec.steps[0].executor, CPUExecutionProvider)


def test_translate_cpu_container_steps_to_subprocess_does_not_use_implicit_defaults() -> None:
    spec = HelixJobSpec(steps=[_cpu_step("docker-step")])

    translated = translate_cpu_container_steps_to_subprocess(spec, set())

    assert isinstance(translated.steps[0].executor, CPUExecutionProvider)


def test_validate_job_spec_matches_provider_and_profile() -> None:
    spec = HelixJobSpec(
        steps=[
            HelixJobStepSpec(
                name="local-step",
                executor=SubprocessExecutionProvider(provider="subprocess", profile="default", command=["true"]),
            )
        ]
    )
    profiles = [
        DockerJobExecutionProfile(
            provider="cpu", profile="default", backend="docker", config=DockerJobExecutionProfileConfig()
        )
    ]

    with pytest.raises(HTTPException, match="subprocess/default"):
        validate_job_spec(spec, profiles, creator="alice@example.com")


def _builder_only_profiles() -> list[ExecutionProfileT]:
    return [
        KubernetesJobExecutionProfile.model_validate(
            {"provider": "cpu", "profile": "build-push", "allowed_services": ["builder"], "config": {}}
        )
    ]


def test_a_profile_restricted_to_a_service_refuses_anyone_else() -> None:
    spec = HelixJobSpec(steps=[_cpu_step("push", profile="build-push")])

    with pytest.raises(HTTPException) as refused:
        validate_job_spec(spec, _builder_only_profiles(), creator="alice@example.com")

    assert refused.value.status_code == 403
    assert "build-push" in str(refused.value.detail)


def test_a_profile_restricted_to_a_service_refuses_another_service() -> None:
    spec = HelixJobSpec(steps=[_cpu_step("push", profile="build-push")])

    with pytest.raises(HTTPException) as refused:
        validate_job_spec(spec, _builder_only_profiles(), creator="service:evals")

    assert refused.value.status_code == 403


def test_a_profile_restricted_to_a_service_admits_that_service() -> None:
    spec = HelixJobSpec(steps=[_cpu_step("push", profile="build-push")])

    validate_job_spec(spec, _builder_only_profiles(), creator="service:builder")


def test_an_unrestricted_profile_admits_anyone() -> None:
    spec = HelixJobSpec(steps=[_cpu_step("step")])
    profiles: list[ExecutionProfileT] = [
        DockerJobExecutionProfile(
            provider="cpu", profile="default", backend="docker", config=DockerJobExecutionProfileConfig()
        )
    ]

    validate_job_spec(spec, profiles, creator="alice@example.com")
