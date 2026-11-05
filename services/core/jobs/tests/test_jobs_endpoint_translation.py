# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi import HTTPException
from nemo_helix_plugin.jobs.execution_profiles import OpenShellJobExecutionProfile, OpenShellJobExecutionProfileConfig
from nemo_helix_plugin.openshell.types import GatewayProviderBinding
from nhx.core.jobs.api.v2.jobs.endpoints import translate_cpu_container_steps_to_subprocess, validate_job_spec
from nhx.core.jobs.app.providers import ContainerSpec, CPUExecutionProvider, SubprocessExecutionProvider
from nhx.core.jobs.app.schemas import HelixJobSpec, HelixJobStepSpec
from nhx.core.jobs.controllers.backends.docker import DockerJobExecutionProfile, DockerJobExecutionProfileConfig


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
        validate_job_spec(spec, profiles)


@pytest.mark.parametrize("backend,verified", [("docker", False), ("openshell", False), ("openshell", True)])
def test_provider_jobs_require_configured_verified_openshell(backend, verified):
    step = _cpu_step("run")
    step.gateway_providers = {
        "service": GatewayProviderBinding(profile="service-read", credentials={"API_TOKEN": "token"})
    }
    if backend == "openshell":
        profile = OpenShellJobExecutionProfile(
            provider="cpu",
            profile="default",
            config=OpenShellJobExecutionProfileConfig(
                gateway_credentials={
                    "gateway_id": "gateway",
                    "registration_key_file": "/run/key",
                    "credential_refresh_verified": verified,
                    "profiles": {"service-read": {"workspaces": ["default"], "credential_keys": ["API_TOKEN"]}},
                }
            ),
        )
    else:
        profile = DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())
    spec = HelixJobSpec(steps=[step])
    if verified:
        validate_job_spec(spec, [profile])
    else:
        with pytest.raises(HTTPException, match="Gateway providers require"):
            validate_job_spec(spec, [profile])
