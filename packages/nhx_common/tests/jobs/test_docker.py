# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nhx.common.jobs.docker (Docker-specific GPU validation)."""

from unittest.mock import MagicMock, patch

import pytest
from nemo_helix_plugin.config import Runtime
from nemo_helix_plugin.jobs.api_factory import (
    ContainerSpec,
    CPUExecutionProviderSpec,
    GPUExecutionProviderSpec,
    PlatformJobSpec,
    PlatformJobStep,
)
from nhx.common.jobs.docker import spec_has_gpu_step, validate_gpu_available_for_docker
from nhx.common.jobs.exceptions import PlatformJobCompilationError


def test_spec_has_gpu_step():
    """spec_has_gpu_step returns True when any step has provider gpu or gpu_distributed."""
    cpu_executor = CPUExecutionProviderSpec(
        provider="cpu", profile="default", container=ContainerSpec(image="foo_image")
    )
    gpu_executor = GPUExecutionProviderSpec(
        provider="gpu", profile="default", container=ContainerSpec(image="gpu_image")
    )
    cpu_only_job = PlatformJobSpec(
        steps=[
            PlatformJobStep(name="cpu_step", environment={}, executor=cpu_executor, config={}),
        ]
    )
    assert spec_has_gpu_step(cpu_only_job) is False

    gpu_job = PlatformJobSpec(
        steps=[
            PlatformJobStep(name="gpu_step", environment={}, executor=gpu_executor, config={}),
        ]
    )
    assert spec_has_gpu_step(gpu_job) is True

    mixed_job = PlatformJobSpec(
        steps=[
            PlatformJobStep(name="cpu_step", environment={}, executor=cpu_executor, config={}),
            PlatformJobStep(name="gpu_step", environment={}, executor=gpu_executor, config={}),
        ]
    )
    assert spec_has_gpu_step(mixed_job) is True


@pytest.mark.parametrize(
    "runtime,reserved_gpu_ids,config_raises,docker_available,expect_raise,message_contains",
    [
        (Runtime.DOCKER, [], False, True, True, ("no GPUs configured", "Docker")),
        (Runtime.DOCKER, [0, 1], False, True, False, ()),
        (Runtime.KUBERNETES, [], False, True, False, ()),
        (Runtime.NONE, [], False, True, True, ("no GPUs configured", "Docker")),
        (Runtime.NONE, [], False, False, False, ()),
        (None, None, True, True, False, ()),
    ],
    ids=[
        "docker_no_gpus_raises",
        "docker_has_gpus_passes",
        "kubernetes_passes",
        "none_with_docker_no_gpus_raises",
        "none_without_docker_skips",
        "config_fails_skips",
    ],
)
def test_validate_gpu_available_for_docker(
    runtime, reserved_gpu_ids, config_raises, docker_available, expect_raise, message_contains
):
    """GPU job validation: raise when Docker has no GPUs; pass or skip otherwise."""
    gpu_executor = GPUExecutionProviderSpec(
        provider="gpu", profile="default", container=ContainerSpec(image="gpu_image")
    )
    gpu_job = PlatformJobSpec(
        steps=[
            PlatformJobStep(name="gpu_step", environment={}, executor=gpu_executor, config={}),
        ]
    )
    if config_raises:
        get_config = patch(
            "nemo_helix_plugin.jobs.docker.get_platform_config",
            side_effect=ValueError("no config"),
        )
    else:
        mock_platform_config = MagicMock()
        mock_platform_config.runtime = runtime
        mock_platform_config.docker.get_reserved_gpu_ids.return_value = reserved_gpu_ids
        get_config = patch(
            "nemo_helix_plugin.jobs.docker.get_platform_config",
            return_value=mock_platform_config,
        )

    with (
        get_config,
        patch(
            "nemo_helix_plugin.jobs.docker.probe_docker",
            return_value=MagicMock(available=docker_available),
        ),
    ):
        if expect_raise:
            with pytest.raises(PlatformJobCompilationError) as exc_info:
                validate_gpu_available_for_docker(gpu_job)
            err_msg = str(exc_info.value)
            for sub in message_contains:
                assert sub in err_msg
        else:
            validate_gpu_available_for_docker(gpu_job)
