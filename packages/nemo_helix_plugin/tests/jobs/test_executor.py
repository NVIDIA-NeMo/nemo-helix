# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from nemo_helix_plugin.client.errors import NemoTransportError
from nemo_helix_plugin.jobs.api_factory import CPUExecutionProviderSpec, SubprocessExecutionProviderSpec
from nemo_helix_plugin.jobs.exceptions import HelixJobCompilationError, HelixJobDependencyUnavailableError
from nemo_helix_plugin.jobs.execution_profiles import (
    DockerJobExecutionProfile,
    DockerJobExecutionProfileConfig,
    SubprocessJobExecutionProfile,
)
from nemo_helix_plugin.jobs.executor import resolve_task_executor

SUBPROCESS_PROFILE = SubprocessJobExecutionProfile(profile="default")
CPU_PROFILE = DockerJobExecutionProfile(provider="cpu", profile="default", config=DockerJobExecutionProfileConfig())


@contextmanager
def registered(*execution_profiles: Any) -> Iterator[None]:
    async def _get_execution_profiles() -> Any:
        return SimpleNamespace(data=lambda: list(execution_profiles))

    client = MagicMock()
    client.get_execution_profiles = _get_execution_profiles
    with (
        patch("nemo_helix_plugin.jobs.executor.client_from_platform", return_value=client),
        patch("nemo_helix_plugin.jobs.executor.get_qualified_image", side_effect=lambda name: f"reg.example/{name}:t"),
    ):
        yield


async def resolve(profile: str = "default") -> Any:
    return await resolve_task_executor(
        profile=profile,
        async_sdk=MagicMock(),
        image="nhx-tasks",
        entrypoint=["python", "-m"],
        command=["pkg.task"],
        step="demo",
    )


async def test_prefers_a_subprocess_profile() -> None:
    with registered(SUBPROCESS_PROFILE, CPU_PROFILE):
        executor = await resolve()

    assert isinstance(executor, SubprocessExecutionProviderSpec)
    assert executor.profile == "default"
    assert executor.command == ["python", "-m", "pkg.task"]


async def test_falls_back_to_the_cpu_container() -> None:
    with registered(CPU_PROFILE):
        executor = await resolve()

    assert isinstance(executor, CPUExecutionProviderSpec)
    assert executor.container is not None
    assert executor.container.image == "reg.example/nhx-tasks:t"
    assert executor.container.entrypoint == ["python", "-m"]
    assert executor.container.command == ["pkg.task"]


async def test_names_the_step_and_available_profiles_when_none_match() -> None:
    with (
        registered(CPU_PROFILE),
        pytest.raises(HelixJobCompilationError, match=r"demo step has nowhere to run.*cpu/default"),
    ):
        await resolve(profile="gpu-heavy")


async def test_is_retryable_when_jobs_is_unreachable() -> None:
    async def _boom() -> Any:
        raise NemoTransportError(httpx.ConnectError("connection refused", request=httpx.Request("GET", "http://x")))

    client = MagicMock()
    client.get_execution_profiles = _boom
    with (
        patch("nemo_helix_plugin.jobs.executor.client_from_platform", return_value=client),
        pytest.raises(HelixJobDependencyUnavailableError, match="temporarily unavailable"),
    ):
        await resolve()
