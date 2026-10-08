# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SDK environment boundary (process/filesystem execution seam)."""

from __future__ import annotations

import subprocess

import pytest
from nhx_evals_sdk.agent_eval.runtimes import environment as env_mod
from nhx_evals_sdk.agent_eval.runtimes.environment import (
    DockerEnvironmentHandle,
    DockerEnvironmentProvider,
    EnvCommandResult,
    EnvRunSpec,
    default_image_tag,
)
from nhx_evals_sdk.agent_eval.tasks import AgentEvalTask


@pytest.mark.asyncio
async def test_docker_handle_routes_roles_through_single_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_docker_run(image: str, spec: EnvRunSpec) -> EnvCommandResult:
        calls.append((image, spec.command))
        return EnvCommandResult(exit_code=0)

    monkeypatch.setattr(env_mod, "_docker_run", fake_docker_run)

    handle = DockerEnvironmentHandle("img:latest")
    spec = EnvRunSpec(command=["echo", "hi"])
    assert (await handle.run_agent(spec)).ok
    assert (await handle.run_verifier(spec)).ok
    assert calls == [("img:latest", ["echo", "hi"]), ("img:latest", ["echo", "hi"])]


@pytest.mark.asyncio
async def test_docker_handle_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    monkeypatch.setattr(env_mod.subprocess, "run", fake_run)
    result = await DockerEnvironmentHandle("img").run(EnvRunSpec(command=["sleep"]), "agent")
    assert result.timed_out and result.exit_code == 124 and not result.ok


@pytest.mark.asyncio
async def test_provider_uses_injected_image_tag_fn() -> None:
    assert default_image_tag("t") == "t:latest"
    provider = DockerEnvironmentProvider(image_tag_fn=lambda task_id: f"custom-{task_id}")
    handle = await provider.prepare(AgentEvalTask(id="demo", intent="x", inputs={}))
    assert isinstance(handle, DockerEnvironmentHandle)
    assert handle.image == "custom-demo"
