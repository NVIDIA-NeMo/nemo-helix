# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot invocation helpers for Platform-owned Fabric agent configs."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.environment import ensure_local_workspace_dir
from nemo_agents_plugin.fabric.progress import LocalProgressObserver
from nemo_agents_plugin.fabric.progress import logger as progress_logger
from nemo_agents_plugin.fabric.runtime import FabricOneShotRequest, FabricRuntimeResult, run_fabric_agent_once
from nemo_agents_plugin.fabric.translator import translate_agent_config

FABRIC_BASE_DIR_NAME = "fabric"


@dataclass(frozen=True, slots=True)
class AgentConfigInvocationRequest:
    """Platform-owned request for one Fabric invocation from an agent config."""

    agent_config: AgentConfig
    input: Any
    base_dir: Path
    request_id: str | None = None
    caller_context: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None
    observe_progress: bool = False
    # When set, the agent's platform base URL resolves to this per-job auth-proxy
    # origin so its SDK reaches the platform authenticated as the job's identity.
    proxy_origin: str | None = None


@dataclass(frozen=True, slots=True)
class FabricDirectories:
    base: Path
    workspace: Path
    artifacts: Path

    @classmethod
    def create(cls, agent_config: AgentConfig, root: Path) -> Self:
        base = root / FABRIC_BASE_DIR_NAME
        workspace = _local_environment_path(
            base,
            agent_config.environment.workspace,
            field="workspace",
        )
        artifacts = _local_environment_path(
            base,
            agent_config.environment.artifacts,
            field="artifacts",
        )

        _reset_directory(workspace)
        _reset_directory(artifacts)

        return cls(base=base, workspace=workspace, artifacts=artifacts)


def _reset_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _local_environment_path(base_dir: Path, configured_path: str, *, field: str) -> Path:
    """Resolve a Fabric local environment path beneath ``base_dir``."""
    path = Path(configured_path)
    if path.is_absolute():
        raise ValueError(f"Local {field} path must be relative to the agent base directory.")

    resolved_base_dir = base_dir.resolve()
    resolved = (resolved_base_dir / path).resolve()
    if not resolved.is_relative_to(resolved_base_dir):
        raise ValueError(f"Local {field} path must remain within the agent base directory.")
    return resolved


async def invoke_agent_config_request_once(request: AgentConfigInvocationRequest) -> FabricRuntimeResult:
    """Translate a Platform agent config and run one input through Fabric once."""
    fabric_config = translate_agent_config(request.agent_config, proxy_origin=request.proxy_origin)
    await asyncio.to_thread(ensure_local_workspace_dir, request.agent_config, request.base_dir)
    if request.agent_config.environment.provider == "local":
        artifacts_dir = _local_environment_path(
            request.base_dir,
            request.agent_config.environment.artifacts,
            field="artifacts",
        )
        await asyncio.to_thread(artifacts_dir.mkdir, parents=True, exist_ok=True)

    observer_task: asyncio.Task[None] | None = None
    if request.observe_progress:
        try:
            if request.agent_config.environment.provider != "local":
                progress_logger.info("Live agent telemetry unavailable for a non-local environment.")
            else:
                observer = LocalProgressObserver(fabric_config, request.base_dir.resolve(), artifacts_dir)
                observer_task = asyncio.create_task(observer.run())
        except Exception:
            progress_logger.warning("Live agent telemetry observer setup failed; invocation continues.")
    try:
        return await run_fabric_agent_once(
            FabricOneShotRequest(
                fabric_config=fabric_config,
                base_dir=request.base_dir,
                input=request.input,
                request_id=request.request_id,
                caller_context=request.caller_context,
                timeout_seconds=request.timeout_seconds,
            )
        )
    finally:
        if observer_task is not None:
            observer_task.cancel()
            with suppress(asyncio.CancelledError):
                await observer_task


async def invoke_agent_config_once(
    agent_config: AgentConfig,
    inputs: Sequence[Any],
    *,
    base_dir: Path,
) -> list[FabricRuntimeResult]:
    """Translate a Platform agent config and run each input through Fabric once."""
    results: list[FabricRuntimeResult] = []
    for item in inputs:
        results.append(
            await invoke_agent_config_request_once(
                AgentConfigInvocationRequest(
                    agent_config=agent_config,
                    input=item,
                    base_dir=base_dir,
                )
            )
        )
    return results
