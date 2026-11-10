# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from deepagents.backends import FilesystemBackend
from mcp.client.stdio import get_default_environment
from nemo_agents_plugin.agent_config import AgentConfig, McpConfig, McpServerConfig
from nemo_agents_plugin.fabric import environment, server, session_manager
from nemo_agents_plugin.fabric.runtime import (
    FabricInvocationRequest,
    FabricOneShotRequest,
    FabricRuntimeResult,
    FabricRuntimeStream,
)
from nemo_agents_plugin.fabric.session_manager import FabricSessionManager
from nemo_agents_plugin.fabric.session_registry import (
    FabricSessionNotFoundError,
    FabricSessionRegistry,
)
from nemo_agents_plugin.fabric.streaming import StreamingMethod
from nemo_fabric import FabricConfigError, RunPlan, RuntimeStatus


class _FakeRuntime:
    def __init__(self) -> None:
        self.stop_calls = 0
        self.status = RuntimeStatus.ACTIVE

    async def stop(self) -> None:
        self.stop_calls += 1
        self.status = RuntimeStatus.STOPPED


class _FakeFabricConfig:
    def __init__(self, *, copied: bool = False) -> None:
        self.harness = None
        self.copied = copied
        self.relay_enabled = False

    def model_copy(self, *, deep: bool) -> "_FakeFabricConfig":
        assert deep is True
        return _FakeFabricConfig(copied=True)

    def enable_relay(self) -> "_FakeFabricConfig":
        self.relay_enabled = True
        return self


class _FakeFabric:
    def __init__(self, runtime: _FakeRuntime, *, method: StreamingMethod = StreamingMethod.RELAY) -> None:
        self.runtime = runtime
        self.method = method
        self.plan_calls: list[Any] = []
        self.start_calls: list[dict[str, Any]] = []

    def plan(self, config: Any, *, base_dir: Path) -> RunPlan:
        self.plan_calls.append(config)
        return RunPlan.from_mapping(
            {
                "base_dir": str(base_dir),
                "config": {
                    "metadata": {"name": "test-agent"},
                    "harness": {"adapter_id": "nvidia.fabric.custom"},
                },
                "adapter": {"adapter_id": "nvidia.fabric.custom", "adapter_kind": "python"},
                "capabilities": {"streaming": self.method is StreamingMethod.OPENAI},
                "adapter_descriptor": {
                    "descriptor": {
                        "telemetry": {"providers": {"relay": {}} if self.method is StreamingMethod.RELAY else {}},
                        "capabilities": {"streaming": self.method is StreamingMethod.OPENAI},
                    }
                },
            }
        )

    async def start_runtime(self, config: Any, *, base_dir: Path, streaming: bool = False) -> _FakeRuntime:
        self.start_calls.append({"base_dir": base_dir, "config": config, "streaming": streaming})
        return self.runtime


class _SequenceFabric(_FakeFabric):
    def __init__(self, runtimes: list[_FakeRuntime]) -> None:
        super().__init__(runtimes[0])
        self.runtimes = iter(runtimes)

    async def start_runtime(self, config: Any, *, base_dir: Path, streaming: bool = False) -> _FakeRuntime:
        self.start_calls.append({"base_dir": base_dir, "config": config, "streaming": streaming})
        return next(self.runtimes)


def _agent_config() -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "config_format": "nemo-agents-spec-v1",
            "name": "test-agent",
            "default_harness": "hermes",
            "harnesses": {
                "hermes": {
                    "kind": "hermes",
                    "model": {
                        "provider": "nvidia",
                        "model": "nvidia/test-model",
                    },
                }
            },
        }
    )


@pytest.fixture
def private_manager(tmp_path: Path) -> FabricSessionManager:
    config = _agent_config()
    config.harnesses["hermes"].kind = "deepagents"
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace/input.txt").write_text("deployment input")
    return FabricSessionManager(
        config,
        base_dir=tmp_path,
        session_registry=FabricSessionRegistry(),
        workspace_baseline=environment.prepare_deepagents_workspace_baseline(config, tmp_path),
        fabric=cast(Any, _FakeFabric(_FakeRuntime(), method=StreamingMethod.BUFFERED)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "invocation"),
    [
        ("copy", "session"),
        ("plan", "session"),
        ("start", "session"),
        ("register", "session"),
        ("plan", "once"),
        ("plan", "stream"),
    ],
)
async def test_failed_session_start_removes_private_workspace(
    private_manager: FabricSessionManager, monkeypatch: pytest.MonkeyPatch, failure: str, invocation: str
) -> None:
    if failure == "copy":

        def fail_copy(source: Path, target: Path, **kwargs: Any) -> None:
            (target / "partial.txt").write_text("incomplete")
            raise OSError("copy failed")

        monkeypatch.setattr(environment.shutil, "copytree", fail_copy)
    else:
        target, method = {
            "plan": (private_manager, "_materialize_fabric_config"),
            "start": (private_manager._fabric, "start_runtime"),
            "register": (private_manager._session_registry, "register"),
        }[failure]
        monkeypatch.setattr(target, method, AsyncMock(side_effect=session_manager.FabricRuntimeStartError("failed")))

    with pytest.raises(session_manager.FabricRuntimeStartError):
        if invocation == "session":
            await private_manager.open_session()
        elif invocation == "once":
            await private_manager.invoke_once(FabricInvocationRequest(input="hello"))
        else:
            async with private_manager.stream_once(FabricInvocationRequest(input="hello")):
                pass

    baseline = private_manager._workspace_baseline
    assert baseline is not None
    assert list((baseline.parent / "sessions").iterdir()) == []
    assert (baseline / "input.txt").read_text() == "deployment input"
    assert await private_manager._session_registry.count() == 0


@pytest.mark.asyncio
async def test_cancelled_copy_finishes_before_workspace_removal(
    private_manager: FabricSessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()
    copy = environment.shutil.copytree

    def slow_copy(source: Path, target: Path, **kwargs: Any) -> Any:
        started.set()
        assert release.wait(timeout=5)
        return copy(source, target, **kwargs)

    monkeypatch.setattr(environment.shutil, "copytree", slow_copy)
    opening = asyncio.create_task(private_manager.open_session())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        opening.cancel()
        await asyncio.sleep(0)
        opening.cancel()
        assert not opening.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await opening
    baseline = private_manager._workspace_baseline
    assert baseline is not None
    assert list((baseline.parent / "sessions").iterdir()) == []
    assert (baseline / "input.txt").exists()


@pytest.mark.asyncio
async def test_cancelled_start_stops_runtime_before_workspace_removal(
    private_manager: FabricSessionManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runtime = _FakeRuntime()

    async def start(*args: Any, **kwargs: Any) -> _FakeRuntime:
        started.set()
        await release.wait()
        return runtime

    monkeypatch.setattr(private_manager._fabric, "start_runtime", start)
    opening = asyncio.create_task(private_manager.open_session())
    await started.wait()
    opening.cancel()
    await asyncio.sleep(0)
    assert not opening.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert runtime.stop_calls == 1
    baseline = private_manager._workspace_baseline
    assert baseline is not None
    assert list((baseline.parent / "sessions").iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", list(StreamingMethod))
async def test_open_session_materializes_config_and_starts_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: StreamingMethod,
) -> None:
    fabric_config = _FakeFabricConfig()
    translation_calls: list[AgentConfig] = []

    def translate(config: AgentConfig) -> Any:
        translation_calls.append(config)
        return fabric_config

    monkeypatch.setattr(session_manager, "translate_agent_config", translate)
    runtime = _FakeRuntime()
    fabric = _FakeFabric(runtime, method=method)
    registry = FabricSessionRegistry()
    agent_config = _agent_config()
    manager = FabricSessionManager(
        agent_config,
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, fabric),
    )

    assert translation_calls == []
    assert fabric.start_calls == []

    session = await manager.open_session()

    assert translation_calls == [agent_config]
    assert (tmp_path / "workspace").is_dir()
    assert len(fabric.start_calls) == 1
    assert fabric.start_calls[0]["base_dir"] == tmp_path
    relay_streaming = method is StreamingMethod.RELAY
    assert fabric.start_calls[0]["streaming"] is relay_streaming
    started_config = fabric.start_calls[0]["config"]
    assert started_config.copied is relay_streaming
    assert started_config.relay_enabled is relay_streaming
    assert fabric.plan_calls == [fabric_config]
    assert fabric_config.relay_enabled is False
    assert session.runtime is runtime
    assert session.streaming_method is method
    assert await registry.get(session.session_id) is session


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [FabricConfigError, session_manager.FabricError])
async def test_open_session_reports_plan_failure_before_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    fabric = _FakeFabric(_FakeRuntime())

    def fail_plan(config: Any, *, base_dir: Path) -> RunPlan:
        raise error_type("adapter unavailable")

    monkeypatch.setattr(fabric, "plan", fail_plan)
    manager = FabricSessionManager(
        _agent_config(), base_dir=tmp_path, session_registry=FabricSessionRegistry(), fabric=cast(Any, fabric)
    )

    with pytest.raises(session_manager.FabricSessionStartError, match="adapter unavailable"):
        await manager.open_session()

    assert fabric.start_calls == []


@pytest.mark.asyncio
async def test_open_session_stops_runtime_when_registration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    runtime = _FakeRuntime()
    fabric = _FakeFabric(runtime)
    registry = FabricSessionRegistry()

    async def fail_registration(
        runtime: Any, *, session_id: str | None = None, streaming_method: StreamingMethod
    ) -> None:
        raise RuntimeError("registration failed")

    monkeypatch.setattr(registry, "register", fail_registration)
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, fabric),
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        await manager.open_session()

    assert runtime.stop_calls == 1
    assert await registry.count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("relay_session_root", [None, "c46d3588-d205-497f-b96b-37f938263064"])
async def test_invoke_once_uses_unregistered_runtime(
    relay_session_root: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric_config = _FakeFabricConfig()
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: fabric_config)
    captured: list[tuple[Any, Any]] = []

    async def run_fabric_agent_once(request: Any, *, fabric: Any) -> FabricRuntimeResult:
        captured.append((request, fabric))
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "run_fabric_agent_once", run_fabric_agent_once)
    registry = FabricSessionRegistry()
    fabric = object()
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, fabric),
    )

    result = await manager.invoke_once(FabricInvocationRequest(input="hello", relay_session_root=relay_session_root))

    assert result.response == "hello"
    request, captured_fabric = captured[0]
    assert request.fabric_config is fabric_config
    assert request.input == "hello"
    assert request.caller_context == {}
    assert request.relay_session_root == relay_session_root
    assert captured_fabric is fabric
    assert await registry.count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("method", list(StreamingMethod))
async def test_stream_once_selects_runtime_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: StreamingMethod,
) -> None:
    fabric_config = _FakeFabricConfig()
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: fabric_config)
    captured: list[tuple[Any, Any, StreamingMethod]] = []
    expected_stream = object()

    @asynccontextmanager
    async def stream_fabric_agent_once(request: Any, *, fabric: Any, streaming_method: StreamingMethod) -> Any:
        captured.append((request, fabric, streaming_method))
        yield expected_stream

    monkeypatch.setattr(session_manager, "stream_fabric_agent_once", stream_fabric_agent_once)
    registry = FabricSessionRegistry()
    fabric = _FakeFabric(_FakeRuntime(), method=method)
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, fabric),
    )

    async with manager.stream_once(FabricInvocationRequest(input="hello")) as stream:
        assert stream is expected_stream

    request, captured_fabric, streaming_method = captured[0]
    assert streaming_method is method
    relay_streaming = method is StreamingMethod.RELAY
    assert request.fabric_config.copied is relay_streaming
    assert request.fabric_config.relay_enabled is relay_streaming
    assert fabric.plan_calls == [fabric_config]
    assert fabric_config.relay_enabled is False
    assert request.input == "hello"
    assert captured_fabric is fabric
    assert await registry.count() == 0


@pytest.mark.asyncio
async def test_resolve_session_lazily_opens_and_reuses_runtime_with_platform_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    runtime = _FakeRuntime()
    fabric = _FakeFabric(runtime)
    registry = FabricSessionRegistry()
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, fabric),
    )

    first = await manager.resolve_session("platform-session-1")
    second = await manager.resolve_session("platform-session-1")

    assert first.session_id == "platform-session-1"
    assert first.runtime is runtime
    assert second is first
    assert await registry.get("platform-session-1") is first
    assert len(fabric.start_calls) == 1


@pytest.mark.asyncio
async def test_different_platform_sessions_get_separate_runtimes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    first_runtime = _FakeRuntime()
    second_runtime = _FakeRuntime()
    fabric = _SequenceFabric([first_runtime, second_runtime])
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=FabricSessionRegistry(),
        fabric=cast(Any, fabric),
    )

    first = await manager.resolve_session("platform-session-1")
    second = await manager.resolve_session("platform-session-2")

    assert first.session_id == "platform-session-1"
    assert second.session_id == "platform-session-2"
    assert first.runtime is first_runtime
    assert second.runtime is second_runtime
    assert len(fabric.start_calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_workspace", ["workspace", "."])
async def test_deepagents_sessions_isolate_files_and_reuse_their_workspace(
    tmp_path: Path, configured_workspace: str
) -> None:
    config = _agent_config()
    config.harnesses["hermes"].kind = "deepagents"
    config.environment.workspace = configured_workspace
    original_config = config.model_copy(deep=True)
    workspace = tmp_path / configured_workspace
    workspace.mkdir(exist_ok=True)
    (workspace / "input.txt").write_text("original input")
    (workspace / "linked-input").symlink_to(workspace / "input.txt")
    baseline = environment.prepare_deepagents_workspace_baseline(config, tmp_path)
    assert baseline is not None
    fabric = _SequenceFabric([_FakeRuntime(), _FakeRuntime()])
    manager = FabricSessionManager(
        config,
        base_dir=tmp_path,
        session_registry=FabricSessionRegistry(),
        workspace_baseline=baseline,
        fabric=cast(Any, fabric),
    )

    first, concurrent = await asyncio.gather(
        manager.resolve_session("../opaque-session"), manager.resolve_session("../opaque-session")
    )
    assert concurrent is first
    assert first.workspace_dir is not None
    (first.workspace_dir / "linked-input").write_text("edited in the first session")
    outputs = ("large_tool_results/result.txt", "conversation_history/history.md", "report.md")
    for name in outputs:
        output = first.workspace_dir / name
        output.parent.mkdir(exist_ok=True)
        output.write_text("private session content")

    second = await manager.resolve_session("second-session")
    assert second.workspace_dir is not None
    assert first.workspace_dir != second.workspace_dir
    assert await manager.resolve_session("../opaque-session") is first
    assert (first.workspace_dir / "input.txt").read_text() == "edited in the first session"
    for root in (workspace, baseline, second.workspace_dir):
        assert (root / "input.txt").read_text() == "original input"
        assert not any((root / name).exists() for name in outputs)
    backend = FilesystemBackend(root_dir=str(second.workspace_dir), virtual_mode=True)
    for name in outputs:
        assert backend.read(f"/{name}").error is not None
    assert not backend.grep("private session content").matches
    with pytest.raises(ValueError, match="traversal"):
        backend.read(f"../{first.workspace_dir.name}/report.md")
    assert len(fabric.start_calls) == 2
    for session, call in zip((first, second), fabric.start_calls, strict=True):
        assert session.workspace_dir is not None
        assert session.workspace_dir.parent == baseline.parent / "sessions"
        assert (session.workspace_dir / "linked-input").resolve() == session.workspace_dir / "input.txt"
        assert tmp_path / call["config"].environment.workspace == session.workspace_dir
        assert call["base_dir"] == tmp_path
    assert config == original_config

    await manager.close_session(first.session_id)
    assert not first.workspace_dir.exists()
    assert second.workspace_dir.exists()
    await manager.close_all_sessions()
    assert not second.workspace_dir.exists()
    assert (baseline / "input.txt").read_text() == "original input"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_deepagents_one_shot_invocations_get_private_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    config = _agent_config()
    config.harnesses["hermes"].kind = "deepagents"
    original_config = config.model_copy(deep=True)
    workspace = tmp_path / config.environment.workspace
    workspace.mkdir()
    (workspace / "input.txt").write_text("original input")
    baseline = environment.prepare_deepagents_workspace_baseline(config, tmp_path)
    registry = FabricSessionRegistry()
    assert baseline is not None
    manager = FabricSessionManager(
        config,
        base_dir=tmp_path,
        session_registry=registry,
        workspace_baseline=baseline,
        fabric=cast(Any, _FakeFabric(_FakeRuntime(), method=StreamingMethod.BUFFERED)),
    )
    workspaces: list[Path] = []

    def record_workspace(request: FabricOneShotRequest) -> None:
        assert request.base_dir == tmp_path
        assert request.fabric_config.environment is not None
        assert request.fabric_config.environment.workspace is not None
        private = tmp_path / request.fabric_config.environment.workspace
        assert request.workspace_dir == private
        assert (private / "input.txt").read_text() == "original input"
        assert not (private / "result.txt").exists()
        (private / "input.txt").write_text("edited by this invocation")
        (private / "result.txt").write_text("private result")
        workspaces.append(private)

    async def run_once(request: FabricOneShotRequest, **kwargs: Any) -> FabricRuntimeResult:
        record_workspace(request)
        return FabricRuntimeResult(status="succeeded", response="done")

    @asynccontextmanager
    async def stream_once(request: FabricOneShotRequest, **kwargs: Any) -> Any:
        record_workspace(request)
        yield object()

    monkeypatch.setattr(session_manager, "run_fabric_agent_once", run_once)
    monkeypatch.setattr(session_manager, "stream_fabric_agent_once", stream_once)
    for _ in range(2):
        request = FabricInvocationRequest(input="hello")
        if streaming:
            async with manager.stream_once(request):
                pass
        else:
            await manager.invoke_once(request)

    assert len(set(workspaces)) == 2
    assert all(private.parent == baseline.parent / "sessions" for private in workspaces)
    assert (workspace / "input.txt").read_text() == "original input"
    assert config == original_config
    assert await registry.count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_workspace", ["workspace", "."])
@pytest.mark.parametrize("command_kind", ["script", "module", "executable", "venv", "path", "env"])
async def test_private_workspace_preserves_local_mcp_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured_workspace: str, command_kind: str
) -> None:
    config = _agent_config()
    config.harnesses["hermes"].kind = "deepagents"
    config.environment.workspace = configured_workspace
    workspace = tmp_path / configured_workspace
    workspace.mkdir(exist_ok=True)
    (workspace / "input.txt").write_text("original MCP input")
    (tmp_path / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    dependencies = tmp_path / "dependencies"
    dependencies.mkdir()
    (dependencies / "probe_dependency.py").write_text("VALUE = 'installed dependency'\n")
    module = workspace / "local_tools"
    module.mkdir()
    script = module / "probe.py"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, probe_dependency\n"
        "from pathlib import Path\n"
        "print(json.dumps(dict(cwd=os.getcwd(), input=Path('input.txt').read_text(), "
        "args=sys.argv[1:], env=dict(os.environ), python=sys.executable, dependency=probe_dependency.VALUE)))\n"
    )
    script.chmod(0o755)
    python = os.path.relpath(tmp_path / ".venv/bin/python", workspace)
    script_path = "./local_tools/probe.py"
    command, args = {
        "script": (sys.executable, [script_path]),
        "module": (sys.executable, ["-m", "local_tools.probe"]),
        "executable": (script_path, []),
        "venv": (python, [script_path]),
        "path": ("python", [script_path]),
        "env": (" ${MCP_TEST_PYTHON} ", [script_path]),
    }[command_kind]
    monkeypatch.setenv("MCP_TEST_PYTHON", "overridden-by-runtime-env")
    config.environment.env["MCP_TEST_PYTHON"] = python
    local = McpServerConfig(
        transport="stdio",
        url=command,
        args=[*args, "argument with spaces", "$literal;value"],
        env={
            "MCP_TEST_VALUE": "configured value",
            "PYTHONPATH": os.path.relpath(dependencies, workspace),
            "PATH": f"{os.path.dirname(python)}{os.pathsep}{os.environ['PATH']}",
        },
    )
    remote = McpServerConfig(
        transport="streamable-http", url="https://tools.invalid/mcp", custom_headers={"X-Probe": "value"}
    )
    config.mcp = McpConfig(servers={"local": local, "remote": remote})
    original_config = config.model_copy(deep=True)
    child_env = {**get_default_environment(), **local.env, "PYTHONDONTWRITEBYTECODE": "1"}
    expected = subprocess.run(
        [python if command_kind == "env" else command, *local.args],
        cwd=workspace,
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )
    baseline = environment.prepare_deepagents_workspace_baseline(config, tmp_path)
    fabric = _FakeFabric(_FakeRuntime(), method=StreamingMethod.BUFFERED)
    manager = FabricSessionManager(
        config,
        base_dir=tmp_path,
        session_registry=FabricSessionRegistry(),
        workspace_baseline=baseline,
        fabric=cast(Any, fabric),
    )
    session = await manager.open_session()
    assert session.workspace_dir is not None
    (session.workspace_dir / "input.txt").write_text("private file-tool input")
    started_config = fabric.start_calls[0]["config"]
    wrapped = started_config.mcp.servers["local"]
    actual = subprocess.run(
        [wrapped.url, *wrapped.args],
        cwd=session.workspace_dir,
        env={**get_default_environment(), **wrapped.env, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(actual.stdout) == json.loads(expected.stdout)
    assert wrapped.env == local.env
    assert started_config.mcp.servers["remote"].url == remote.url
    assert started_config.mcp.servers["remote"].custom_headers == remote.custom_headers
    assert not (session.workspace_dir / ".venv").exists()
    assert config == original_config


@pytest.mark.asyncio
async def test_concurrent_first_turns_start_one_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    runtime = _FakeRuntime()
    startup_started = asyncio.Event()
    release_startup = asyncio.Event()

    class _BlockingFabric(_FakeFabric):
        async def start_runtime(self, config: Any, *, base_dir: Path, streaming: bool = False) -> _FakeRuntime:
            self.start_calls.append({"base_dir": base_dir, "config": config, "streaming": streaming})
            startup_started.set()
            await release_startup.wait()
            return self.runtime

    fabric = _BlockingFabric(runtime)
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=FabricSessionRegistry(),
        fabric=cast(Any, fabric),
    )

    first_resolution = asyncio.create_task(manager.resolve_session("platform-session-1"))
    await startup_started.wait()
    second_resolution = asyncio.create_task(manager.resolve_session("platform-session-1"))
    await asyncio.sleep(0)

    assert len(fabric.start_calls) == 1
    release_startup.set()
    first, second = await asyncio.gather(first_resolution, second_resolution)

    assert first is second
    assert first.runtime is runtime
    assert len(fabric.start_calls) == 1


@pytest.mark.asyncio
async def test_resolve_session_reuses_registered_runtime(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    fabric = _FakeFabric(_FakeRuntime())
    registry = FabricSessionRegistry()
    registered = await registry.register(cast(Any, runtime), session_id="session-1")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, fabric),
    )

    session = await manager.resolve_session("session-1")

    assert session is registered
    assert session.runtime is runtime
    assert fabric.start_calls == []


@pytest.mark.asyncio
async def test_close_session_removes_session_and_stops_runtime(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, runtime), session_id="session-1")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
    )

    await manager.close_session(session.session_id)

    assert runtime.stop_calls == 1
    assert await registry.count() == 0
    with pytest.raises(FabricSessionNotFoundError, match="session-1"):
        await manager.resolve_session(session.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close", [False, True])
async def test_close_session_waits_for_active_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_close: bool,
) -> None:
    runtime = _FakeRuntime()
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, runtime), session_id="session-1")
    session.workspace_dir = tmp_path / "private"
    session.workspace_dir.mkdir()
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
    )
    invocation_started = asyncio.Event()
    release_invocation = asyncio.Event()

    async def invoke_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> FabricRuntimeResult:
        invocation_started.set()
        await release_invocation.wait()
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "invoke_fabric_runtime", invoke_fabric_runtime)
    invocation = asyncio.create_task(
        manager.invoke_session(session, FabricInvocationRequest(input="hello")),
    )
    await invocation_started.wait()

    close = asyncio.create_task(manager.close_session(session.session_id))
    await asyncio.sleep(0)

    assert runtime.stop_calls == 0
    assert session.workspace_dir.exists()
    with pytest.raises(FabricSessionNotFoundError, match="session-1"):
        await manager.resolve_session(session.session_id)

    if cancel_close:
        close.cancel()
    release_invocation.set()
    await invocation
    if cancel_close:
        with pytest.raises(asyncio.CancelledError):
            await close
    else:
        await close

    assert runtime.stop_calls == 1
    assert not session.workspace_dir.exists()


@pytest.mark.asyncio
async def test_resolved_session_cannot_invoke_after_close_starts(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, runtime), session_id="session-1")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
    )

    await manager.close_session(session.session_id)

    with pytest.raises(FabricSessionNotFoundError, match="session-1"):
        await manager.invoke_session(session, FabricInvocationRequest(input="hello"))


@pytest.mark.asyncio
async def test_close_missing_session_tombstones_platform_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    fabric = _FakeFabric(_FakeRuntime())
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=FabricSessionRegistry(),
        fabric=cast(Any, fabric),
    )

    with pytest.raises(FabricSessionNotFoundError, match="session-1"):
        await manager.close_session("session-1")
    with pytest.raises(FabricSessionNotFoundError, match="session-1"):
        await manager.resolve_session("session-1")

    assert fabric.start_calls == []


@pytest.mark.asyncio
async def test_expire_idle_sessions_stops_expired_runtime(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, runtime), session_id="session-1")
    session.workspace_dir = tmp_path / "private"
    session.workspace_dir.mkdir()
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
    )
    session.last_accessed_at = float("-inf")

    expired_count = await manager.expire_idle_sessions(idle_timeout_seconds=30.0)

    assert expired_count == 1
    assert runtime.stop_calls == 1
    assert await registry.count() == 0
    assert not session.workspace_dir.exists()


@pytest.mark.asyncio
async def test_idle_expiration_allows_runtime_to_be_recreated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    first_runtime = _FakeRuntime()
    second_runtime = _FakeRuntime()
    fabric = _SequenceFabric([first_runtime, second_runtime])
    registry = FabricSessionRegistry()
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, fabric),
    )
    first_session = await manager.resolve_session("session-1")
    first_session.last_accessed_at = float("-inf")

    assert await manager.expire_idle_sessions(idle_timeout_seconds=30.0) == 1
    second_session = await manager.resolve_session("session-1")

    assert first_runtime.stop_calls == 1
    assert second_session is not first_session
    assert second_session.runtime is second_runtime
    assert len(fabric.start_calls) == 2


@pytest.mark.asyncio
async def test_close_all_sessions_stops_every_runtime(tmp_path: Path) -> None:
    first_runtime = _FakeRuntime()
    second_runtime = _FakeRuntime()
    registry = FabricSessionRegistry()
    await registry.register(cast(Any, first_runtime), session_id="session-1")
    await registry.register(cast(Any, second_runtime), session_id="session-2")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
    )

    closed_count = await manager.close_all_sessions()

    assert closed_count == 2
    assert first_runtime.stop_calls == 1
    assert second_runtime.stop_calls == 1
    assert await registry.count() == 0


@pytest.mark.asyncio
async def test_close_all_sessions_continues_after_stop_failure(tmp_path: Path) -> None:
    class _FailingRuntime(_FakeRuntime):
        async def stop(self) -> None:
            self.stop_calls += 1
            raise session_manager.FabricError("stop failed")

    failing_runtime = _FailingRuntime()
    healthy_runtime = _FakeRuntime()
    registry = FabricSessionRegistry()
    failing = await registry.register(cast(Any, failing_runtime), session_id="session-1")
    healthy = await registry.register(cast(Any, healthy_runtime), session_id="session-2")
    for session in (failing, healthy):
        session.workspace_dir = tmp_path / session.session_id
        session.workspace_dir.mkdir()
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
    )

    closed_count = await manager.close_all_sessions()

    assert closed_count == 2
    assert failing_runtime.stop_calls == 1
    assert healthy_runtime.stop_calls == 1
    assert failing.workspace_dir is not None
    assert healthy.workspace_dir is not None
    assert failing.workspace_dir.is_dir()
    assert not healthy.workspace_dir.exists()


@pytest.mark.asyncio
async def test_close_all_sessions_continues_after_unexpected_stop_failure(tmp_path: Path) -> None:
    class _FailingRuntime(_FakeRuntime):
        async def stop(self) -> None:
            self.stop_calls += 1
            raise RuntimeError("unexpected stop failure")

    failing_runtime = _FailingRuntime()
    healthy_runtime = _FakeRuntime()
    registry = FabricSessionRegistry()
    await registry.register(cast(Any, failing_runtime), session_id="session-1")
    await registry.register(cast(Any, healthy_runtime), session_id="session-2")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
    )

    closed_count = await manager.close_all_sessions()

    assert closed_count == 2
    assert failing_runtime.stop_calls == 1
    assert healthy_runtime.stop_calls == 1


@pytest.mark.asyncio
async def test_invoke_session_refreshes_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
    )
    refresh_calls: list[Any] = []

    async def invoke_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> FabricRuntimeResult:
        return FabricRuntimeResult(status="succeeded", response=request.input)

    async def refresh_activity(resolved_session: Any) -> None:
        refresh_calls.append(resolved_session)

    monkeypatch.setattr(session_manager, "invoke_fabric_runtime", invoke_fabric_runtime)
    monkeypatch.setattr(registry, "refresh_activity", refresh_activity)

    await manager.invoke_session(session, FabricInvocationRequest(input="hello"))

    assert refresh_calls == [session]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", list(StreamingMethod))
async def test_stream_session_refreshes_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: StreamingMethod,
) -> None:
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1", streaming_method=method)
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
    )
    stream = object()
    refresh_calls: list[Any] = []

    def stream_fabric_runtime(
        runtime: Any, request: FabricInvocationRequest, *, streaming_method: StreamingMethod
    ) -> object:
        assert streaming_method is method
        return stream

    async def refresh_activity(resolved_session: Any) -> None:
        refresh_calls.append(resolved_session)

    monkeypatch.setattr(session_manager, "stream_fabric_runtime", stream_fabric_runtime)
    monkeypatch.setattr(registry, "refresh_activity", refresh_activity)

    async with manager.stream_session(session, FabricInvocationRequest(input="hello")) as resolved_stream:
        assert resolved_stream is stream

    assert refresh_calls == [session]


@pytest.mark.asyncio
async def test_invoke_session_serializes_turns_for_same_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, _FakeFabric(_FakeRuntime())),
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    invocation_order: list[str] = []
    active_invocations = 0
    max_active_invocations = 0

    async def invoke_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> FabricRuntimeResult:
        nonlocal active_invocations, max_active_invocations
        active_invocations += 1
        max_active_invocations = max(max_active_invocations, active_invocations)
        invocation_order.append(request.input)
        if request.input == "first":
            first_started.set()
            await release_first.wait()
        active_invocations -= 1
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "invoke_fabric_runtime", invoke_fabric_runtime)

    first = asyncio.create_task(
        manager.invoke_session(session, FabricInvocationRequest(input="first")),
    )
    await first_started.wait()
    second = asyncio.create_task(
        manager.invoke_session(session, FabricInvocationRequest(input="second")),
    )
    await asyncio.sleep(0)

    assert invocation_order == ["first"]

    release_first.set()
    results = await asyncio.gather(first, second)

    assert invocation_order == ["first", "second"]
    assert max_active_invocations == 1
    assert [result.response for result in results] == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", list(StreamingMethod))
async def test_stream_session_holds_lock_until_cleanup_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: StreamingMethod,
) -> None:
    monkeypatch.setattr(server, "_FABRIC_STREAM_CLEANUP_TIMEOUT_SECONDS", 0.01)
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1", streaming_method=method)
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, _FakeFabric(_FakeRuntime())),
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    invocation_order: list[str] = []

    async def close_stream() -> None:
        first_started.set()
        await release_first.wait()

    def stream_fabric_runtime(
        runtime: Any, request: FabricInvocationRequest, *, streaming_method: StreamingMethod
    ) -> FabricRuntimeStream:
        assert streaming_method is method
        invocation_order.append(request.input)
        return cast(FabricRuntimeStream, SimpleNamespace(aclose=close_stream))

    monkeypatch.setattr(session_manager, "stream_fabric_runtime", stream_fabric_runtime)

    async def stream_first() -> server._StreamingChatCompletionIterator:
        context = manager.stream_session(session, FabricInvocationRequest(input="first"))
        stream = await context.__aenter__()
        iterator = server._iter_streaming_chat_completion(context, stream, completion_id="first", model="test-model")
        await iterator.aclose()
        return iterator

    async def stream_second() -> None:
        async with manager.stream_session(session, FabricInvocationRequest(input="second")):
            pass

    first = asyncio.create_task(stream_first())
    await first_started.wait()
    second = asyncio.create_task(stream_second())
    iterator = await asyncio.wait_for(first, timeout=1)

    assert invocation_order == ["first"]
    assert session.invocation_lock.locked()

    release_first.set()
    assert iterator._close_task is not None
    await asyncio.gather(iterator._close_task, second)

    assert invocation_order == ["first", "second"]
    assert not session.invocation_lock.locked()


@pytest.mark.asyncio
async def test_invoke_session_releases_lock_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, _FakeFabric(_FakeRuntime())),
    )
    invocation_count = 0

    async def invoke_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> FabricRuntimeResult:
        nonlocal invocation_count
        invocation_count += 1
        if invocation_count == 1:
            raise RuntimeError("invoke failed")
        return FabricRuntimeResult(status="succeeded", response="recovered")

    monkeypatch.setattr(session_manager, "invoke_fabric_runtime", invoke_fabric_runtime)

    with pytest.raises(RuntimeError, match="invoke failed"):
        await manager.invoke_session(session, FabricInvocationRequest(input="first"))

    result = await manager.invoke_session(session, FabricInvocationRequest(input="second"))

    assert result.response == "recovered"


@pytest.mark.asyncio
async def test_invoke_session_releases_lock_after_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, _FakeFabric(_FakeRuntime())),
    )
    invocation_started = asyncio.Event()

    async def invoke_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> FabricRuntimeResult:
        if request.input == "cancel":
            invocation_started.set()
            await asyncio.Event().wait()
        return FabricRuntimeResult(status="succeeded", response="recovered")

    monkeypatch.setattr(session_manager, "invoke_fabric_runtime", invoke_fabric_runtime)
    cancelled = asyncio.create_task(
        manager.invoke_session(session, FabricInvocationRequest(input="cancel")),
    )
    await invocation_started.wait()

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    result = await manager.invoke_session(session, FabricInvocationRequest(input="next"))

    assert result.response == "recovered"


def test_rejects_negative_concurrency_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be greater than or equal to zero"):
        FabricSessionManager(
            _agent_config(),
            base_dir=tmp_path,
            session_registry=FabricSessionRegistry(),
            max_concurrent_invocations=-1,
        )


@pytest.mark.asyncio
async def test_invoke_session_limits_concurrency_across_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FabricSessionRegistry()
    first_session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1")
    second_session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-2")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        max_concurrent_invocations=1,
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    invocation_order: list[str] = []

    async def invoke_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> FabricRuntimeResult:
        invocation_order.append(request.input)
        if request.input == "first":
            first_started.set()
            await release_first.wait()
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "invoke_fabric_runtime", invoke_fabric_runtime)
    first = asyncio.create_task(
        manager.invoke_session(first_session, FabricInvocationRequest(input="first")),
    )
    await first_started.wait()
    second = asyncio.create_task(
        manager.invoke_session(second_session, FabricInvocationRequest(input="second")),
    )
    await asyncio.sleep(0)

    assert invocation_order == ["first"]

    release_first.set()
    await asyncio.gather(first, second)

    assert invocation_order == ["first", "second"]


@pytest.mark.asyncio
async def test_cancelled_capacity_waiter_does_not_leak_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FabricSessionRegistry()
    first_session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1")
    second_session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-2")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        max_concurrent_invocations=1,
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    invocation_order: list[str] = []

    async def invoke_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> FabricRuntimeResult:
        invocation_order.append(request.input)
        if request.input == "first":
            first_started.set()
            await release_first.wait()
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "invoke_fabric_runtime", invoke_fabric_runtime)
    first = asyncio.create_task(
        manager.invoke_session(first_session, FabricInvocationRequest(input="first")),
    )
    await first_started.wait()
    cancelled = asyncio.create_task(
        manager.invoke_session(second_session, FabricInvocationRequest(input="cancelled")),
    )
    await asyncio.sleep(0)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    release_first.set()
    await first
    result = await manager.invoke_session(second_session, FabricInvocationRequest(input="next"))

    assert invocation_order == ["first", "next"]
    assert result.response == "next"


@pytest.mark.asyncio
async def test_invoke_once_limits_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=FabricSessionRegistry(),
        max_concurrent_invocations=1,
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    invocation_order: list[str] = []

    async def run_fabric_agent_once(request: Any, *, fabric: Any) -> FabricRuntimeResult:
        invocation_order.append(request.input)
        if request.input == "first":
            first_started.set()
            await release_first.wait()
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "run_fabric_agent_once", run_fabric_agent_once)
    first = asyncio.create_task(manager.invoke_once(FabricInvocationRequest(input="first")))
    await first_started.wait()
    second = asyncio.create_task(manager.invoke_once(FabricInvocationRequest(input="second")))
    await asyncio.sleep(0)

    assert invocation_order == ["first"]

    release_first.set()
    results = await asyncio.gather(first, second)

    assert invocation_order == ["first", "second"]
    assert [result.response for result in results] == ["first", "second"]


@pytest.mark.asyncio
async def test_one_shot_and_session_invocations_share_concurrency_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        max_concurrent_invocations=1,
    )
    session_started = asyncio.Event()
    release_session = asyncio.Event()
    invocation_order: list[str] = []

    async def invoke_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> FabricRuntimeResult:
        invocation_order.append(request.input)
        session_started.set()
        await release_session.wait()
        return FabricRuntimeResult(status="succeeded", response=request.input)

    async def run_fabric_agent_once(request: Any, *, fabric: Any) -> FabricRuntimeResult:
        invocation_order.append(request.input)
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "invoke_fabric_runtime", invoke_fabric_runtime)
    monkeypatch.setattr(session_manager, "run_fabric_agent_once", run_fabric_agent_once)
    stateful = asyncio.create_task(manager.invoke_session(session, FabricInvocationRequest(input="stateful")))
    await session_started.wait()
    one_shot = asyncio.create_task(manager.invoke_once(FabricInvocationRequest(input="one-shot")))
    await asyncio.sleep(0)

    assert invocation_order == ["stateful"]

    release_session.set()
    await asyncio.gather(stateful, one_shot)

    assert invocation_order == ["stateful", "one-shot"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", list(StreamingMethod))
async def test_stream_once_holds_concurrency_limit_until_runtime_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: StreamingMethod,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=FabricSessionRegistry(),
        max_concurrent_invocations=1,
        fabric=cast(Any, _FakeFabric(_FakeRuntime(), method=method)),
    )
    stream_started = asyncio.Event()
    release_stream = asyncio.Event()
    stream_cleaned_up = asyncio.Event()
    invocation_order: list[str] = []

    @asynccontextmanager
    async def stream_fabric_agent_once(request: Any, *, fabric: Any, streaming_method: StreamingMethod) -> Any:
        assert streaming_method is method
        invocation_order.append(request.input)
        try:
            yield object()
        finally:
            stream_started.set()
            await release_stream.wait()
            stream_cleaned_up.set()

    async def run_fabric_agent_once(request: Any, *, fabric: Any) -> FabricRuntimeResult:
        assert stream_cleaned_up.is_set()
        invocation_order.append(request.input)
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "stream_fabric_agent_once", stream_fabric_agent_once)
    monkeypatch.setattr(session_manager, "run_fabric_agent_once", run_fabric_agent_once)

    async def consume_stream() -> None:
        async with manager.stream_once(FabricInvocationRequest(input="stream")):
            pass

    streaming = asyncio.create_task(consume_stream())
    await stream_started.wait()
    one_shot = asyncio.create_task(manager.invoke_once(FabricInvocationRequest(input="one-shot")))
    await asyncio.sleep(0)

    assert invocation_order == ["stream"]

    release_stream.set()
    await asyncio.gather(streaming, one_shot)

    assert invocation_order == ["stream", "one-shot"]


@pytest.mark.asyncio
async def test_zero_concurrency_limit_allows_parallel_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = FabricSessionRegistry()
    first_session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-1")
    second_session = await registry.register(cast(Any, _FakeRuntime()), session_id="session-2")
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        max_concurrent_invocations=0,
    )
    both_started = asyncio.Event()
    release = asyncio.Event()
    active_invocations = 0

    async def invoke_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> FabricRuntimeResult:
        nonlocal active_invocations
        active_invocations += 1
        if active_invocations == 2:
            both_started.set()
        await release.wait()
        active_invocations -= 1
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "invoke_fabric_runtime", invoke_fabric_runtime)
    first = asyncio.create_task(
        manager.invoke_session(first_session, FabricInvocationRequest(input="first")),
    )
    second = asyncio.create_task(
        manager.invoke_session(second_session, FabricInvocationRequest(input="second")),
    )

    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    await asyncio.gather(first, second)
