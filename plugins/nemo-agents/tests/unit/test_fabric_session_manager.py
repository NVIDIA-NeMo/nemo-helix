# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric import session_manager
from nemo_agents_plugin.fabric.runtime import FabricInvocationRequest, FabricRuntimeResult
from nemo_agents_plugin.fabric.session_manager import FabricSessionManager
from nemo_agents_plugin.fabric.session_registry import (
    FabricSessionNotFoundError,
    FabricSessionRegistry,
)


class _FakeRuntime:
    def __init__(self) -> None:
        self.stop_calls = 0

    async def stop(self) -> None:
        self.stop_calls += 1


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
    def __init__(self, runtime: _FakeRuntime) -> None:
        self.runtime = runtime
        self.start_calls: list[dict[str, Any]] = []

    async def start_runtime(self, config: Any, *, base_dir: Path, streaming: bool = False) -> _FakeRuntime:
        self.start_calls.append({"base_dir": base_dir, "config": config, "streaming": streaming})
        return self.runtime


class _SequenceFabric:
    def __init__(self, runtimes: list[_FakeRuntime]) -> None:
        self.runtimes = iter(runtimes)
        self.start_calls: list[dict[str, Any]] = []

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


@pytest.mark.asyncio
async def test_open_session_materializes_config_and_starts_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric_config = _FakeFabricConfig()
    translation_calls: list[AgentConfig] = []

    def translate(config: AgentConfig) -> Any:
        translation_calls.append(config)
        return fabric_config

    monkeypatch.setattr(session_manager, "translate_agent_config", translate)
    runtime = _FakeRuntime()
    fabric = _FakeFabric(runtime)
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
    assert fabric.start_calls[0]["streaming"] is True
    started_config = fabric.start_calls[0]["config"]
    assert started_config is not fabric_config
    assert started_config.copied is True
    assert started_config.relay_enabled is True
    assert session.runtime is runtime
    assert await registry.get(session.session_id) is session


@pytest.mark.asyncio
async def test_open_session_stops_runtime_when_registration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: _FakeFabricConfig())
    runtime = _FakeRuntime()
    fabric = _FakeFabric(runtime)
    registry = FabricSessionRegistry()

    async def fail_registration(runtime: Any, *, session_id: str | None = None) -> None:
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
async def test_invoke_once_uses_unregistered_runtime(
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

    result = await manager.invoke_once(
        FabricInvocationRequest(input="hello", relay_session_root="018f47a4-0000-7d94-8e61-9f0f89b5d312")
    )

    assert result.response == "hello"
    request, captured_fabric = captured[0]
    assert request.fabric_config is fabric_config
    assert request.input == "hello"
    assert request.caller_context == {}
    assert request.relay_session_root == "018f47a4-0000-7d94-8e61-9f0f89b5d312"
    assert captured_fabric is fabric
    assert await registry.count() == 0


@pytest.mark.asyncio
async def test_stream_once_uses_relay_enabled_unregistered_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric_config = _FakeFabricConfig()
    monkeypatch.setattr(session_manager, "translate_agent_config", lambda config: fabric_config)
    captured: list[tuple[Any, Any]] = []
    expected_stream = object()

    @asynccontextmanager
    async def stream_fabric_agent_once(request: Any, *, fabric: Any) -> Any:
        captured.append((request, fabric))
        yield expected_stream

    monkeypatch.setattr(session_manager, "stream_fabric_agent_once", stream_fabric_agent_once)
    registry = FabricSessionRegistry()
    fabric = object()
    manager = FabricSessionManager(
        _agent_config(),
        base_dir=tmp_path,
        session_registry=registry,
        fabric=cast(Any, fabric),
    )

    async with manager.stream_once(FabricInvocationRequest(input="hello")) as stream:
        assert stream is expected_stream

    request, captured_fabric = captured[0]
    assert request.fabric_config is not fabric_config
    assert request.fabric_config.copied is True
    assert request.fabric_config.relay_enabled is True
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
async def test_close_session_waits_for_active_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime()
    registry = FabricSessionRegistry()
    session = await registry.register(cast(Any, runtime), session_id="session-1")
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
    with pytest.raises(FabricSessionNotFoundError, match="session-1"):
        await manager.resolve_session(session.session_id)

    release_invocation.set()
    await invocation
    await close

    assert runtime.stop_calls == 1


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
async def test_stream_session_refreshes_activity(
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
    stream = object()
    refresh_calls: list[Any] = []

    def stream_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> object:
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
async def test_stream_session_serializes_turns_for_same_runtime(
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

    def stream_fabric_runtime(runtime: Any, request: FabricInvocationRequest) -> object:
        invocation_order.append(request.input)
        return object()

    monkeypatch.setattr(session_manager, "stream_fabric_runtime", stream_fabric_runtime)

    async def stream_first() -> None:
        async with manager.stream_session(session, FabricInvocationRequest(input="first")):
            first_started.set()
            await release_first.wait()

    async def stream_second() -> None:
        async with manager.stream_session(session, FabricInvocationRequest(input="second")):
            pass

    first = asyncio.create_task(stream_first())
    await first_started.wait()
    second = asyncio.create_task(stream_second())
    await asyncio.sleep(0)

    assert invocation_order == ["first"]

    release_first.set()
    await asyncio.gather(first, second)

    assert invocation_order == ["first", "second"]


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
async def test_stream_once_holds_concurrency_limit_until_runtime_cleanup(
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
    stream_started = asyncio.Event()
    release_stream = asyncio.Event()
    stream_cleaned_up = asyncio.Event()
    invocation_order: list[str] = []

    @asynccontextmanager
    async def stream_fabric_agent_once(request: Any, *, fabric: Any) -> Any:
        invocation_order.append(request.input)
        try:
            yield object()
        finally:
            stream_cleaned_up.set()

    async def run_fabric_agent_once(request: Any, *, fabric: Any) -> FabricRuntimeResult:
        assert stream_cleaned_up.is_set()
        invocation_order.append(request.input)
        return FabricRuntimeResult(status="succeeded", response=request.input)

    monkeypatch.setattr(session_manager, "stream_fabric_agent_once", stream_fabric_agent_once)
    monkeypatch.setattr(session_manager, "run_fabric_agent_once", run_fabric_agent_once)

    async def consume_stream() -> None:
        async with manager.stream_once(FabricInvocationRequest(input="stream")):
            stream_started.set()
            await release_stream.wait()

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
