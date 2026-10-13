# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient
from nemo_agents_plugin.agent_config import AgentConfig, AgentConfigLoadError
from nemo_agents_plugin.fabric import server
from nemo_agents_plugin.fabric.runtime import (
    FabricRuntimeExecutionError,
    FabricRuntimeResult,
    FabricRuntimeStartError,
    FabricRuntimeStream,
    FabricRuntimeTimeoutError,
)
from nemo_agents_plugin.fabric.server import SESSION_ID_HEADER, FabricServingSettings, create_fabric_serving_app
from nemo_agents_plugin.fabric.serving_models import ChatCompletionRequest
from nemo_agents_plugin.fabric.session_manager import (
    FabricSessionManager,
    FabricSessionStartError,
    FabricSessionStopError,
)
from nemo_agents_plugin.fabric.session_registry import FabricSessionNotFoundError, FabricSessionRegistry
from starlette.requests import ClientDisconnect


class _FakeStreamContext:
    def __init__(
        self,
        stream: Any = None,
        enter_error: BaseException | None = None,
        *,
        exit_checkpoint: bool = False,
        exit_waiter: asyncio.Event | None = None,
        exit_error: BaseException | None = None,
    ) -> None:
        self.stream = stream
        self.enter_error = enter_error
        self.exit_checkpoint = exit_checkpoint
        self.exit_waiter = exit_waiter
        self.exit_error = exit_error
        self.exit_calls = 0
        self.exit_completions = 0

    async def __aenter__(self) -> Any:
        if self.enter_error is not None:
            raise self.enter_error
        return self.stream

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exit_calls += 1
        if self.exit_checkpoint:
            await asyncio.sleep(0)
        if self.exit_waiter is not None:
            await self.exit_waiter.wait()
        if self.exit_error is not None:
            raise self.exit_error
        self.exit_completions += 1


class _FakeFabricStream:
    def __init__(
        self,
        records: list[dict[str, Any]] | None = None,
        *,
        result: FabricRuntimeResult | None = None,
        block_after_records: bool = False,
        close_checkpoint: bool = False,
    ) -> None:
        self._records = records or []
        self._result = result or FabricRuntimeResult(status="succeeded", response="done")
        self._block_after_records = block_after_records
        self._close_checkpoint = close_checkpoint
        self.aclose_calls = 0
        self.aclose_completions = 0

    async def records(self) -> Any:
        for record in self._records:
            yield record
        if self._block_after_records:
            await asyncio.Event().wait()

    async def result(self) -> FabricRuntimeResult:
        return self._result

    async def aclose(self) -> None:
        self.aclose_calls += 1
        if self._close_checkpoint:
            await asyncio.sleep(0)
        self.aclose_completions += 1


@pytest.fixture()
def mock_validate_agent_config(monkeypatch: pytest.MonkeyPatch) -> list[tuple[AgentConfig, Path]]:
    validation_calls: list[tuple[AgentConfig, Path]] = []

    async def validate(config: AgentConfig, *, base_dir: Path) -> object:
        validation_calls.append((config, base_dir))
        return object()

    monkeypatch.setattr(server, "_validate_agent_config", validate)
    return validation_calls


def _example_config() -> dict[str, Any]:
    return {
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


def _write_agent_config(tmp_path: Path, config: dict[str, Any] | None = None) -> Path:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.safe_dump(config or _example_config()), encoding="utf-8")
    return config_path


def _sse_payload(event: str) -> dict[str, object]:
    assert event.startswith("data: ")
    assert event.endswith("\n\n")
    return json.loads(event.removeprefix("data: ").removesuffix("\n\n"))


def _sse_line_payload(line: str) -> dict[str, object]:
    assert line.startswith("data: ")
    return json.loads(line.removeprefix("data: "))


def test_shutdown_removes_a_staged_agent_root(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
) -> None:
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    config_path = _write_agent_config(mounted)
    mounted.chmod(0o555)
    app = create_fabric_serving_app(config_path)
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            staged = Path(app.state.base_dir)

            assert staged != mounted
            assert (staged / "agent.yaml").exists()

        assert not staged.exists()
        assert config_path.exists()
    finally:
        mounted.chmod(0o755)


def test_failed_startup_removes_a_staged_agent_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mounted = tmp_path / "mounted"
    mounted.mkdir()
    config_path = _write_agent_config(mounted)
    mounted.chmod(0o555)
    staged: list[Path] = []
    real_resolve = server.resolve_runtime_base_dir

    def record(path: Path) -> Any:
        base_dir = real_resolve(path)
        staged.append(base_dir.path)
        return base_dir

    async def fail(config: AgentConfig, *, base_dir: Path) -> object:
        raise AgentConfigLoadError("invalid agent")

    monkeypatch.setattr(server, "resolve_runtime_base_dir", record)
    monkeypatch.setattr(server, "_validate_agent_config", fail)
    app = create_fabric_serving_app(config_path)
    try:
        with pytest.raises(AgentConfigLoadError), TestClient(app):
            pass

        assert len(staged) == 1
        assert not staged[0].exists()
    finally:
        mounted.chmod(0o755)


def test_startup_loads_and_validates_agent_config(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
) -> None:
    config_path = _write_agent_config(tmp_path)
    app = create_fabric_serving_app(config_path)

    with TestClient(app) as client:
        response = client.get("/health")
        repeated_response = client.get("/health")

        assert response.status_code == 200
        health = response.json()
        assert health["status"] == "ok"
        assert uuid.UUID(health["runtime_instance_id"])
        assert repeated_response.json()["runtime_instance_id"] == health["runtime_instance_id"]
        assert datetime.fromisoformat(health["runtime_started_at"]).tzinfo is not None
        assert repeated_response.json()["runtime_started_at"] == health["runtime_started_at"]
        assert app.state.agent_config.name == "test-agent"
        assert app.state.base_dir == tmp_path
        assert app.state.validation_result is not None
        assert isinstance(app.state.session_registry, FabricSessionRegistry)
        assert isinstance(app.state.session_manager, FabricSessionManager)

    assert mock_validate_agent_config == [(app.state.agent_config, tmp_path)]


@pytest.mark.parametrize("socket_fd", [None, 123])
def test_main_starts_packaged_server(tmp_path: Path, socket_fd: int | None) -> None:
    config_path = _write_agent_config(tmp_path)
    app = object()

    with (
        patch("nemo_agents_plugin.fabric.server.create_fabric_serving_app", return_value=app) as create_app,
        patch("uvicorn.run") as run,
    ):
        result = server.main(
            [
                "--agent-config",
                str(config_path),
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
            + (["--socket-fd", str(socket_fd)] if socket_fd is not None else [])
        )

    assert result == 0
    create_app.assert_called_once_with(config_path, settings=FabricServingSettings())
    run.assert_called_once_with(app, host="0.0.0.0", port=8000, fd=socket_fd, log_config=None)


def test_shutdown_stops_all_registered_runtimes(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
) -> None:
    class _Runtime:
        def __init__(self) -> None:
            self.stop_calls = 0

        async def stop(self) -> None:
            self.stop_calls += 1

    runtime = _Runtime()
    app = create_fabric_serving_app(_write_agent_config(tmp_path))

    with TestClient(app) as client:
        registry = app.state.session_registry

        async def register_runtime() -> None:
            await registry.register(cast(Any, runtime), session_id="session-1")

        assert client.portal is not None
        client.portal.call(register_runtime)

    assert runtime.stop_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_error", [RuntimeError("cleanup failed"), asyncio.CancelledError()])
async def test_shutdown_stops_sessions_when_cleanup_task_fails(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
    cleanup_error: BaseException,
) -> None:
    close_calls = 0

    async def fail_cleanup(*args: Any, **kwargs: Any) -> None:
        raise cleanup_error

    async def close_all_sessions(self: FabricSessionManager) -> int:
        nonlocal close_calls
        close_calls += 1
        return 0

    monkeypatch.setattr(server, "_run_idle_session_cleanup", fail_cleanup)
    monkeypatch.setattr(FabricSessionManager, "close_all_sessions", close_all_sessions)
    app = create_fabric_serving_app(_write_agent_config(tmp_path))

    with pytest.raises(type(cleanup_error)):
        async with app.router.lifespan_context(app):
            pass

    assert close_calls == 1


def test_startup_fails_for_invalid_agent_config(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
) -> None:
    config_path = _write_agent_config(tmp_path, {"name": "invalid"})
    app = create_fabric_serving_app(config_path)

    with pytest.raises(AgentConfigLoadError), TestClient(app):
        pass

    assert mock_validate_agent_config == []


def test_rejects_invalid_serving_settings() -> None:
    with pytest.raises(ValueError, match="max_concurrent_invocations"):
        FabricServingSettings(max_concurrent_invocations=-1)
    with pytest.raises(ValueError, match="idle_session_timeout_seconds"):
        FabricServingSettings(idle_session_timeout_seconds=0)
    with pytest.raises(ValueError, match="session_cleanup_interval_seconds"):
        FabricServingSettings(session_cleanup_interval_seconds=0)


def test_chat_completion_without_session_id_invokes_once_without_creating_session(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_agent_config(tmp_path)
    app = create_fabric_serving_app(config_path)
    invocation_calls: list[Any] = []

    async def resolve_session(session_id: str) -> Any:
        pytest.fail(f"headerless request resolved session {session_id!r}")

    async def invoke_once(request: Any) -> FabricRuntimeResult:
        invocation_calls.append(request)
        return FabricRuntimeResult(
            status="succeeded",
            output={"response": "hello", "usage": {"total_tokens": 3}},
            response="hello",
            invocation_id="invocation-1",
        )

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "resolve_session", resolve_session)
        monkeypatch.setattr(app.state.session_manager, "invoke_once", invoke_once)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        repeated_response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "again"}]},
        )

    assert response.status_code == 200
    assert repeated_response.status_code == 200
    assert SESSION_ID_HEADER not in response.headers
    assert SESSION_ID_HEADER not in repeated_response.headers
    assert response.json() == {
        "id": "invocation-1",
        "object": "chat.completion",
        "model": "unknown-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 3},
    }
    assert [request.input for request in invocation_calls] == ["hello", "again"]
    assert all(request.caller_context == {} for request in invocation_calls)
    assert asyncio.run(app.state.session_registry.count()) == 0


def test_chat_completion_with_session_id_reuses_session(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_agent_config(tmp_path)
    app = create_fabric_serving_app(config_path)
    resolve_calls: list[str] = []

    async def resolve_session(session_id: str) -> Any:
        resolve_calls.append(session_id)
        return SimpleNamespace(session_id="session-1", runtime=object())

    async def invoke_session(session: Any, request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(status="succeeded", response="hello again")

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "resolve_session", resolve_session)
        monkeypatch.setattr(app.state.session_manager, "invoke_session", invoke_session)
        response = client.post(
            "/v1/chat/completions",
            headers={SESSION_ID_HEADER: "session-1"},
            json={"messages": [{"role": "user", "content": "hello again"}]},
        )

    assert response.status_code == 200
    assert response.headers[SESSION_ID_HEADER] == "session-1"
    assert resolve_calls == ["session-1"]


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (FabricRuntimeStartError("startup failed"), 503),
        (FabricRuntimeTimeoutError("timed out"), 504),
        (FabricRuntimeExecutionError("invoke failed"), 502),
    ],
)
def test_chat_completion_maps_runtime_errors(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
) -> None:
    app = create_fabric_serving_app(_write_agent_config(tmp_path))

    async def invoke_once(request: Any) -> FabricRuntimeResult:
        raise error

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "invoke_once", invoke_once)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == status_code
    assert SESSION_ID_HEADER not in response.headers
    assert response.json() == {"detail": str(error)}


def test_session_chat_completion_runtime_error_returns_session_header(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_fabric_serving_app(_write_agent_config(tmp_path))

    async def resolve_session(session_id: str) -> Any:
        return SimpleNamespace(session_id=session_id, runtime=object())

    async def invoke_session(session: Any, request: Any) -> FabricRuntimeResult:
        raise FabricRuntimeTimeoutError("timed out")

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "resolve_session", resolve_session)
        monkeypatch.setattr(app.state.session_manager, "invoke_session", invoke_session)
        response = client.post(
            "/v1/chat/completions",
            headers={SESSION_ID_HEADER: "session-1"},
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 504
    assert response.headers[SESSION_ID_HEADER] == "session-1"
    assert response.json() == {"detail": "timed out"}


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (FabricSessionNotFoundError("missing session"), 404),
        (FabricSessionStartError("startup failed"), 503),
    ],
)
def test_chat_completion_maps_session_resolution_errors(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
) -> None:
    app = create_fabric_serving_app(_write_agent_config(tmp_path))

    async def resolve_session(session_id: str) -> Any:
        raise error

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "resolve_session", resolve_session)
        response = client.post(
            "/v1/chat/completions",
            headers={SESSION_ID_HEADER: "missing"},
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == status_code
    assert SESSION_ID_HEADER not in response.headers
    assert response.json() == {"detail": str(error)}


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (FabricRuntimeStartError("startup failed"), 503),
        (FabricRuntimeExecutionError("stream failed"), 502),
    ],
)
def test_streaming_chat_completion_without_session_maps_stream_start_errors(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
) -> None:
    app = create_fabric_serving_app(_write_agent_config(tmp_path))

    def stream_once(request: Any) -> _FakeStreamContext:
        return _FakeStreamContext(enter_error=error)

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "stream_once", stream_once)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        )

    assert response.status_code == status_code
    assert SESSION_ID_HEADER not in response.headers
    assert response.json() == {"detail": str(error)}


def test_streaming_chat_completion_without_session_uses_one_shot_stream(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_fabric_serving_app(_write_agent_config(tmp_path))
    stream_calls: list[Any] = []
    fabric_stream = _FakeFabricStream(
        [
            {"kind": "scope", "scope_category": "start", "name": "request"},
            {"data": {"choices": [{"delta": {"content": "hel"}}]}},
            {"data": {"type": "agentMessage", "phase": "final_answer", "text": "lo"}},
        ],
        result=FabricRuntimeResult(status="succeeded", response="hello"),
    )
    stream_context = _FakeStreamContext(fabric_stream)

    def stream_once(request: Any) -> _FakeStreamContext:
        stream_calls.append(request)
        return stream_context

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "stream_once", stream_once)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True, "model": "test-model"},
        ) as response:
            lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert SESSION_ID_HEADER not in response.headers
    invocation_request = stream_calls[0]
    assert invocation_request.input == "hello"
    assert invocation_request.caller_context == {}
    assert stream_context.exit_calls == 1
    assert fabric_stream.aclose_calls == 0

    assert lines[-1] == "data: [DONE]"
    assert _sse_line_payload(lines[0])["choices"] == [{"index": 0, "delta": {"role": "assistant"}}]
    assert _sse_line_payload(lines[1])["choices"] == [{"index": 0, "delta": {"content": "hel"}}]
    assert _sse_line_payload(lines[2])["choices"] == [{"index": 0, "delta": {"content": "lo"}}]
    assert _sse_line_payload(lines[3])["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]


def test_streaming_chat_completion_reuses_supplied_session_id(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_fabric_serving_app(_write_agent_config(tmp_path))
    resolve_calls: list[str] = []

    async def resolve_session(session_id: str) -> Any:
        resolve_calls.append(session_id)
        return SimpleNamespace(session_id="session-1", runtime=object())

    def stream_session(session: Any, request: Any) -> _FakeStreamContext:
        return _FakeStreamContext(_FakeFabricStream(result=FabricRuntimeResult(status="succeeded", response="done")))

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "resolve_session", resolve_session)
        monkeypatch.setattr(app.state.session_manager, "stream_session", stream_session)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            headers={SESSION_ID_HEADER: "session-1"},
            json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
        ) as response:
            lines = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert response.headers[SESSION_ID_HEADER] == "session-1"
    assert resolve_calls == ["session-1"]
    assert _sse_line_payload(lines[1])["choices"] == [{"index": 0, "delta": {"content": "done"}}]


@pytest.mark.asyncio
async def test_streaming_chat_completion_emits_sse_error_and_closes_stream() -> None:
    fabric_stream = _FakeFabricStream(
        [{"data": {"choices": [{"delta": {"content": "partial"}}]}}],
        result=FabricRuntimeResult(
            status="failed",
            error={"message": "terminal failure"},
        ),
    )
    stream_context = _FakeStreamContext(fabric_stream)

    events = [
        event
        async for event in server._iter_streaming_chat_completion(
            stream_context,
            cast(FabricRuntimeStream, fabric_stream),
            completion_id="chatcmpl-test",
            model="test-model",
        )
    ]

    assert _sse_payload(events[1])["choices"] == [{"index": 0, "delta": {"content": "partial"}}]
    assert _sse_payload(events[2]) == {
        "error": {
            "message": "terminal failure",
            "type": "FabricStreamResultError",
        }
    }
    assert fabric_stream.aclose_calls == 1
    assert stream_context.exit_calls == 1


@pytest.mark.asyncio
async def test_streaming_chat_completion_closes_stream_on_generator_close() -> None:
    fabric_stream = _FakeFabricStream([{"data": {"choices": [{"delta": {"content": "partial"}}]}}])
    stream_context = _FakeStreamContext(fabric_stream)
    events = server._iter_streaming_chat_completion(
        stream_context,
        cast(FabricRuntimeStream, fabric_stream),
        completion_id="chatcmpl-test",
        model="test-model",
    )

    assert _sse_payload(await events.__anext__())["choices"] == [{"index": 0, "delta": {"role": "assistant"}}]

    await cast(AsyncGenerator[str], events).aclose()

    assert fabric_stream.aclose_calls == 1
    assert stream_context.exit_calls == 1


@pytest.mark.asyncio
async def test_streaming_chat_completion_closes_stream_before_first_event() -> None:
    fabric_stream = _FakeFabricStream([{"data": {"choices": [{"delta": {"content": "partial"}}]}}])
    stream_context = _FakeStreamContext(fabric_stream)
    events = server._iter_streaming_chat_completion(
        stream_context,
        cast(FabricRuntimeStream, fabric_stream),
        completion_id="chatcmpl-test",
        model="test-model",
    )

    await cast(AsyncGenerator[str], events).aclose()

    assert fabric_stream.aclose_calls == 1
    assert stream_context.exit_calls == 1


@pytest.mark.asyncio
async def test_streaming_response_closes_stream_after_client_disconnect() -> None:
    fabric_stream = _FakeFabricStream([{"data": {"choices": [{"delta": {"content": "partial"}}]}}])
    stream_context = _FakeStreamContext(fabric_stream)
    iterator = server._iter_streaming_chat_completion(
        stream_context,
        cast(FabricRuntimeStream, fabric_stream),
        completion_id="chatcmpl-test",
        model="test-model",
    )
    response = server._FabricStreamingResponse(iterator, media_type="text/event-stream")

    async def receive() -> Any:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    scope = cast(Any, {"type": "http", "asgi": {"spec_version": "2.4"}})
    with pytest.raises(ClientDisconnect):
        await response(scope, receive, cast(Any, send))

    assert fabric_stream.aclose_calls == 1
    assert stream_context.exit_calls == 1


@pytest.mark.asyncio
async def test_streaming_response_cleanup_failure_does_not_mask_client_disconnect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cleanup_error = RuntimeError("cleanup failed")
    fabric_stream = _FakeFabricStream()
    stream_context = _FakeStreamContext(fabric_stream, exit_error=cleanup_error)
    iterator = server._iter_streaming_chat_completion(
        stream_context,
        cast(FabricRuntimeStream, fabric_stream),
        completion_id="chatcmpl-test",
        model="test-model",
    )
    response = server._FabricStreamingResponse(iterator, media_type="text/event-stream")

    async def receive() -> Any:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    scope = cast(Any, {"type": "http", "asgi": {"spec_version": "2.4"}})
    with caplog.at_level("ERROR"):
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, cast(Any, send))

    assert "Fabric stream cleanup failed for completion chatcmpl-test." in caplog.text
    assert iterator._closed
    await iterator.aclose()


@pytest.mark.asyncio
async def test_streaming_response_completes_cleanup_after_asgi_cancellation() -> None:
    fabric_stream = _FakeFabricStream(
        [{"data": {"choices": [{"delta": {"content": "partial"}}]}}],
        block_after_records=True,
        close_checkpoint=True,
    )
    stream_context = _FakeStreamContext(fabric_stream, exit_checkpoint=True)
    iterator = server._iter_streaming_chat_completion(
        stream_context,
        cast(FabricRuntimeStream, fabric_stream),
        completion_id="chatcmpl-test",
        model="test-model",
    )
    response = server._FabricStreamingResponse(iterator, media_type="text/event-stream")
    body_sent = asyncio.Event()

    async def receive() -> Any:
        await body_sent.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            body_sent.set()

    scope = cast(Any, {"type": "http", "asgi": {"spec_version": "2.3"}})
    await asyncio.wait_for(response(scope, receive, cast(Any, send)), timeout=1)

    assert fabric_stream.aclose_calls == 1
    assert fabric_stream.aclose_completions == 1
    assert stream_context.exit_calls == 1
    assert stream_context.exit_completions == 1


@pytest.mark.asyncio
async def test_streaming_response_bounds_cleanup_wait_without_cancelling_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cleanup_release = asyncio.Event()
    fabric_stream = _FakeFabricStream()
    stream_context = _FakeStreamContext(fabric_stream, exit_waiter=cleanup_release)
    iterator = server._iter_streaming_chat_completion(
        stream_context,
        cast(FabricRuntimeStream, fabric_stream),
        completion_id="chatcmpl-test",
        model="test-model",
    )
    response = server._FabricStreamingResponse(iterator, media_type="text/event-stream")
    monkeypatch.setattr(server, "_FABRIC_STREAM_CLEANUP_TIMEOUT_SECONDS", 0.01)

    async def receive() -> Any:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    scope = cast(Any, {"type": "http", "asgi": {"spec_version": "2.4"}})
    with caplog.at_level("WARNING"):
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, cast(Any, send))

    assert stream_context.exit_calls == 1
    assert stream_context.exit_completions == 0
    assert "cleanup will continue in the background" in caplog.text

    cleanup_release.set()
    assert iterator._close_task is not None
    await iterator._close_task

    assert stream_context.exit_completions == 1


@pytest.mark.asyncio
async def test_streaming_response_bounds_cleanup_after_direct_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cleanup_release = asyncio.Event()
    fabric_stream = _FakeFabricStream(
        [{"data": {"choices": [{"delta": {"content": "partial"}}]}}],
        block_after_records=True,
    )
    stream_context = _FakeStreamContext(fabric_stream, exit_waiter=cleanup_release)
    iterator = server._iter_streaming_chat_completion(
        stream_context,
        cast(FabricRuntimeStream, fabric_stream),
        completion_id="chatcmpl-test",
        model="test-model",
    )
    response = server._FabricStreamingResponse(iterator, media_type="text/event-stream")
    body_sent = asyncio.Event()
    monkeypatch.setattr(server, "_FABRIC_STREAM_CLEANUP_TIMEOUT_SECONDS", 0.01)

    async def receive() -> Any:
        await asyncio.Event().wait()

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            body_sent.set()

    scope = cast(Any, {"type": "http", "asgi": {"spec_version": "2.4"}})
    response_task = asyncio.create_task(response(scope, receive, cast(Any, send)))
    await body_sent.wait()
    response_task.cancel()

    with caplog.at_level("WARNING"):
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(response_task, timeout=1)

    assert stream_context.exit_calls == 1
    assert stream_context.exit_completions == 0
    assert "cleanup will continue in the background" in caplog.text

    cleanup_release.set()
    assert iterator._close_task is not None
    await iterator._close_task

    assert stream_context.exit_completions == 1


@pytest.mark.asyncio
async def test_streaming_response_logs_cleanup_failure_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cleanup_release = asyncio.Event()
    cleanup_error = RuntimeError("cleanup failed")
    fabric_stream = _FakeFabricStream()
    stream_context = _FakeStreamContext(
        fabric_stream,
        exit_waiter=cleanup_release,
        exit_error=cleanup_error,
    )
    iterator = server._iter_streaming_chat_completion(
        stream_context,
        cast(FabricRuntimeStream, fabric_stream),
        completion_id="chatcmpl-test",
        model="test-model",
    )
    response = server._FabricStreamingResponse(iterator, media_type="text/event-stream")
    monkeypatch.setattr(server, "_FABRIC_STREAM_CLEANUP_TIMEOUT_SECONDS", 0.01)

    async def receive() -> Any:
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    scope = cast(Any, {"type": "http", "asgi": {"spec_version": "2.4"}})
    with caplog.at_level("WARNING"):
        with pytest.raises(ClientDisconnect):
            await response(scope, receive, cast(Any, send))

        cleanup_release.set()
        assert iterator._close_task is not None
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await iterator._close_task
        await asyncio.sleep(0)

    assert "Background Fabric stream cleanup failed for completion chatcmpl-test." in caplog.text
    assert iterator._closed
    await iterator.aclose()


def test_chat_completion_maps_failed_run_result(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_fabric_serving_app(_write_agent_config(tmp_path))

    async def invoke_once(request: Any) -> FabricRuntimeResult:
        return FabricRuntimeResult(
            status="failed",
            error={"stage": "invoke", "message": "adapter failed"},
        )

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "invoke_once", invoke_once)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 502
    assert SESSION_ID_HEADER not in response.headers
    assert response.json() == {"detail": "adapter failed"}


def test_chat_completion_request_preserves_single_user_turn() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Say hello."}],
            "model": "test-model",
            "stream": False,
        }
    )

    invocation_request = server._to_fabric_invocation_request(request, session_id="session-1")

    assert invocation_request.input == "Say hello."
    assert invocation_request.caller_context == {"session_id": "session-1"}


def test_chat_completion_request_serializes_full_transcript() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "assistant", "content": "How can I help?"},
                {"role": "user", "content": "Say hello."},
            ],
            "model": "test-model",
            "stream": False,
        }
    )

    invocation_request = server._to_fabric_invocation_request(request, session_id="session-1")

    assert invocation_request.input == ("system: Be concise.\n\nassistant: How can I help?\n\nuser: Say hello.")
    assert invocation_request.caller_context == {"session_id": "session-1"}


def test_stateless_chat_completion_request_serializes_full_transcript_without_session_context() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "assistant", "content": "How can I help?"},
                {"role": "user", "content": "Say hello."},
            ]
        }
    )

    invocation_request = server._to_fabric_invocation_request(request, session_id=None)

    assert invocation_request.input == ("system: Be concise.\n\nassistant: How can I help?\n\nuser: Say hello.")
    assert invocation_request.caller_context == {}


def test_close_session_stops_registered_runtime(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_fabric_serving_app(_write_agent_config(tmp_path))
    close_calls: list[str] = []

    async def close_session(session_id: str) -> None:
        close_calls.append(session_id)

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "close_session", close_session)
        response = client.delete("/v1/sessions/session-1")

    assert response.status_code == 204
    assert response.content == b""
    assert close_calls == ["session-1"]


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (FabricSessionNotFoundError("missing session"), 404),
        (FabricSessionStopError("shutdown failed"), 502),
    ],
)
def test_close_session_maps_errors(
    tmp_path: Path,
    mock_validate_agent_config: list[tuple[AgentConfig, Path]],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
) -> None:
    app = create_fabric_serving_app(_write_agent_config(tmp_path))

    async def close_session(session_id: str) -> None:
        raise error

    with TestClient(app) as client:
        monkeypatch.setattr(app.state.session_manager, "close_session", close_session)
        response = client.delete("/v1/sessions/session-1")

    assert response.status_code == status_code
    assert response.json() == {"detail": str(error)}


@pytest.mark.asyncio
async def test_idle_cleanup_runs_periodically_until_shutdown() -> None:
    cleanup_calls: list[float] = []
    cleanup_ran = asyncio.Event()
    shutdown = asyncio.Event()

    class _Manager:
        async def expire_idle_sessions(self, *, idle_timeout_seconds: float) -> int:
            cleanup_calls.append(idle_timeout_seconds)
            cleanup_ran.set()
            return 0

    cleanup = asyncio.create_task(
        server._run_idle_session_cleanup(
            cast(Any, _Manager()),
            idle_timeout_seconds=30.0,
            cleanup_interval_seconds=0.01,
            shutdown_event=shutdown,
        )
    )
    await asyncio.wait_for(cleanup_ran.wait(), timeout=1)
    shutdown.set()
    await cleanup

    assert cleanup_calls == [30.0]
