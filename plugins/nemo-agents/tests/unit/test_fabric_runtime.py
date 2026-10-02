# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Fabric runtime execution helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from nemo_agents_plugin.fabric import runtime as fabric_runtime
from nemo_agents_plugin.fabric.runtime import (
    FabricInvocationRequest,
    FabricOneShotRequest,
    FabricRuntimeExecutionError,
    FabricRuntimeStartError,
    FabricRuntimeTimeoutError,
    invoke_fabric_runtime,
    run_fabric_agent_once,
    stream_fabric_agent_once,
    stream_fabric_runtime,
)
from nemo_fabric import FabricConfig, HarnessConfig, MetadataConfig


class _FabricMapping:
    def __init__(self, mapping: dict[str, Any]) -> None:
        self._mapping = mapping

    def to_mapping(self) -> dict[str, Any]:
        return self._mapping


class _FakeRunResult:
    def __init__(
        self,
        *,
        status: str = "succeeded",
        output: Any | None = None,
        error: Any | None = None,
    ) -> None:
        self.status = status
        self.output = output if output is not None else _FabricMapping({"response": "hello"})
        self.error = error
        self.artifacts = _FabricMapping({"root": "/tmp/artifacts", "artifacts": []})
        self.telemetry = (_FabricMapping({"provider": "relay", "kind": "trace"}),)
        self.events = (_FabricMapping({"kind": "runtime_start", "message": "started"}),)
        self.metadata = {"adapter_runner": "python"}
        self.runtime_id = "runtime-1"
        self.invocation_id = "invocation-1"
        self.request_id = "request-1"


class _FakeRuntime:
    def __init__(
        self,
        *,
        result: Any | None = None,
        invoke_error: Exception | None = None,
        invoke_delay: float = 0.0,
        enter_error: Exception | None = None,
        enter_delay: float = 0.0,
        exit_error: Exception | None = None,
        stream: _FakeInvokeStream | None = None,
    ) -> None:
        self.result = result if result is not None else _FakeRunResult()
        self.invoke_error = invoke_error
        self.invoke_delay = invoke_delay
        self.enter_error = enter_error
        self.enter_delay = enter_delay
        self.exit_error = exit_error
        self.entered = False
        self.exited = False
        self.exit_calls = 0
        self.runtime_id = "runtime-1"
        self.invoke_requests: list[Any] = []
        self.invoke_stream_requests: list[Any] = []
        self.stream = stream

    async def __aenter__(self) -> "_FakeRuntime":
        if self.enter_delay:
            await asyncio.sleep(self.enter_delay)
        if self.enter_error is not None:
            raise self.enter_error
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        self.exited = True
        self.exit_calls += 1
        if self.exit_error is not None:
            raise self.exit_error

    async def invoke(self, *, request: Any) -> Any:
        self.invoke_requests.append(request)
        if self.invoke_delay:
            await asyncio.sleep(self.invoke_delay)
        if self.invoke_error is not None:
            raise self.invoke_error
        return self.result

    def invoke_stream(self, *, request: Any) -> "_FakeInvokeStream":
        self.invoke_stream_requests.append(request)
        if self.invoke_error is not None:
            raise self.invoke_error
        self.stream = self.stream or _FakeInvokeStream(result=self.result)
        return self.stream


class _FakeInvokeStream:
    def __init__(
        self,
        *,
        records: list[dict[str, Any]] | None = None,
        result: Any | None = None,
        result_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.records = list(records or [{"type": "span", "message": "thinking"}])
        self.result_value = result if result is not None else _FakeRunResult()
        self.result_error = result_error
        self.close_error = close_error
        self.close_calls = 0

    def __aiter__(self) -> "_FakeInvokeStream":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self.records:
            raise StopAsyncIteration
        return self.records.pop(0)

    async def result(self) -> Any:
        if self.result_error is not None:
            raise self.result_error
        return self.result_value

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _FakeFabric:
    def __init__(
        self,
        *,
        runtime: _FakeRuntime | None = None,
        start_error: Exception | None = None,
        start_delay: float = 0.0,
    ) -> None:
        self.runtime = runtime if runtime is not None else _FakeRuntime()
        self.start_error = start_error
        self.start_delay = start_delay
        self.start_calls: list[dict[str, Any]] = []

    async def start_runtime(
        self,
        fabric_config: Any,
        *,
        base_dir: Path | str,
        overrides: dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> _FakeRuntime:
        call = {
            "base_dir": base_dir,
            "fabric_config": fabric_config,
            "overrides": overrides,
        }
        if streaming:
            call["streaming"] = True
        self.start_calls.append(call)
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        if self.start_error is not None:
            raise self.start_error
        return self.runtime


class _SequenceFabric:
    def __init__(self, runtimes: list[_FakeRuntime]) -> None:
        self._runtimes = iter(runtimes)

    async def start_runtime(
        self,
        fabric_config: Any,
        *,
        base_dir: Path | str,
        overrides: dict[str, Any] | None = None,
        streaming: bool = False,
    ) -> _FakeRuntime:
        return next(self._runtimes)


@pytest.mark.asyncio
class TestRunFabricAgentOnce:
    async def test_starts_invokes_and_cleans_up_ephemeral_runtime(self) -> None:
        fabric_config = cast(FabricConfig, object())
        fake_runtime = _FakeRuntime()
        fake_fabric = _FakeFabric(runtime=fake_runtime)
        request = FabricOneShotRequest(
            fabric_config=fabric_config,
            base_dir=Path("/tmp/agent"),
            input={"prompt": "hi"},
            request_id="platform-request-1",
            caller_context={"session_id": "session-1"},
            overrides={"models": {"default": {"temperature": 0.1}}},
        )

        result = await run_fabric_agent_once(request, fabric=fake_fabric)

        assert fake_fabric.start_calls == [
            {
                "base_dir": Path("/tmp/agent"),
                "fabric_config": fabric_config,
                "overrides": {"models": {"default": {"temperature": 0.1}}},
            }
        ]
        assert fake_runtime.entered is True
        assert fake_runtime.exited is True
        assert fake_runtime.exit_calls == 1
        fabric_request = fake_runtime.invoke_requests[0]
        assert fabric_request.input == {"prompt": "hi"}
        assert fabric_request.request_id == "platform-request-1"
        assert fabric_request.context == {"session_id": "session-1"}
        assert result.status == "succeeded"
        assert result.response == "hello"
        assert result.runtime_id == "runtime-1"
        assert result.invocation_id == "invocation-1"
        assert result.request_id == "request-1"

    async def test_wraps_timeout(self) -> None:
        fake_runtime = _FakeRuntime(invoke_delay=1.0)
        fake_fabric = _FakeFabric(runtime=fake_runtime)
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
            timeout_seconds=0.01,
        )

        with pytest.raises(FabricRuntimeTimeoutError, match="Fabric runtime invocation timed out after 0.01s"):
            await run_fabric_agent_once(request, fabric=fake_fabric)

        assert fake_runtime.exited is True
        assert fake_runtime.exit_calls == 1

    async def test_runtime_setup_does_not_consume_invocation_timeout(self) -> None:
        fake_runtime = _FakeRuntime(enter_delay=0.06, invoke_delay=0.06)
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
            timeout_seconds=0.1,
        )

        result = await run_fabric_agent_once(
            request,
            fabric=_FakeFabric(runtime=fake_runtime, start_delay=0.06),
        )

        assert result.status == "succeeded"
        assert fake_runtime.exit_calls == 1

    async def test_wraps_runtime_timeout_without_configured_deadline(self) -> None:
        timeout_error = TimeoutError("adapter timed out")
        fake_runtime = _FakeRuntime(invoke_error=timeout_error)
        fake_fabric = _FakeFabric(runtime=fake_runtime)
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
        )

        with pytest.raises(FabricRuntimeTimeoutError, match=r"Fabric runtime invocation timed out\.$") as exc_info:
            await run_fabric_agent_once(request, fabric=fake_fabric)

        assert exc_info.value.__cause__ is timeout_error

    async def test_wraps_runtime_start_errors(self) -> None:
        fake_fabric = _FakeFabric(start_error=fabric_runtime.FabricError("native unavailable"))
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
        )

        with pytest.raises(FabricRuntimeStartError, match="Fabric runtime startup failed: native unavailable"):
            await run_fabric_agent_once(request, fabric=fake_fabric)

    async def test_wraps_runtime_context_entry_errors_as_start_errors(self) -> None:
        fake_runtime = _FakeRuntime(enter_error=fabric_runtime.FabricError("harness unavailable"))
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
        )

        with pytest.raises(FabricRuntimeStartError, match="Fabric runtime startup failed: harness unavailable"):
            await run_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime))

        assert fake_runtime.entered is False
        assert fake_runtime.exit_calls == 0

    async def test_uses_a_separate_runtime_for_each_invocation(self) -> None:
        first_runtime = _FakeRuntime()
        second_runtime = _FakeRuntime()
        fabric = _SequenceFabric([first_runtime, second_runtime])
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
        )

        await run_fabric_agent_once(request, fabric=fabric)
        await run_fabric_agent_once(request, fabric=fabric)

        assert len(first_runtime.invoke_requests) == 1
        assert len(second_runtime.invoke_requests) == 1
        assert first_runtime.exit_calls == 1
        assert second_runtime.exit_calls == 1

    async def test_cleans_up_once_after_invocation_error(self) -> None:
        fake_runtime = _FakeRuntime(invoke_error=fabric_runtime.FabricError("invoke failed"))
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
        )

        with pytest.raises(FabricRuntimeExecutionError, match="Fabric runtime invocation failed: invoke failed"):
            await run_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime))

        assert fake_runtime.exit_calls == 1

    async def test_cleans_up_once_after_cancellation(self) -> None:
        fake_runtime = _FakeRuntime(invoke_delay=60)
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
        )
        invocation = asyncio.create_task(run_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime)))
        while not fake_runtime.invoke_requests:
            await asyncio.sleep(0)

        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation

        assert fake_runtime.exit_calls == 1

    async def test_maps_cleanup_errors_after_successful_invocation(self) -> None:
        fake_runtime = _FakeRuntime(exit_error=fabric_runtime.FabricError("stop failed"))
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
        )

        with pytest.raises(FabricRuntimeExecutionError, match="Fabric runtime cleanup failed: stop failed"):
            await run_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime))

        assert fake_runtime.exit_calls == 1

    async def test_failed_run_result_is_returned_as_normalized_result(self) -> None:
        failed_result = _FakeRunResult(
            status="failed",
            output=_FabricMapping({"response": ""}),
            error=_FabricMapping({"stage": "invoke", "message": "adapter failed"}),
        )
        fake_fabric = _FakeFabric(runtime=_FakeRuntime(result=failed_result))
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
        )

        result = await run_fabric_agent_once(request, fabric=fake_fabric)

        assert result.status == "failed"
        assert result.error == {"stage": "invoke", "message": "adapter failed"}
        assert fake_fabric.runtime.exit_calls == 1

    async def test_normalizes_fabric_mapping_fields_to_plain_values(self) -> None:
        fake_result = _FakeRunResult(
            output={
                "response": "done",
                "messages": (_FabricMapping({"role": "assistant", "content": "done"}),),
            },
        )
        fake_fabric = _FakeFabric(runtime=_FakeRuntime(result=fake_result))
        request = FabricOneShotRequest(
            fabric_config=cast(FabricConfig, object()),
            base_dir=Path("/tmp/agent"),
        )

        result = await run_fabric_agent_once(request, fabric=fake_fabric)

        assert result.output == {
            "response": "done",
            "messages": [{"role": "assistant", "content": "done"}],
        }
        assert result.response == "done"
        assert result.artifacts == {"root": "/tmp/artifacts", "artifacts": []}
        assert result.telemetry == [{"provider": "relay", "kind": "trace"}]
        assert result.events == [{"kind": "runtime_start", "message": "started"}]
        assert result.metadata == {"adapter_runner": "python"}


@pytest.mark.asyncio
class TestStreamFabricAgentOnce:
    async def test_keeps_ephemeral_runtime_alive_for_stream_context(self) -> None:
        fabric_config = FabricConfig(
            metadata=MetadataConfig(name="test-agent"), harness=HarnessConfig(adapter_id="nvidia.fabric.codex")
        )
        fake_runtime = _FakeRuntime()
        fake_fabric = _FakeFabric(runtime=fake_runtime)
        request = FabricOneShotRequest(
            fabric_config=fabric_config,
            base_dir=Path("/tmp/agent"),
            input="hello",
            request_id="platform-request-1",
            caller_context={"source": "server"},
        )

        async with stream_fabric_agent_once(request, fabric=fake_fabric) as stream:
            assert fake_runtime.entered is True
            assert fake_runtime.exited is False
            assert [record async for record in stream.records()] == [{"type": "span", "message": "thinking"}]
            result = await stream.result()

        assert fake_runtime.exited is True
        assert fake_runtime.exit_calls == 1
        assert fake_fabric.start_calls == [
            {
                "base_dir": Path("/tmp/agent"),
                "fabric_config": fabric_config,
                "overrides": None,
                "streaming": True,
            }
        ]
        fabric_request = fake_runtime.invoke_stream_requests[0]
        assert fabric_request.input == "hello"
        assert fabric_request.request_id == "platform-request-1"
        assert fabric_request.context == {"source": "server"}
        assert result.response == "hello"

    async def test_runtime_setup_does_not_consume_result_timeout(self) -> None:
        class _SlowInvokeStream(_FakeInvokeStream):
            async def result(self) -> Any:
                await asyncio.sleep(0.06)
                return await super().result()

        fake_runtime = _FakeRuntime(stream=_SlowInvokeStream(), enter_delay=0.06)
        request = FabricOneShotRequest(
            fabric_config=FabricConfig(
                metadata=MetadataConfig(name="test-agent"), harness=HarnessConfig(adapter_id="nvidia.fabric.codex")
            ),
            base_dir=Path("/tmp/agent"),
            timeout_seconds=0.1,
        )

        async with stream_fabric_agent_once(
            request,
            fabric=_FakeFabric(runtime=fake_runtime, start_delay=0.06),
        ) as stream:
            result = await stream.result()

        assert result.status == "succeeded"
        assert fake_runtime.exit_calls == 1

    async def test_cleans_up_runtime_when_stream_start_fails(self) -> None:
        fake_runtime = _FakeRuntime(invoke_error=fabric_runtime.FabricError("stream unavailable"))
        request = FabricOneShotRequest(
            fabric_config=FabricConfig(
                metadata=MetadataConfig(name="test-agent"), harness=HarnessConfig(adapter_id="nvidia.fabric.codex")
            ),
            base_dir=Path("/tmp/agent"),
        )

        with pytest.raises(FabricRuntimeExecutionError, match="stream unavailable"):
            async with stream_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime)):
                pass

        assert fake_runtime.exited is True
        assert fake_runtime.exit_calls == 1

    async def test_wraps_runtime_context_entry_errors_as_start_errors(self) -> None:
        fake_runtime = _FakeRuntime(enter_error=fabric_runtime.FabricError("harness unavailable"))
        request = FabricOneShotRequest(
            fabric_config=FabricConfig(
                metadata=MetadataConfig(name="test-agent"), harness=HarnessConfig(adapter_id="nvidia.fabric.codex")
            ),
            base_dir=Path("/tmp/agent"),
        )

        with pytest.raises(FabricRuntimeStartError, match="Fabric runtime startup failed: harness unavailable"):
            async with stream_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime)):
                pass

        assert fake_runtime.entered is False
        assert fake_runtime.exit_calls == 0

    async def test_maps_runtime_context_exit_errors_as_cleanup_errors(self) -> None:
        fake_runtime = _FakeRuntime(exit_error=fabric_runtime.FabricError("stop failed"))
        request = FabricOneShotRequest(
            fabric_config=FabricConfig(
                metadata=MetadataConfig(name="test-agent"), harness=HarnessConfig(adapter_id="nvidia.fabric.codex")
            ),
            base_dir=Path("/tmp/agent"),
        )

        with pytest.raises(FabricRuntimeExecutionError, match="Fabric runtime cleanup failed: stop failed"):
            async with stream_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime)):
                pass

        assert fake_runtime.exit_calls == 1

    async def test_cleans_up_runtime_after_stream_result_error(self) -> None:
        stream = _FakeInvokeStream(result_error=fabric_runtime.FabricError("result failed"))
        fake_runtime = _FakeRuntime(stream=stream)
        request = FabricOneShotRequest(
            fabric_config=FabricConfig(
                metadata=MetadataConfig(name="test-agent"), harness=HarnessConfig(adapter_id="nvidia.fabric.codex")
            ),
            base_dir=Path("/tmp/agent"),
        )

        with pytest.raises(FabricRuntimeExecutionError, match="Fabric runtime streaming failed: result failed"):
            async with stream_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime)) as runtime_stream:
                await runtime_stream.result()

        assert fake_runtime.exit_calls == 1

    async def test_cleans_up_runtime_after_stream_result_timeout(self) -> None:
        class _SlowInvokeStream(_FakeInvokeStream):
            async def result(self) -> Any:
                await asyncio.sleep(1.0)
                return await super().result()

        fake_runtime = _FakeRuntime(stream=_SlowInvokeStream())
        request = FabricOneShotRequest(
            fabric_config=FabricConfig(
                metadata=MetadataConfig(name="test-agent"), harness=HarnessConfig(adapter_id="nvidia.fabric.codex")
            ),
            base_dir=Path("/tmp/agent"),
            timeout_seconds=0.01,
        )

        with pytest.raises(FabricRuntimeTimeoutError, match="timed out after 0.01s"):
            async with stream_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime)) as runtime_stream:
                await runtime_stream.result()

        assert fake_runtime.exit_calls == 1

    async def test_cleans_up_once_after_cancellation(self) -> None:
        fake_runtime = _FakeRuntime()
        request = FabricOneShotRequest(
            fabric_config=FabricConfig(
                metadata=MetadataConfig(name="test-agent"), harness=HarnessConfig(adapter_id="nvidia.fabric.codex")
            ),
            base_dir=Path("/tmp/agent"),
        )
        stream_started = asyncio.Event()

        async def consume() -> None:
            async with stream_fabric_agent_once(request, fabric=_FakeFabric(runtime=fake_runtime)):
                stream_started.set()
                await asyncio.Event().wait()

        invocation = asyncio.create_task(consume())
        await stream_started.wait()
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation

        assert fake_runtime.exit_calls == 1


@pytest.mark.asyncio
class TestInvokeFabricRuntime:
    async def test_invokes_active_runtime_without_changing_its_lifecycle(self) -> None:
        fake_runtime = _FakeRuntime()
        request = FabricInvocationRequest(
            input={"prompt": "hi"},
            request_id="platform-request-1",
            caller_context={"session_id": "session-1"},
        )

        result = await invoke_fabric_runtime(cast(Any, fake_runtime), request)

        assert fake_runtime.entered is False
        assert fake_runtime.exited is False
        fabric_request = fake_runtime.invoke_requests[0]
        assert fabric_request.input == {"prompt": "hi"}
        assert fabric_request.request_id == "platform-request-1"
        assert fabric_request.context == {"session_id": "session-1"}
        assert getattr(fabric_request, "relay_session_root", None) is None
        assert result.status == "succeeded"
        assert result.response == "hello"
        assert result.runtime_id == "runtime-1"

    async def test_forwards_the_relay_session_root(self) -> None:
        fake_runtime = _FakeRuntime()
        request = FabricInvocationRequest(input="hi", relay_session_root="018f47a4-0000-7d94-8e61-9f0f89b5d312")

        await invoke_fabric_runtime(cast(Any, fake_runtime), request)

        assert fake_runtime.invoke_requests[0].relay_session_root == "018f47a4-0000-7d94-8e61-9f0f89b5d312"

    async def test_wraps_timeout_without_stopping_runtime(self) -> None:
        fake_runtime = _FakeRuntime(invoke_delay=1.0)
        request = FabricInvocationRequest(timeout_seconds=0.01)

        with pytest.raises(FabricRuntimeTimeoutError, match="timed out after 0.01s"):
            await invoke_fabric_runtime(cast(Any, fake_runtime), request)

    async def test_wraps_active_runtime_timeout_without_configured_deadline(self) -> None:
        timeout_error = TimeoutError("adapter timed out")
        fake_runtime = _FakeRuntime(invoke_error=timeout_error)

        with pytest.raises(FabricRuntimeTimeoutError, match=r"Fabric runtime invocation timed out\.$") as exc_info:
            await invoke_fabric_runtime(cast(Any, fake_runtime), FabricInvocationRequest())

        assert exc_info.value.__cause__ is timeout_error

        assert fake_runtime.entered is False
        assert fake_runtime.exited is False


@pytest.mark.asyncio
class TestStreamFabricRuntime:
    async def test_starts_streaming_turn_and_preserves_request_context(self) -> None:
        fake_runtime = _FakeRuntime()
        request = FabricInvocationRequest(
            input={"prompt": "hi"},
            request_id="platform-request-1",
            caller_context={"session_id": "session-1"},
        )

        stream = stream_fabric_runtime(cast(Any, fake_runtime), request)

        fabric_request = fake_runtime.invoke_stream_requests[0]
        assert fabric_request.input == {"prompt": "hi"}
        assert fabric_request.request_id == "platform-request-1"
        assert fabric_request.context == {"session_id": "session-1"}

        records = [record async for record in stream.records()]
        assert records == [{"type": "span", "message": "thinking"}]

        result = await stream.result()
        assert result.status == "succeeded"
        assert result.response == "hello"
        assert result.runtime_id == "runtime-1"

    async def test_wraps_stream_start_errors(self) -> None:
        fake_runtime = _FakeRuntime(invoke_error=fabric_runtime.FabricError("streaming unavailable"))

        with pytest.raises(FabricRuntimeExecutionError, match="Fabric runtime streaming failed"):
            stream_fabric_runtime(cast(Any, fake_runtime), FabricInvocationRequest())

    async def test_wraps_stream_result_errors(self) -> None:
        fake_runtime = _FakeRuntime()
        stream = _FakeInvokeStream(result_error=fabric_runtime.FabricError("stream failed"))
        fake_runtime.stream = stream

        runtime_stream = fabric_runtime.FabricRuntimeStream(stream)

        with pytest.raises(FabricRuntimeExecutionError, match="Fabric runtime streaming failed"):
            await runtime_stream.result()

    async def test_wraps_stream_result_timeout(self) -> None:
        class _SlowInvokeStream(_FakeInvokeStream):
            async def result(self) -> Any:
                await asyncio.sleep(1.0)
                return await super().result()

        runtime_stream = fabric_runtime.FabricRuntimeStream(_SlowInvokeStream(), 0.01)

        with pytest.raises(FabricRuntimeTimeoutError, match="timed out after 0.01s"):
            await runtime_stream.result()

    async def test_closes_underlying_stream(self) -> None:
        fake_stream = _FakeInvokeStream()
        runtime_stream = fabric_runtime.FabricRuntimeStream(fake_stream)

        await runtime_stream.aclose()

        assert fake_stream.close_calls == 1


@pytest.mark.parametrize(
    ("settings", "atof", "message"),
    [
        ({}, {}, "relay_streaming: true"),
        ({"relay_streaming": True, "api_type": "anthropic-messages"}, {}, "openai-responses or openai-completions"),
        ({"relay_streaming": True}, {}, "shared collector"),
        ({"relay_streaming": True}, {"enabled": False}, "shared collector"),
    ],
)
def test_remote_streaming_rejects_incomplete_configuration(
    settings: dict[str, Any], atof: dict[str, Any], message: str
) -> None:
    config = FabricConfig.model_validate(
        {
            "metadata": {"name": "remote-agent"},
            "harness": {"adapter_id": "nvidia.fabric.remote-agent", "settings": settings},
            "relay": {"observability": {"atof": atof}},
        }
    )
    with pytest.raises(FabricRuntimeStartError, match=message):
        fabric_runtime.fabric_streaming_options(config)
