# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import nemo_relay
import pytest
from nemo_insights_plugin.analyst import relay_compat
from nooa import nemo_relay_middleware
from nooa.events import ExecutionResult, ExecutionSignal
from nooa.runtime.event_manager import EventManager
from nooa.runtime.middleware import (
    MIDDLEWARE_AGENT_CALL,
    MIDDLEWARE_EXECUTE_PYTHON,
    MIDDLEWARE_LLM_CALL,
    AgentCallContext,
    ExecutePythonContext,
)


class _ReturnSignal(ExecutionSignal):
    result = {"result": ["signal-result"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        None,
        ExecutionResult(returned_value=None),
        ExecutionResult(returned_value=[{"insight": "test"}]),
        ExecutionResult(returned_value={"value": 1}),
        ExecutionResult(returned_value="result"),
        ExecutionResult(stdout="printed result"),
        ExecutionResult(signal=_ReturnSignal()),
    ],
)
async def test_real_relay_accepts_results_and_emits_tool_events(result):
    manager = EventManager()
    ctx = ExecutePythonContext(code="example", params={"timeout": 1})
    event_names = []
    original_middleware = nemo_relay_middleware.nemo_relay_tool_middleware
    original_execute = nemo_relay.tools.execute

    async def execute(context):
        assert context.code == "rewritten"
        assert context.params["timeout"] == 2
        assert context.params["tool_call_id"] == "call-1"
        context.result = result
        return context

    async def invoke_agent(context):
        context.result = await manager.run_middleware(MIDDLEWARE_EXECUTE_PYTHON, ctx, execute)
        return context

    original_scope = nemo_relay_middleware._current_relay_scope.get()
    with nemo_relay.scope.scope("parent", nemo_relay.ScopeType.Agent) as parent:
        async with relay_compat.relay_scope(manager, "compiler") as handle:
            assert nemo_relay_middleware.nemo_relay_tool_middleware is original_middleware
            assert nemo_relay.tools.execute is original_execute
            nemo_relay.scope_local.register_tool_request(
                handle,
                "rewrite",
                0,
                False,
                lambda name, args: {"code": "rewritten", "timeout": 2, "tool_call_id": "call-1"},
            )
            nemo_relay.scope_local.register_subscriber(handle, "capture", lambda event: event_names.append(event.name))
            invocation = await manager.run_middleware(
                MIDDLEWARE_AGENT_CALL, AgentCallContext(method_name="compile_insights"), invoke_agent
            )
            actual = invocation.result
            await nemo_relay.subscribers.flush_async()
            assert actual is ctx
            assert actual.result is result
            assert event_names.count("execute_python") >= 2
        assert nemo_relay.scope.get_handle().uuid == parent.uuid
        assert nemo_relay_middleware._current_relay_scope.get() is original_scope
    assert not any(manager._middleware.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"answer": 42}, ["intercepted"], None])
async def test_relay_interceptor_can_return_result_without_running_callback(payload):
    manager = EventManager()
    ctx = ExecutePythonContext(code="original", params={"tool_call_id": "call-1"})

    async def execute(context):
        pytest.fail("Short-circuited tool must not execute")

    async def intercept(context, next_call):
        assert context.tool_call_id == "call-1"
        return nemo_relay.ToolExecutionInterceptOutcome(payload)

    name = "helix-nooa-compat-short-circuit"
    nemo_relay.intercepts.register_tool_execution(name, 0, intercept)
    try:
        async with relay_compat.relay_scope(manager, "intercepted"):
            actual = await manager.run_middleware(MIDDLEWARE_EXECUTE_PYTHON, ctx, execute)
            assert actual is ctx
            assert isinstance(actual.result, ExecutionResult)
            assert actual.result.returned_value == payload
    finally:
        nemo_relay.intercepts.deregister_tool_execution(name)
    assert not any(manager._middleware.values())


@pytest.mark.asyncio
async def test_relay_guardrail_still_blocks_execution():
    manager = EventManager()

    async def execute(context):
        pytest.fail("Rejected tool must not execute")

    async with relay_compat.relay_scope(manager, "blocked") as handle:
        nemo_relay.scope_local.register_tool_conditional_execution(handle, "deny", 0, lambda name, args: "blocked")
        with pytest.raises(RuntimeError, match="blocked"):
            await manager.run_middleware(MIDDLEWARE_EXECUTE_PYTHON, ExecutePythonContext(code="blocked"), execute)
    assert not any(manager._middleware.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("tool failed"), asyncio.CancelledError()])
async def test_callback_errors_clean_up_scope_and_middleware(error):
    manager = EventManager()

    async def execute(context):
        raise error

    with nemo_relay.scope.scope("parent", nemo_relay.ScopeType.Agent) as parent:
        # Relay 0.9 converts exceptions raised by its callback into RuntimeError.
        with pytest.raises(RuntimeError, match=type(error).__name__):
            async with relay_compat.relay_scope(manager, "compiler"):
                await manager.run_middleware(MIDDLEWARE_EXECUTE_PYTHON, ExecutePythonContext(code="fails"), execute)
        assert nemo_relay.scope.get_handle().uuid == parent.uuid
    assert not any(manager._middleware.values())


def test_installer_preserves_other_middleware_and_uninstalls_only_its_own():
    manager = EventManager()

    async def existing(context, nxt):
        return await nxt(context)

    manager.intercept(MIDDLEWARE_EXECUTE_PYTHON, existing)
    uninstall = relay_compat.install_nemo_relay_compat(manager)
    assert manager._middleware[MIDDLEWARE_AGENT_CALL] == [nemo_relay_middleware.nemo_relay_agent_call_middleware]
    assert manager._middleware[MIDDLEWARE_LLM_CALL] == [nemo_relay_middleware.nemo_relay_llm_middleware]
    assert manager._middleware[MIDDLEWARE_EXECUTE_PYTHON] == [existing, relay_compat._tool_middleware]
    uninstall()
    assert manager._middleware[MIDDLEWARE_AGENT_CALL] == []
    assert manager._middleware[MIDDLEWARE_LLM_CALL] == []
    assert manager._middleware[MIDDLEWARE_EXECUTE_PYTHON] == [existing]
