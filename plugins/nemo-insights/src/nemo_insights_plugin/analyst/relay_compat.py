# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary NOOA 0.0.10 tool-result bridge for Relay 0.9.

Remove this module once Helix requires a NOOA release whose Relay tool
middleware returns ToolExecutionResult. Never patch Relay or NOOA globally.
Follows NVIDIA/NeMo-Fabric#348's compatibility installer.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import nemo_relay
from nemo_relay import Json, ScopeHandle, ToolExecutionResult
from nooa import nemo_relay_middleware
from nooa.events import _NO_RETURN, ExecutionResult
from nooa.runtime.event_manager import EventManager
from nooa.runtime.middleware import (
    MIDDLEWARE_AGENT_CALL,
    MIDDLEWARE_EXECUTE_PYTHON,
    MIDDLEWARE_LLM_CALL,
    ExecutePythonContext,
    ExecutePythonNext,
)


async def _tool_middleware(ctx: ExecutePythonContext, nxt: ExecutePythonNext) -> ExecutePythonContext:
    """Preserve NOOA's result selection and accept Relay interceptor results."""
    args = {"code": ctx.code, **{key: ctx.params[key] for key in ("tool_call_id", "timeout") if key in ctx.params}}
    codec = nemo_relay.typed.BestEffortAnyCodec()
    captured_ctx: ExecutePythonContext | None = None

    async def execute(inner_args: Json) -> ToolExecutionResult[Json]:
        nonlocal captured_ctx
        if isinstance(inner_args, dict):
            if "code" in inner_args:
                ctx.code = inner_args["code"]
            for key in ("tool_call_id", "timeout"):
                if key in inner_args:
                    ctx.params[key] = inner_args[key]
        captured_ctx = await nxt(ctx)
        result = captured_ctx.result
        if result is None:
            return ToolExecutionResult(codec.to_json(None))
        value = result.returned_value
        if value is _NO_RETURN:
            if result.signal is not None:
                signal_result = getattr(result.signal, "result", None)
                value = signal_result.get("result") if isinstance(signal_result, dict) else None
            else:
                value = result.stdout or None
        return ToolExecutionResult(codec.to_json(value))

    relay_result = await nemo_relay.tools.execute(
        "execute_python", args, execute, tool_call_id=ctx.params.get("tool_call_id")
    )
    if captured_ctx is not None:
        return captured_ctx
    if isinstance(relay_result, ToolExecutionResult):
        ctx.result = ExecutionResult(returned_value=relay_result.result)
        return ctx
    raise RuntimeError("NeMo Relay guardrail blocked code execution before running.")


def install_nemo_relay_compat(event_manager: EventManager) -> Callable[[], None]:
    """Install NOOA's agent/LLM middleware and the Relay-compatible tool callback."""
    unsub_agent = event_manager.intercept(MIDDLEWARE_AGENT_CALL, nemo_relay_middleware.nemo_relay_agent_call_middleware)
    unsub_llm = event_manager.intercept(MIDDLEWARE_LLM_CALL, nemo_relay_middleware.nemo_relay_llm_middleware)
    unsub_exec = event_manager.intercept(MIDDLEWARE_EXECUTE_PYTHON, _tool_middleware)

    def uninstall() -> None:
        unsub_agent()
        unsub_llm()
        unsub_exec()

    return uninstall


@asynccontextmanager
async def relay_scope(event_manager: EventManager, scope_name: str) -> AsyncIterator[ScopeHandle]:
    """Keep NOOA's scope ownership and cleanup while using the local installer."""
    uninstall = install_nemo_relay_compat(event_manager)
    try:
        with nemo_relay.scope.scope(scope_name, nemo_relay.ScopeType.Agent) as handle:
            # Match NOOA 0.0.10's scope helper so nested agent calls retain their
            # parent and same-task scope-local guardrails; concurrent calls still
            # use NOOA's task isolation. Remove with the compatibility installer.
            token = nemo_relay_middleware._current_relay_scope.set((handle, asyncio.current_task()))
            try:
                yield handle
            finally:
                nemo_relay_middleware._current_relay_scope.reset(token)
    finally:
        uninstall()
