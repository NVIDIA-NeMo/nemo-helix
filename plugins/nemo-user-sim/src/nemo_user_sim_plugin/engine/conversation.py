# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runs one simulated conversation to a finished UserSim row.

UserSim calls a conversation an episode in its hosting API, so its types keep that name here.
"""

import asyncio
import json
import logging
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from nemo_user_sim_plugin.engine.assistants import AgentTurnError
from nemo_user_sim_plugin.engine.export import with_tool_steps
from nemo_user_sim_plugin.engine.roles import RoleAnswerer
from nemo_user_sim_plugin.engine.tool_bridge import ToolBridge
from usersim.engine.external import ActivationRequest, HostRoleModel, ProbeEpisodeRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationDeps:
    roles: RoleAnswerer
    assistant_factory: Callable[[ToolBridge], Any]
    assistant_model: str
    embeddings: Any = None
    turn_timeout: float = 600
    max_steps: int = 400


def _failed(category: str, message: str, tool_calls: int) -> dict:
    error = {"category": category, "message": message}
    return {"status": "failed", "row": None, "error": error, "tool_calls": tool_calls}


def simulator_failure(row: dict) -> str | None:
    """Why UserSim abandoned a finished row, when the cause was its own models rather than the agent."""
    outcome = json.loads(row.get("simulation_outcome") or "{}")
    if outcome.get("status") != "failed" or outcome.get("failure_attribution") == "assistant_model":
        return None
    return f"{outcome.get('failure_class')}: {outcome.get('failure_detail')}"


@asynccontextmanager
async def _serve_tools(bridge: ToolBridge) -> AsyncIterator[str]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(bridge.mcp_app(), log_level="warning"))
    # Serving the socket we bound means no other conversation can take the port in between.
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if serving.done():
                serving.result()
            await asyncio.sleep(0.05)
        yield f"http://127.0.0.1:{port}/mcp/"
    finally:
        server.should_exit = True
        await serving
        sock.close()


async def run_conversation(row: dict, deps: ConversationDeps) -> dict:
    """Return ``{"status", "row", "error", "tool_calls"}`` for one materialized UserSim row."""
    names = deps.roles.models
    models: dict[str, Any] = {
        "user_model": HostRoleModel(model_name=names.user, max_tokens=2048),
        "judge_model": HostRoleModel(model_name=names.judge, max_tokens=2048),
        "summary_model": HostRoleModel(model_name=names.summary, max_tokens=2048),
        "assistant_model": HostRoleModel(model_name=deps.assistant_model, max_tokens=4096),
    }
    if deps.embeddings is not None:
        models["embedding_model"] = deps.embeddings
    runtime = ProbeEpisodeRuntime.from_resolved_row(row, models=models)

    async def answer_other(event: Any) -> Any:
        return await runtime.advance(await deps.roles.answer(event))

    bridge = ToolBridge(runtime, answer_other)
    session = deps.assistant_factory(bridge)
    started = False

    async def drive(mcp_url: str | None) -> dict:
        nonlocal started
        event = await runtime.advance()
        for _ in range(deps.max_steps):
            if not isinstance(event, ActivationRequest):
                result = await runtime.finalize()
                if reason := simulator_failure(result):
                    return _failed("simulator_error", reason, bridge.executed)
                if any(bridge.own_steps):
                    merged = with_tool_steps(result["conversation_messages"], bridge.own_steps)
                    result = {**result, "conversation_messages": merged}
                own = [
                    call["id"] for steps in bridge.own_steps for step in steps for call in step.get("tool_calls", [])
                ]
                return {
                    "status": "ok",
                    "row": result,
                    "error": None,
                    "tool_calls": bridge.executed,
                    "own_tool_calls": own,
                }
            if event.role != "assistant":
                event = await answer_other(event)
                continue
            deadline = asyncio.timeout(deps.turn_timeout)
            try:
                async with deadline:
                    if not started:
                        bridge.tools = list(event.tools or [])
                        system = next((m["content"] for m in event.messages if m.get("role") == "system"), None)
                        started = True
                        await session.start(bridge.tools, system, mcp_url)
                    event = await bridge.run_assistant_turn(event, session)
            except Exception as exc:
                await runtime.close()
                if bridge.simulator_failure is not None:
                    failure = bridge.simulator_failure
                    return _failed("simulator_error", f"{type(failure).__name__}: {failure}", bridge.executed)
                expired = deadline.expired()
                reason = (
                    f"assistant turn exceeded {deps.turn_timeout:.0f}s"
                    if expired
                    else str(exc)
                    if isinstance(exc, AgentTurnError)
                    else f"{type(exc).__name__}: {exc}"
                )
                return _failed("agent_error", reason, bridge.executed)
        await runtime.close()
        return _failed("step_limit", f"conversation did not finish within {deps.max_steps} steps", bridge.executed)

    try:
        if session.uses_mcp:
            async with _serve_tools(bridge) as mcp_url:
                return await drive(mcp_url)
        return await drive(None)
    except Exception as exc:  # a simulated-user, judge or summary call that failed past its retries
        await runtime.close()
        return _failed("simulator_error", f"{type(exc).__name__}: {exc}", bridge.executed)
    finally:
        if started:
            try:
                await session.stop()
            except Exception as exc:  # the conversation's outcome stands; a runtime that won't stop is only logged
                logger.warning("Agent runtime didn't stop cleanly: %s", exc)
