# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from nemo_user_sim_plugin.engine.assistants import AgentTurnError, rate_limited
from nemo_user_sim_plugin.engine.tool_bridge import OVER, ToolBridge
from usersim.engine.external import EpisodeLifecycleComplete

TOOLS = [{"type": "function", "function": {"name": "kb_search", "parameters": {"type": "object"}}}]


def _activation(activation_id: str, role: str = "assistant", tools_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        activation_id=activation_id,
        role=role,
        tools_enabled=tools_enabled,
        messages=[{"role": "user", "content": "Where is my refund?"}],
    )


class FakeRuntime:
    """Returns a continuing assistant activation after a tool call, then a user activation."""

    def __init__(self, *following: Any):
        self.recorded: list[dict] = []
        self._following = list(following)

    async def advance(self, payload: dict) -> SimpleNamespace:
        self.recorded.append(payload)
        return self._following.pop(0)

    async def tool_result(self, call_id: str) -> str:
        return json.dumps({"call_id": call_id})


class OneToolCall:
    def __init__(self, bridge: ToolBridge, tool: str, reply: str = "Refund issued."):
        self.bridge, self.tool, self.reply = bridge, tool, reply

    async def invoke(self, user_text: str) -> str:
        self.result = json.loads(await self.bridge.handle_tool_call(self.tool, {"query": user_text}))
        return self.reply


async def _no_other(_event):
    raise AssertionError("no non-assistant activation expected")


async def test_tool_call_is_executed_by_usersim_and_reply_recorded():
    runtime = FakeRuntime(_activation("a2"), _activation("u1", role="user"))
    bridge = ToolBridge(runtime, _no_other)
    bridge.tools = TOOLS
    agent = OneToolCall(bridge, "kb_search")

    following = await bridge.run_assistant_turn(_activation("a1"), agent)

    call = runtime.recorded[0]["response"]["tool_calls"][0]
    assert agent.result == {"call_id": call["id"]}
    assert runtime.recorded[1] == {
        "activation_id": "a2",
        "response": {"role": "assistant", "content": "Refund issued."},
    }
    assert following.role == "user" and bridge.executed == 1


async def test_unoffered_tool_is_refused_without_reaching_usersim():
    runtime = FakeRuntime(_activation("u1", role="user"))
    bridge = ToolBridge(runtime, _no_other)
    bridge.tools = TOOLS
    agent = OneToolCall(bridge, "wire_money", reply="I can't move money.")

    await bridge.run_assistant_turn(_activation("a1"), agent)

    assert agent.result == {"error": "Tool 'wire_money' is not available in this conversation."}
    assert runtime.recorded == [
        {"activation_id": "a1", "response": {"role": "assistant", "content": "I can't move money."}}
    ]


async def test_tool_call_outside_an_assistant_step_is_refused():
    bridge = ToolBridge(FakeRuntime(), _no_other)
    bridge.tools = TOOLS

    result = json.loads(await bridge.handle_tool_call("kb_search", {}))

    assert result == {"error": "Tools are not available for this step. Answer the customer."}


async def test_a_simulator_failure_during_a_tool_call_is_recorded_as_the_simulators():
    async def failing_judge(_event):
        raise ConnectionError("judge unreachable")

    bridge = ToolBridge(FakeRuntime(_activation("a2", role="judge")), failing_judge)
    bridge.tools = TOOLS
    bridge._pending = _activation("a1")

    with pytest.raises(ConnectionError):
        await bridge.handle_tool_call("kb_search", {"query": "refund"})

    assert isinstance(bridge.simulator_failure, ConnectionError)


class KeepsCallingTools:
    """An agent that calls the tool, then keeps calling it until it's told the conversation is over."""

    def __init__(self, bridge: ToolBridge):
        self.bridge, self.results = bridge, []

    async def invoke(self, user_text: str) -> str:
        while True:
            self.results.append(json.loads(await self.bridge.handle_tool_call("kb_search", {"query": user_text})))
            await asyncio.sleep(0)


async def test_usersim_ending_the_conversation_mid_turn_ends_the_agents_turn():
    # UserSim's tool-use probes stop after a set number of tool rounds, which can fall mid agent turn.
    done = EpisodeLifecycleComplete(result={"trajectory_id": "t1"})
    bridge = ToolBridge(FakeRuntime(done), _no_other)
    bridge.tools = TOOLS
    agent = KeepsCallingTools(bridge)

    following = await asyncio.wait_for(bridge.run_assistant_turn(_activation("a1"), agent), timeout=1)

    assert following is done and bridge.executed == 1
    assert all(result == {"error": OVER} for result in agent.results[1:])


class FailsThenReplies:
    """An agent whose model call is refused ``failures`` times before it answers."""

    def __init__(self, failures: int, error: str = "Error code: 429 - Model capacity reached"):
        self.failures, self.error, self.tries = failures, error, 0

    async def invoke(self, user_text: str) -> str:
        self.tries += 1
        if self.tries <= self.failures:
            raise AgentTurnError(self.error, retryable=rate_limited(self.error))
        return "Done."


async def test_a_rate_limited_model_call_is_tried_again(monkeypatch):
    monkeypatch.setattr("nemo_user_sim_plugin.engine.tool_bridge.RETRY_DELAYS", (0, 0))
    runtime = FakeRuntime(_activation("u1", role="user"))
    agent = FailsThenReplies(failures=2)

    await ToolBridge(runtime, _no_other).run_assistant_turn(_activation("a1"), agent)

    assert agent.tries == 3
    assert runtime.recorded == [{"activation_id": "a1", "response": {"role": "assistant", "content": "Done."}}]


async def test_a_model_call_that_keeps_failing_fails_the_turn_in_plain_words(monkeypatch):
    monkeypatch.setattr("nemo_user_sim_plugin.engine.tool_bridge.RETRY_DELAYS", (0, 0))
    runtime = FakeRuntime()

    with pytest.raises(AgentTurnError, match="^The agent's model call failed: rate limited"):
        await ToolBridge(runtime, _no_other).run_assistant_turn(_activation("a1"), FailsThenReplies(failures=3))

    assert runtime.recorded == []  # never reaches UserSim, so nothing is scored


async def test_an_error_a_retry_cant_fix_is_not_retried():
    agent = FailsThenReplies(failures=1, error="Invalid tool schema")

    with pytest.raises(AgentTurnError, match="^The agent's turn failed: Invalid tool schema"):
        await ToolBridge(FakeRuntime(), _no_other).run_assistant_turn(_activation("a1"), agent)

    assert agent.tries == 1


async def test_a_turn_with_no_reply_and_no_tool_call_fails_as_no_reply():
    class SaysNothing:
        async def invoke(self, user_text: str) -> str:
            return ""

    runtime = FakeRuntime()

    with pytest.raises(AgentTurnError, match="^The agent didn't reply"):
        await ToolBridge(runtime, _no_other).run_assistant_turn(_activation("a1"), SaysNothing())

    assert runtime.recorded == []
