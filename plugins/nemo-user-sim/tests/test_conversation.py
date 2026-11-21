# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from nemo_agents_plugin.agent_config import AgentConfig
from nemo_agents_plugin.fabric.translator import translate_agent_config
from nemo_fabric import McpConfig, McpServerConfig, ToolsConfig
from nemo_user_sim_plugin.engine.assistants import (
    DEEPAGENTS_BUILTINS,
    AgentTurnError,
    FabricAssistant,
    ModelAssistant,
    agent_base_dir,
    own_tool_steps,
)
from nemo_user_sim_plugin.engine.conversation import ConversationDeps, run_conversation, simulator_failure
from nemo_user_sim_plugin.engine.export import with_tool_steps
from nemo_user_sim_plugin.engine.roles import RoleModels
from nemo_user_sim_plugin.gateway import TOKEN_ENV

FIXTURES = Path(__file__).parent / "fixtures"
ROW = json.loads((FIXTURES / "conversation_inputs.json").read_text())
REPLY = "It's $16,100 for a single filer."
TAVILY = {"tavily": "http://tavily.local/mcp"}

SCRIPT = {
    "judge": "<explanation>Fine.</explanation><rating>success</rating>",
    "summary": "yes",
    "user": "Thanks, that answers it.",
}


class ScriptedRoles:
    models = RoleModels(user="u", judge="j", summary="s")

    async def answer(self, event):
        response = {"role": "assistant", "content": SCRIPT[event.role]}
        return {"activation_id": event.activation_id, "response": response}


class Replies:
    uses_mcp = False

    def __init__(self, reply: str | Exception, *, start_error: Exception | None = None, stop_error=None):
        self.reply = reply
        self.start_error = start_error
        self.stop_error = stop_error

    async def start(self, tools, system, mcp_url):
        if self.start_error:
            raise self.start_error

    async def invoke(self, user_text):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    async def stop(self):
        if self.stop_error:
            raise self.stop_error


def _deps(reply: str | Exception, assistant=Replies, **failures) -> ConversationDeps:
    return ConversationDeps(
        roles=ScriptedRoles(),  # ty: ignore[invalid-argument-type]
        assistant_factory=lambda _bridge: assistant(reply, **failures),
        assistant_model="test/agent",
    )


async def test_conversation_finishes_with_a_usersim_row():
    result = await run_conversation(ROW, _deps("I've opened a dispute for the $1,499 charge."))

    assert result["status"] == "ok"
    messages = json.loads(result["row"]["conversation_messages"])
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "I've opened a dispute for the $1,499 charge."


@pytest.mark.parametrize(
    ("deps", "message"),
    [
        (_deps(TimeoutError("harness timed out")), "TimeoutError: harness timed out"),
        (_deps("unused", start_error=RuntimeError("adapter crashed")), "RuntimeError: adapter crashed"),
    ],
)
async def test_an_agent_that_fails_to_start_or_answer_fails_only_its_conversation(deps, message):
    result = await run_conversation(ROW, deps)

    assert result["status"] == "failed"
    assert result["error"] == {"category": "agent_error", "message": message}


async def test_an_agent_that_fails_to_stop_keeps_the_finished_conversation():
    result = await run_conversation(ROW, _deps("Done.", stop_error=RuntimeError("runtime already gone")))

    assert result["status"] == "ok"


def _assistant(runtime, *, keep_own_tools: bool = False) -> FabricAssistant:
    """A FabricAssistant on a fake runtime, skipping ``start``."""
    assistant = FabricAssistant.__new__(FabricAssistant)
    assistant._runtime, assistant._keep_own_tools, assistant._usersim_tools = runtime, keep_own_tools, set()
    return assistant


async def test_an_agent_reply_comes_from_this_turn_only():
    class Runtime:
        def __init__(self):
            self.messages = []
            self.turns = iter(
                [[{"role": "assistant", "content": "Your balance is $12."}], [{"role": "assistant", "content": ""}]]
            )

        async def invoke(self, input):
            self.messages += [{"role": "user", "content": input}, *next(self.turns)]
            return SimpleNamespace(status="succeeded", error=None)

    assistant = _assistant(Runtime())

    assert await assistant.invoke("Balance?") == "Your balance is $12."
    assert await assistant.invoke("And savings?") == ""


async def test_a_failed_agent_turn_raises_instead_of_reading_as_an_empty_reply():
    # Fabric reports a harness failure, such as a refused model call, in the result rather than raising.
    class Runtime:
        messages: list = []

        async def invoke(self, input):
            error = SimpleNamespace(message="litellm.RateLimitError: RPM limit=1000", retryable=False)
            return SimpleNamespace(status="failed", error=error)

    with pytest.raises(AgentTurnError, match="RateLimitError") as failure:
        await _assistant(Runtime()).invoke("Balance?")

    assert failure.value.retryable  # a rate limit is worth another try whatever the harness says


def _agent(servers: dict[str, str] | None = None, **extra):
    """A translated agent config with ``servers`` as its own MCP servers."""
    agent = {
        "config_format": "nemo-agents-spec-v1",
        "name": "tax",
        "default_harness": "deepagents",
        "harnesses": {"deepagents": {"kind": "deepagents", "settings": {"deepagents": {}}}},
        "models": {"default": {"provider": "nvidia", "model": "m", "api_key_env": "NVIDIA_API_KEY"}},
        **extra,
    }
    mcp = servers and McpConfig(
        servers={name: McpServerConfig(transport="streamable-http", url=url) for name, url in servers.items()}
    )
    return translate_agent_config(AgentConfig.model_validate(agent)).model_copy(update={"mcp": mcp})


def test_each_agent_conversation_gets_its_own_workspace():
    config = _agent(environment={"workspace": "./workspace", "artifacts": "./artifacts"})

    first, second = agent_base_dir(config), agent_base_dir(config)

    assert first.name != second.name
    assert (Path(first.name) / "workspace").is_dir() and (Path(first.name) / "artifacts").is_dir()
    first.cleanup()
    second.cleanup()


def test_a_conversation_usersim_abandoned_is_a_simulator_error_not_the_agents():
    finished = json.loads((FIXTURES / "finished_row.json").read_text())
    outcome = {
        "status": "failed",
        "failure_class": "user_followup_gate_exhausted",
        "failure_attribution": "user_model",
        "failure_detail": "user follow-up gate failed after 3 attempts at turn 1",
    }

    assert simulator_failure({**finished, "simulation_outcome": json.dumps(outcome)}) == (
        "user_followup_gate_exhausted: user follow-up gate failed after 3 attempts at turn 1"
    )
    assert simulator_failure(finished) is None


async def test_a_model_that_never_stops_calling_tools_is_an_agent_error():
    async def resolved(value):
        return value

    call = SimpleNamespace(id="1", type="function", function=SimpleNamespace(name="kb_search", arguments="{}"))
    message = SimpleNamespace(tool_calls=[call], content=None, model_dump=lambda **_: {"role": "assistant"})
    completions = SimpleNamespace(
        create=lambda **_: resolved(SimpleNamespace(choices=[SimpleNamespace(message=message)]))
    )
    bridge = SimpleNamespace(handle_tool_call=lambda name, arguments: resolved("{}"))
    assistant = ModelAssistant(
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),  # ty: ignore[invalid-argument-type]
        "m",
        bridge,
        max_tool_rounds=2,
    )

    with pytest.raises(RuntimeError, match="2 rounds of tool calls"):
        await assistant.invoke("Where is my refund?")


async def _started(monkeypatch, config, **kwargs):
    """The config FabricAssistant hands Fabric when a conversation starts."""
    seen = {}

    class FakeFabric:
        async def start_runtime(self, config, base_dir):
            seen["config"] = config
            return SimpleNamespace(stop=lambda: None)

    monkeypatch.setattr("nemo_user_sim_plugin.engine.assistants.Fabric", FakeFabric)
    assistant = FabricAssistant(config, **kwargs)
    await assistant.start([], None, "http://usersim.local/mcp")
    assert assistant._root is not None
    assistant._root.cleanup()
    return seen["config"]


@pytest.mark.parametrize(
    ("servers", "switch", "expected"),
    [
        (TAVILY, {}, {"usersim"}),  # off by default: UserSim's server replaces the agent's
        (TAVILY, {"keep_own_tools": True}, {"tavily", "usersim"}),
        (None, {"keep_own_tools": True}, {"usersim"}),
    ],
)
async def test_the_agent_keeps_its_own_servers_only_when_asked_and_never_the_builtins(
    monkeypatch, servers, switch, expected
):
    config = _agent(servers)

    started = await _started(monkeypatch, config, **switch)

    assert set(started.mcp.servers) == expected
    assert all(started.mcp.servers[name] == config.mcp.servers[name] for name in expected - {"usersim"})
    assert started.tools.blocked == list(DEEPAGENTS_BUILTINS)


async def test_usersims_server_wins_a_name_clash(monkeypatch):
    config = _agent({"usersim": "http://example.invalid/mcp"})

    started = await _started(monkeypatch, config, keep_own_tools=True)

    assert started.mcp.servers["usersim"].url == "http://usersim.local/mcp"


async def test_delete_is_blocked_and_the_agents_own_blocked_tools_stay_blocked(monkeypatch):
    config = _agent(TAVILY).model_copy(update={"tools": ToolsConfig(blocked=["tavily_extract"])})

    started = await _started(monkeypatch, config, keep_own_tools=True)

    assert {"delete", "tavily_extract", "execute", "write_file"} <= set(started.tools.blocked)


async def test_the_agent_never_sees_the_clients_gateway_token(monkeypatch):
    # The agent's process inherits this one's environment, where UserSim's own models read the token.
    monkeypatch.setenv(TOKEN_ENV, "the-clients-token")
    config = _agent(None)

    started = await _started(monkeypatch, config)

    assert started.environment.env == {**config.environment.env, TOKEN_ENV: "not-used"}


SEARCH = {"name": "tavily_search", "args": {"query": "standard deduction 2026"}, "id": "c1", "type": "tool_call"}
# What the harness reports for one turn. The text beside the call never reached the user, so it isn't kept.
HARNESS_TURN = [
    {"role": "user", "content": "What's the standard deduction?"},
    {"role": "ai", "content": "Let me look that up.", "tool_calls": [SEARCH]},
    {"role": "tool", "content": [{"type": "text", "text": "Single: $16,100 (irs.gov)"}]},
    {"role": "ai", "content": REPLY},
]
CHAT_STEPS = [
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "tavily_search", "arguments": json.dumps({"query": "standard deduction 2026"})},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "Single: $16,100 (irs.gov)"},
]


@pytest.mark.parametrize(("usersim_tools", "steps"), [(set(), CHAT_STEPS), ({"tavily_search"}, [])])
def test_the_agents_own_tool_steps_come_out_in_chat_format_without_usersims(usersim_tools, steps):
    assert own_tool_steps(HARNESS_TURN, usersim_tools=usersim_tools) == steps


@pytest.mark.parametrize("keep", [True, False])
async def test_an_agent_turn_keeps_its_own_tool_steps_only_with_agent_tools_on(keep):
    class Runtime:
        messages: list = []

        async def invoke(self, input):
            self.messages = HARNESS_TURN
            return SimpleNamespace(status="succeeded", error=None)

    assistant = _assistant(Runtime(), keep_own_tools=keep)

    assert await assistant.invoke("What's the standard deduction?") == REPLY
    assert assistant.last_tool_steps == (CHAT_STEPS if keep else [])


def test_tool_steps_go_before_the_reply_of_the_turn_that_made_them():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello. What's your question?"},
        {"role": "user", "content": "Standard deduction?"},
        {"role": "assistant", "content": REPLY},
    ]

    merged = json.loads(with_tool_steps(json.dumps(messages), [[], CHAT_STEPS]))

    assert merged == [*messages[:3], *CHAT_STEPS, messages[3]]


async def test_a_conversation_records_the_agents_own_tool_calls():
    class SearchesOnce(Replies):
        last_tool_steps: list = []

        async def invoke(self, user_text):
            self.last_tool_steps = [] if hasattr(self, "_searched") else CHAT_STEPS
            self._searched = True
            return await super().invoke(user_text)

    result = await run_conversation(ROW, _deps(REPLY, SearchesOnce))

    messages = json.loads(result["row"]["conversation_messages"])
    reply = next(index for index, message in enumerate(messages) if message.get("content") == REPLY)
    assert messages[reply - 2 : reply] == CHAT_STEPS
    assert result["own_tool_calls"] == ["c1"]
