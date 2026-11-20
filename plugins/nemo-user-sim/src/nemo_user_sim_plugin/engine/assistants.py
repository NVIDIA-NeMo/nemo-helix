# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What answers the simulated user: a registered agent through Fabric, or a model through the gateway."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

import openai
from nemo_fabric import Fabric, FabricConfig, InstructionsConfig, McpConfig, McpServerConfig, ToolsConfig
from nemo_fabric.models import EnvironmentConfig, InstructionConfig
from nemo_user_sim_plugin.gateway import TOKEN_ENV, UNAUTHENTICATED

# Deep Agents built-ins. Blocked so the agent can act only through UserSim's simulated tools.
DEEPAGENTS_BUILTINS = (
    "write_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
    "task",
    "execute",
)


def agent_base_dir(config: FabricConfig) -> tempfile.TemporaryDirectory:
    """A fresh agent root with the local workspace and artifacts dirs the config names, as the Agents plugin prepares.

    Fabric resolves those paths against the root and runs the adapter there, so they must exist before it starts.
    One root per conversation keeps what an agent writes in one conversation out of the next.
    """
    root = tempfile.TemporaryDirectory(prefix="user-sim-agent-")
    environment = config.environment
    if environment is not None and environment.provider == "local":
        for path in (environment.workspace, environment.artifacts):
            if path is not None:
                (Path(root.name) / path).mkdir(parents=True, exist_ok=True)
    return root


def rate_limited(message: str) -> bool:
    """Whether a model error says the endpoint turned the call away for load, which a later try can pass."""
    text = message.lower()
    return any(sign in text for sign in ("429", "rate limit", "ratelimit", "capacity reached"))


def _text(content: Any) -> str:
    """A message's text, from a plain string or a list of content parts."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content if isinstance(content, str) else ""


def own_tool_steps(messages: list[dict], *, usersim_tools: set[str]) -> list[dict]:
    """The agent's calls to its own tools in one turn, with their results, as chat-format messages.

    The harness reports LangChain-style messages: an ``ai`` message carrying ``tool_calls`` as
    ``{name, args, id}``, then one ``tool`` message per call, in order and without the call's id. Calls to
    UserSim's tools are left out, since the bridge already records them in the conversation.
    """
    steps: list[dict] = []
    waiting: list[str | None] = []  # one entry per call still owed a result; None for UserSim's
    for message in messages:
        role = message.get("role") or message.get("type")
        if role in ("ai", "assistant") and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function") or {}
                name = call.get("name") or function.get("name")
                if name in usersim_tools:
                    waiting.append(None)
                    continue
                args = call.get("args", function.get("arguments", {}))
                arguments = args if isinstance(args, str) else json.dumps(args)
                calls.append(
                    {"id": call.get("id"), "type": "function", "function": {"name": name, "arguments": arguments}}
                )
                waiting.append(call.get("id"))
            if calls:
                # Text written beside a call never reached the simulated user, so it isn't kept.
                steps.append({"role": "assistant", "content": "", "tool_calls": calls})
        elif role == "tool" and waiting:
            call_id = waiting.pop(0)
            if call_id is not None:
                steps.append({"role": "tool", "tool_call_id": call_id, "content": _text(message.get("content"))})
    return steps


class AgentTurnError(RuntimeError):
    """The agent's turn failed before it replied, most often because its model call was refused.

    ``retryable`` says whether the same turn may succeed on another try, such as after a rate limit.
    """

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


class AssistantSession(Protocol):
    uses_mcp: bool

    async def start(self, tools: list[dict], system: str | None, mcp_url: str | None) -> None: ...
    async def invoke(self, user_text: str) -> str: ...
    async def stop(self) -> None: ...


class FabricAssistant:
    """One live Fabric runtime per conversation, so the agent keeps its history across turns.

    By default the agent's own MCP servers are replaced, so it acts only through UserSim's tools. With
    ``keep_own_tools`` it keeps them next to UserSim's, for runs whose tools are safe to call for real, such
    as search. The Deep Agents built-ins stay blocked either way.
    """

    uses_mcp = True

    def __init__(self, config: FabricConfig, *, keep_own_tools: bool = False):
        self._config = config
        self._keep_own_tools = keep_own_tools
        self._usersim_tools: set[str] = set()
        # The agent's calls to its own tools in the last turn, in chat format. Empty unless keep_own_tools.
        self.last_tool_steps: list[dict] = []
        self._runtime: Any = None
        self._root: tempfile.TemporaryDirectory | None = None

    async def start(self, tools: list[dict], system: str | None, mcp_url: str | None) -> None:
        self._usersim_tools = {tool["function"]["name"] for tool in tools}
        own = self._config.mcp.servers if self._keep_own_tools and self._config.mcp else {}
        own_blocked = self._config.tools.blocked if self._config.tools else []
        servers = {**own, "usersim": McpServerConfig(transport="streamable-http", url=mcp_url)}
        update: dict[str, Any] = {
            "mcp": McpConfig(servers=servers),
            # The built-ins, plus whatever the agent's own config already blocks.
            "tools": ToolsConfig(blocked=list(dict.fromkeys([*DEEPAGENTS_BUILTINS, *own_blocked]))),
        }
        # The agent's process inherits this one's environment, where UserSim's own models read the client's
        # gateway token. Shadow it, so the agent and its tools never see it.
        environment = self._config.environment or EnvironmentConfig()
        update["environment"] = environment.model_copy(update={"env": {**environment.env, TOKEN_ENV: UNAUTHENTICATED}})
        if system:
            update["instructions"] = InstructionsConfig(system=InstructionConfig(content=system, mode="replace"))
        # Fabric launches its adapter with $ADAPTER_PYTHON, falling back to $VIRTUAL_ENV. Pin it to this
        # interpreter so a stray VIRTUAL_ENV can't hand the adapter an environment without it installed.
        os.environ.setdefault("ADAPTER_PYTHON", sys.executable)
        self._root = agent_base_dir(self._config)
        self._runtime = await Fabric().start_runtime(
            self._config.model_copy(deep=True, update=update), base_dir=Path(self._root.name)
        )

    async def invoke(self, user_text: str) -> str:
        """The agent's last non-empty reply from this turn, or "" when it ended the turn without text.

        Raises ``AgentTurnError`` when the turn failed: Fabric reports a failed harness turn, such as a
        refused model call, in its result rather than raising.
        """
        before = len(self._runtime.messages or [])
        result = await self._runtime.invoke(input=user_text)
        if result.status != "succeeded":
            error = result.error
            raise AgentTurnError(
                error.message if error else f"the agent's turn ended with status {result.status}",
                retryable=bool(error and (error.retryable or rate_limited(error.message))),
            )
        turn = (self._runtime.messages or [])[before:]
        self.last_tool_steps = own_tool_steps(turn, usersim_tools=self._usersim_tools) if self._keep_own_tools else []
        for message in reversed(turn):
            reply = _text(message.get("content"))
            if (message.get("role") or message.get("type")) in ("assistant", "ai") and reply.strip():
                return reply
        return ""

    async def stop(self) -> None:
        if self._runtime is not None:
            await self._runtime.stop()
        if self._root is not None:
            self._root.cleanup()


class ModelAssistant:
    """A plain model as the assistant. Its tool calls go through the same bridge as an agent's."""

    uses_mcp = False

    def __init__(self, client: openai.AsyncOpenAI, model: str, bridge: Any, *, max_tool_rounds: int = 16):
        self._client = client
        self._model = model
        self._bridge = bridge
        self._max_tool_rounds = max_tool_rounds
        self._history: list[dict] = []
        self._tools: list[dict] = []

    async def start(self, tools: list[dict], system: str | None, mcp_url: str | None) -> None:
        self._tools = tools
        if system:
            self._history.append({"role": "system", "content": system})

    async def invoke(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        tools = self._tools or openai.NOT_GIVEN
        for _ in range(self._max_tool_rounds):
            completion = await self._client.chat.completions.create(
                model=self._model, messages=self._history, tools=tools
            )
            message = completion.choices[0].message
            self._history.append(message.model_dump(include={"role", "content", "tool_calls"}, exclude_none=True))
            if not message.tool_calls:
                return message.content or ""
            for call in message.tool_calls:
                arguments = json.loads(call.function.arguments or "{}")
                result = await self._bridge.handle_tool_call(call.function.name, arguments)
                self._history.append({"role": "tool", "tool_call_id": call.id, "content": result})
        raise RuntimeError(f"the model made {self._max_tool_rounds} rounds of tool calls without answering")

    async def stop(self) -> None:
        return None
