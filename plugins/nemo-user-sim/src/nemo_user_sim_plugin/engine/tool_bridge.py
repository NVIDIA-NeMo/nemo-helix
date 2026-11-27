# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routes an assistant's tool calls into UserSim, which executes them against its simulated tools.

UserSim pauses at each assistant activation. When the assistant calls a tool, the bridge records the
call on the pending activation, lets UserSim run it and hands the result back to the assistant. When
the assistant finishes its turn, the bridge records the final reply.

UserSim's tool-use probes stop after a set number of tool rounds, so a conversation can end during an
assistant turn. Its last reply can't be recorded then, so the bridge ends the turn instead of waiting.
"""

import asyncio
import contextlib
import json
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool
from nemo_user_sim_plugin.engine.assistants import AgentTurnError, rate_limited
from starlette.applications import Starlette
from starlette.routing import Mount
from usersim.engine.external import EpisodeLifecycleComplete

NOT_OFFERED = "Tool '{name}' is not available in this conversation."
NOT_NOW = "Tools are not available for this step. Answer the customer."
OVER = "The conversation has ended. Don't call any more tools."
NO_REPLY = "The agent didn't reply."
# Waits before retrying a turn whose model call was refused: three tries in all, as UserSim's evaluator makes.
# The conversation's per-turn deadline bounds them too.
RETRY_DELAYS = (5.0, 20.0)


def _plain(exc: AgentTurnError) -> str:
    """A failed agent turn as Studio shows it."""
    text = str(exc)
    if rate_limited(text):
        return "The agent's model call failed: rate limited by the model endpoint."
    first = text.strip().splitlines()[0] if text.strip() else "no reason given"
    return f"The agent's turn failed: {first[:300]}"


class ToolBridge:
    def __init__(self, runtime: Any, answer_other: Callable[[Any], Awaitable[Any]]):
        self.runtime = runtime
        self.answer_other = answer_other
        self.tools: list[dict] = []
        self.executed = 0
        # Per recorded assistant reply, the agent's calls to its own tools in that turn (see own_tool_steps).
        self.own_steps: list[list[dict]] = []
        # Set when answering the simulator's own roles failed mid tool call, so the conversation can say
        # the simulator failed, not the agent, whatever the agent's harness made of the error.
        self.simulator_failure: Exception | None = None
        self._pending: Any = None
        self._lock = asyncio.Lock()
        # Set when UserSim finishes the conversation during an assistant turn.
        self._finished: EpisodeLifecycleComplete | None = None
        self._ended = asyncio.Event()

    async def handle_tool_call(self, name: str, arguments: dict) -> str:
        async with self._lock:
            if name not in {tool["function"]["name"] for tool in self.tools}:
                return json.dumps({"error": NOT_OFFERED.format(name=name)})
            if self._finished is not None:
                return json.dumps({"error": OVER})
            event = self._pending
            if event is None or event.role != "assistant" or not event.tools_enabled:
                return json.dumps({"error": NOT_NOW})
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            response = {"role": "assistant", "content": "", "tool_calls": [call]}
            following = await self.runtime.advance({"activation_id": event.activation_id, "response": response})
            try:
                payload = await self.runtime.tool_result(call_id)
                self.executed += 1
            except Exception as exc:  # a call past the probe's per-turn cap is never run
                payload = json.dumps({"error": f"Tool call was not executed: {exc}"})
            try:
                while following is not None and not self._ends(following) and following.role != "assistant":
                    following = await self.answer_other(following)
            except Exception as exc:
                self.simulator_failure = exc
                raise
            self._pending = None if self._ends(following) else following
            return payload

    def _ends(self, event: Any) -> bool:
        """Whether ``event`` is UserSim finishing the conversation, which ends the agent's turn too."""
        if not isinstance(event, EpisodeLifecycleComplete):
            return False
        self._finished = event
        self._ended.set()
        return True

    async def run_assistant_turn(self, event: Any, session: Any) -> Any:
        """Run one assistant turn through ``session`` and return UserSim's next activation or its end."""
        self._pending = event
        user_turns = [m for m in event.messages if m.get("role") == "user"]
        executed = self.executed
        turn = asyncio.ensure_future(self._invoke(session, user_turns[-1]["content"] if user_turns else ""))
        ended = asyncio.ensure_future(self._ended.wait())
        try:
            await asyncio.wait({turn, ended}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:  # the turn was cut short, as by its timeout: don't leave the agent running
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await turn
            raise
        finally:
            ended.cancel()
        if not turn.done():  # UserSim finished during the turn, so nothing the agent says now is recorded
            turn.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await turn
            return self._finished
        reply = turn.result()
        if not reply and self.executed == executed:  # nothing reached the user: no text and no tool call
            raise AgentTurnError(NO_REPLY, retryable=False)
        steps = list(getattr(session, "last_tool_steps", None) or [])
        async with self._lock:
            event, self._pending = self._pending, None
            if self._finished is not None:
                return self._finished
            if event is None or event.role != "assistant":
                return event
            response = {"role": "assistant", "content": reply or ""}
            self.own_steps.append(steps)
            return await self.runtime.advance({"activation_id": event.activation_id, "response": response})

    async def _invoke(self, session: Any, text: str) -> str:
        """Run the turn, trying again after a refused model call if the failed try called no tool.

        A try that called a tool has moved the conversation on, so it isn't repeated. The agent's harness
        may keep the user's message from a failed try, so a retried turn can show it twice.
        """
        for delay in (*RETRY_DELAYS, None):
            executed = self.executed
            try:
                return await session.invoke(text)
            except AgentTurnError as exc:
                if delay is None or not exc.retryable or self.executed != executed or self._finished:
                    raise AgentTurnError(_plain(exc), retryable=False) from exc
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    def mcp_app(self) -> Starlette:
        """An MCP server exposing UserSim's tools, mounted at ``/mcp``."""
        server = Server("user-sim-tools")

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name=tool["function"]["name"],
                    description=tool["function"].get("description", ""),
                    inputSchema=tool["function"].get("parameters") or {"type": "object", "properties": {}},
                )
                for tool in self.tools
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            return [TextContent(type="text", text=await self.handle_tool_call(name, arguments or {}))]

        sessions = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True)

        @asynccontextmanager
        async def lifespan(_app: Starlette):
            async with sessions.run():
                yield

        return Starlette(routes=[Mount("/mcp", app=sessions.handle_request)], lifespan=lifespan)
