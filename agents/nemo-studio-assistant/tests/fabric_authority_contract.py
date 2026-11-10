# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated-process contract for the image's process-lifetime Fabric patches.

Exercise the installed adapter's actual invoke and MCP session serialization.
Only graph execution and MCP discovery are faked; no model or stdio process is
started. Isolation keeps compatibility patches out of other pytest threads.
"""

import asyncio
from typing import Any

from langchain_mcp_adapters import client as mcp_client
from mcp import ClientSession
from mcp.types import CallToolResult
from nemo_fabric_adapter_contract.models import (
    AgentConfig,
    AgentMcpConfig,
    AgentMcpServerConfig,
    AgentRunRequest,
    AgentRunStatus,
    ArtifactManifest,
    ControlLocation,
    EnvironmentHandle,
    EnvironmentOwnership,
    RuntimeContext,
)
from nemo_fabric_adapters.deepagents import adapter
from nemo_studio_assistant.authorization import CONTEXT_KEY
from nemo_studio_assistant.fabric_compat import apply_caller_authority_compatibility

calls = []
interceptors = []


class DiscoveryClient(mcp_client.MultiServerMCPClient):
    def __init__(self, connections, tool_interceptors):
        assert connections["nemo_studio"]["transport"] == "stdio"
        interceptors.extend(tool_interceptors)

    async def get_tools(self, *, server_name: str | None = None):
        return []


class RecordingSession(ClientSession):
    async def send_request(self, request, *args, **kwargs):
        params = request.root.params
        calls.append((params.name, params.arguments, params.meta))
        # Error results do not trigger a second output-schema discovery request.
        return CallToolResult(content=[], isError=True)


class OfflineRuntime(adapter.DeepAgentsRuntime):
    async def _invoke_agent(self, *args, **kwargs):
        await asyncio.sleep(0)
        session = RecordingSession.__new__(RecordingSession)

        async def handler(request):
            return await session.call_tool(
                request.name,
                arguments={"workspace": "default"},
                meta={CONTEXT_KEY: {"capability": "model-forgery"}},
            )

        for server in ("nemo_studio", "unrelated"):
            request: Any = type("ToolRequest", (), {"name": server, "server_name": server})()
            await interceptors[0](request, handler)
        return adapter.TurnOutcome(result_state={"messages": []}, events=[], turn_messages=[])


async def main():
    # This process exists solely for the compatibility contract. Production uses
    # the unmodified discovery client; fake discovery avoids spawning stdio.
    discovery_module: Any = mcp_client
    discovery_module.MultiServerMCPClient = DiscoveryClient
    apply_caller_authority_compatibility()
    config = AgentConfig(
        mcp=AgentMcpConfig(
            servers={
                "nemo_studio": AgentMcpServerConfig(transport="stdio", url="unused"),
            }
        )
    )
    await adapter._mcp_tools(config)
    context = RuntimeContext(
        runtime_id="test-runtime",
        invocation_id="test-invocation",
        request_id="test-request",
        environment=EnvironmentHandle(
            environment_id="test-environment",
            provider="local",
            control_location=ControlLocation.EXTERNAL_CONTROL,
            workspace=".",
            ownership=EnvironmentOwnership.CALLER_OWNED,
        ),
        artifacts=ArtifactManifest(),
    )

    async def invoke(identity):
        runtime = OfflineRuntime()
        runtime._started = True
        runtime._agent = object()
        runtime._start_payload = {}
        runtime._runtime_id = context.runtime_id
        result = await runtime.invoke(
            AgentRunRequest(input="test", context={CONTEXT_KEY: {"capability": identity}}), context
        )
        assert result.status == AgentRunStatus.SUCCEEDED
        assert identity not in str(result.output)

    await asyncio.gather(invoke("alice-capability"), invoke("bob-capability"))
    assert {getattr(meta, CONTEXT_KEY)["capability"] for name, args, meta in calls if name == "nemo_studio"} == {
        "alice-capability",
        "bob-capability",
    }
    assert all(meta is None for name, args, meta in calls if name == "unrelated")
    assert all(args == {"workspace": "default"} for name, args, meta in calls)
    await RecordingSession.__new__(RecordingSession).call_tool("outside-invocation")
    assert calls[-1][2] is None


if __name__ == "__main__":
    asyncio.run(main())
