# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fabric-native MCP boundary for NeMo Studio Assistant tools."""

from typing import Any

from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware, MiddlewareContext
from httpx import AsyncClient
from nemo_studio_assistant.authorization import CONTEXT_KEY, tool_authority
from nemo_studio_assistant.register import (
    ask_user_question,
    check_status,
    deploy_guardrail,
    job_progress,
    nemo_api,
    select_agent,
    select_dataset_file,
    select_eval_config,
    select_model,
    studio_link,
)

TOOLS = (
    deploy_guardrail,
    nemo_api,
    check_status,
    select_agent,
    select_model,
    select_dataset_file,
    select_eval_config,
    job_progress,
    studio_link,
    ask_user_question,
)


class CallerAuthorityMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next: Any) -> Any:
        # FastMCP reconstructs MiddlewareContext.message with version metadata
        # only. The incoming MCP envelope's _meta lives on request_context.
        request = context.fastmcp_context.request_context if context.fastmcp_context is not None else None
        meta = request.meta if request is not None else context.message.meta
        value = getattr(meta, CONTEXT_KEY, None) if meta is not None else None
        if not isinstance(value, dict):
            raise PermissionError("Missing trusted caller context")
        port = value.get("broker_port")
        capability = value.get("capability")
        if type(port) is not int or not 1 <= port <= 65535 or not isinstance(capability, str) or not capability:
            raise PermissionError("Invalid caller-context broker capability")
        # Only a fixed loopback endpoint is allowed. Never accept a tool-selected
        # URL or send a capability through an ambient HTTP proxy/redirect.
        async with AsyncClient(trust_env=False, follow_redirects=False, timeout=10) as client:
            response = await client.get(
                f"http://127.0.0.1:{port}/_internal/studio-tool-authority",
                headers={"authorization": f"Bearer {capability}"},
            )
            response.raise_for_status()
            resolved = response.json()
        with tool_authority(resolved):
            return await call_next(context)


def create_server() -> FastMCP:
    server = FastMCP("NeMo Studio Assistant")
    server.add_middleware(CallerAuthorityMiddleware())
    for tool in TOOLS:
        server.tool(tool)
    return server


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
