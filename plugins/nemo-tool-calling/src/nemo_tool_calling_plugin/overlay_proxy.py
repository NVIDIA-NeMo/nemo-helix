# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema-overlay MCP proxy: test a tool schema patch without owning the tool.

``python -m nemo_tool_calling_plugin.overlay_proxy --overlay patch.json -- <command> [args...]`` is a stdio MCP
server. It launches the original stdio server from the same cwd and environment, advertises the patched
``description`` / ``inputSchema`` for every tool named in the overlay, and forwards every call unchanged,
so the real server still validates.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

PROXY_MODULE = "nemo_tool_calling_plugin.overlay_proxy"


def apply_overlay(tool: types.Tool, overlay: Mapping[str, Mapping[str, Any]]) -> types.Tool:
    patch = overlay.get(tool.name)
    if patch is None:
        return tool
    update: dict[str, Any] = {}
    if isinstance(patch.get("description"), str):
        update["description"] = patch["description"]
    if isinstance(patch.get("inputSchema"), Mapping):
        update["inputSchema"] = copy.deepcopy(dict(patch["inputSchema"]))
    return tool.model_copy(update=update)


def load_overlay(path: Path) -> dict[str, dict[str, Any]]:
    """Read ``{tool name: definition}`` from a file holding one definition or a list of them."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    definitions = raw if isinstance(raw, list) else [raw]
    return {str(item["name"]): item for item in definitions if isinstance(item, dict) and "name" in item}


async def _list_all_tools(session: ClientSession) -> list[types.Tool]:
    tools: list[types.Tool] = []
    cursor: str | None = None
    while True:
        page = await session.list_tools(params=types.PaginatedRequestParams(cursor=cursor))
        tools.extend(page.tools)
        if not page.nextCursor or page.nextCursor == cursor:
            return tools
        cursor = page.nextCursor


async def serve(overlay: Mapping[str, Mapping[str, Any]], command: str, args: Sequence[str]) -> None:
    upstream_params = StdioServerParameters(command=command, args=list(args), env=dict(os.environ), cwd=os.getcwd())
    async with stdio_client(upstream_params, errlog=sys.stderr) as (upstream_read, upstream_write):
        async with ClientSession(upstream_read, upstream_write) as upstream:
            await upstream.initialize()
            server = Server("schema-overlay")

            @server.list_tools()
            async def list_tools() -> list[types.Tool]:
                return [apply_overlay(tool, overlay) for tool in await _list_all_tools(upstream)]

            @server.call_tool(validate_input=False)
            async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
                return await upstream.call_tool(name, arguments)

            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, server.create_initialization_options())


def proxied_server_spec(spec: Mapping[str, Any], overlay_path: Path) -> dict[str, Any]:
    """An agent-config stdio MCP server entry that runs *spec* behind this proxy."""
    wrapped = copy.deepcopy(dict(spec))
    wrapped["url"] = sys.executable
    wrapped["args"] = [
        "-m",
        PROXY_MODULE,
        "--overlay",
        str(overlay_path.resolve()),
        "--",
        str(spec["url"]),
        *[str(arg) for arg in spec.get("args") or []],
    ]
    return wrapped


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog=PROXY_MODULE, description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True, help="JSON tool definition(s) to advertise.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- <command> [args...] of the original server.")
    parsed = parser.parse_args(argv)
    command = list(parsed.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("the original server command is required after --")
    anyio.run(serve, load_overlay(parsed.overlay), command[0], command[1:])


if __name__ == "__main__":
    main()
