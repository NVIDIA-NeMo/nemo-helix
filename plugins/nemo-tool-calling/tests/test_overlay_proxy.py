# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from nemo_tool_calling_plugin.overlay_proxy import PROXY_MODULE, proxied_server_spec

UPSTREAM_SERVER = """
import json
from mcp.server.fastmcp import FastMCP

server = FastMCP("upstream")


@server.tool(description="Original description.")
def echo(payload: dict) -> str:
    return json.dumps({"echo": payload})


@server.tool(description="Untouched tool.")
def ping() -> str:
    return "pong"


server.run()
"""

PATCHED_SCHEMA = {
    "type": "object",
    "required": ["payload"],
    "additionalProperties": False,
    "properties": {
        "payload": {
            "type": "object",
            "required": ["name"],
            "additionalProperties": False,
            "properties": {"name": {"type": "string"}},
        }
    },
}


async def test_proxy_advertises_the_patch_and_forwards_calls(tmp_path: Path) -> None:
    (tmp_path / "upstream.py").write_text(UPSTREAM_SERVER)
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps([{"name": "echo", "description": "Patched.", "inputSchema": PATCHED_SCHEMA}]))
    spec = proxied_server_spec({"transport": "stdio", "url": sys.executable, "args": ["upstream.py"]}, overlay)
    params = StdioServerParameters(command=spec["url"], args=spec["args"], cwd=tmp_path)

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        forwarded = await session.call_tool("echo", {"payload": {"anything": 1}})

    assert tools["echo"].description == "Patched."
    assert tools["echo"].inputSchema == PATCHED_SCHEMA
    assert tools["ping"].description == "Untouched tool."
    assert not forwarded.isError
    [content] = forwarded.content
    assert isinstance(content, types.TextContent)
    assert json.loads(content.text) == {"echo": {"anything": 1}}


def test_proxied_spec_wraps_the_original_command(tmp_path: Path) -> None:
    spec = proxied_server_spec(
        {"transport": "stdio", "url": "python", "args": ["../mcps/server.py"], "env": {"A": "1"}},
        tmp_path / "o.json",
    )

    assert spec["url"] == sys.executable
    assert spec["args"][:2] == ["-m", PROXY_MODULE]
    assert spec["args"][-3:] == ["--", "python", "../mcps/server.py"]
    assert spec["env"] == {"A": "1"}
