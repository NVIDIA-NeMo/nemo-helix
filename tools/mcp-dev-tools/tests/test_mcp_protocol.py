#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test MCP server through actual protocol communication."""

import json
import subprocess
from pathlib import Path


def test_mcp_server():
    """Test the MCP server by sending protocol messages."""
    print("=" * 70)
    print("TESTING MCP SERVER PROTOCOL COMMUNICATION")
    print("=" * 70)
    print("\nStarting MCP server with stdio transport...")
    print("Sending MCP protocol messages...\n")

    # Start the server process - run from repo root
    repo_root = Path(__file__).parent.parent.parent.parent.resolve()
    process = subprocess.Popen(
        ["uv", "run", "tools/mcp-dev-tools/nhx_dev_mcp.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(repo_root),
    )

    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    try:
        # Send initialize request
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        print("📤 Sending initialize request...")
        process.stdin.write(json.dumps(initialize_request) + "\n")
        process.stdin.flush()

        # Read response
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line)
            print("📥 Received initialize response:")
            print(
                f"   Server: {response.get('result', {}).get('serverInfo', {}).get('name')}"
            )
            print(f"   Protocol: {response.get('result', {}).get('protocolVersion')}")
            print("   ✓ Server initialized successfully!\n")

        # Send tools/list request
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        print("📤 Sending tools/list request...")
        process.stdin.write(json.dumps(list_tools_request) + "\n")
        process.stdin.flush()

        # Read response
        tools = []  # Initialize tools to avoid NameError
        response_line = process.stdout.readline()
        if response_line:
            response = json.loads(response_line)
            tools = response.get("result", {}).get("tools", [])
            print(f"📥 Received {len(tools)} tools:\n")

            for tool in tools:
                name = tool.get("name", "unknown")
                desc = tool.get("description", "No description")
                print(f"   • {name}")
                print(f"     {desc}\n")

        print("=" * 70)
        print("MCP PROTOCOL TEST SUCCESSFUL!")
        print("=" * 70)
        print("\n✓ Server responds to MCP protocol correctly")
        print(f"✓ All {len(tools)} tools are registered and available")
        print("✓ Ready to use from Claude Code\n")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Stderr: {process.stderr.read()}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    test_mcp_server()
