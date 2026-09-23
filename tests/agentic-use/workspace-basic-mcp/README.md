<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Workspace Basic MCP - Harbor Test

This Harbor test validates that Claude Code can create a workspace using the NeMo Helix MCP server.

## Task Overview

The agent (Claude Code) must:
1. Create a workspace named `harbor-test-workspace` with a description
2. Verify it exists by listing workspaces

The workspace is left in place (not deleted) so you can verify it was created successfully.

## Files

- **instruction.md** - The task prompt given to Claude Code
- **task.toml** - Harbor configuration (timeouts, metadata)
- **environment/Dockerfile** - Container with NeMo Helix API server
- **tests/test.sh** - Test runner script
- **tests/test_outputs.py** - Pytest verification that workspace was created
- **solution/solve.sh** - Optional oracle solution

## Testing Locally (Without Harbor)

Before running with Harbor, you can test the components locally:

### 1. Start NeMo Helix API Server

```bash
# From repository root
uv run python -m uvicorn nhx.platform.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Test the MCP Tools

```python
# test_mcp_local.py
import asyncio
from nhx.core.mcp.server import create_server

async def test():
    server = create_server("http://localhost:8000")

    # Create
    _, result = await server.call_tool("create_workspace", {
        "id": "harbor-test-workspace",
        "description": "Test workspace"
    })
    print(f"Create: {result}")

    # List
    _, result = await server.call_tool("list_workspaces", {})
    print(f"List: {result}")

    # Verify creation
    _, result = await server.call_tool("list_workspaces", {})
    ids = [w["id"] for w in result["workspaces"]]
    assert "harbor-test-workspace" in ids
    print("Success! Workspace created.")

asyncio.run(test())
```

Run it:
```bash
uv run python test_mcp_local.py
```

### 3. Test the Verification Script

```bash
# Set environment variable
export NHX_BASE_URL=http://localhost:8000

# Install dependencies and run the test
uvx --with pytest==8.4.1 --with requests==2.32.3 \
  pytest tests/test_outputs.py -v
```

## Running with Harbor

Once local testing passes, run with Harbor:

```bash
# From repository root
harbor run -p tests/agentic-use/workspace-basic-mcp \
    --agent claude-code \
    --model anthropic/claude-sonnet-4-5
```

Harbor will:
1. Build the Docker container from `environment/Dockerfile`
2. Start the NeMo Helix API server inside the container
3. Run Claude Code with the instruction from `instruction.md`
4. Execute the verification tests in `tests/test.sh`
5. Generate a report with results

## Expected Behavior

### Success Case
- Claude Code creates the workspace using `create_workspace` MCP tool
- Claude Code lists workspaces to verify creation
- Test verifies workspace exists
- Reward: 1.0

### Failure Cases
- Workspace is not created → Test fails (reward: 0.0)
- MCP tools not available → Claude Code fails
- API server not running → Connection error

## Troubleshooting

### "Connection refused" errors
Ensure the NeMo Helix API server is running and accessible at `http://localhost:8000`

### "Workspace already exists" errors
Clean up from previous test runs:
```python
from nhx.common.mcp import create_nemo_client
client = create_nemo_client("http://localhost:8000")
try:
    client.workspaces.delete(name="harbor-test-workspace")
except:
    pass
```

### Docker build fails
Make sure you're running from the repository root, as the Dockerfile needs access to the entire NeMo Helix workspace.

## Next Steps

After this test works, you can create additional workspace tests:
- Error handling (delete non-existent workspace)
- Workspace with projects
- Workspace listing pagination
- Concurrent workspace operations
