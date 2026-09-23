<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo MCP Server

Model Context Protocol (MCP) server for NeMo Helix.

## Overview

This MCP server exposes NeMo Helix functionality to AI assistants through the Model Context Protocol. It provides curated, workflow-oriented tools designed for agent-friendly interactions.

Located in `services/core/mcp/`, this is a core infrastructure service that follows the NeMo Helix v2 architecture patterns.

## Installation

```bash
# Install as part of workspace
uv sync --all-packages

# Or install just this service
uv pip install -e services/core/mcp
```

## Usage

### Start the Server

```bash
# Start with stdio transport
uv run nemo-mcp

# Or with custom configuration
uv run nemo-mcp --base-url https://your-nhx-instance.com

# With HTTP transport (for debugging)
uv run nemo-mcp --transport streamable-http --port 8080
```

### Manually Interact with Server

MCP uses stdio transport, so curl won't work directly. Use the MCP Inspector to interact with the server:

```bash
# Install globally
npm install -g @modelcontextprotocol/inspector

# Invoke with the server (set NHX_BASE_URL to your NeMo Helix instance)
NHX_BASE_URL=http://localhost:8080 npx @modelcontextprotocol/inspector uv run nemo-mcp
```

This brings up both the server and the inspector and launches it in your default browser. Then go to:
**Tools → List Tools → pick your tool → Run Tool!**

Try `list_workspaces` as a starting point.

### Configuration

The server uses NeMo SDK configuration:

- **`NHX_BASE_URL`**: URL of your NeMo Helix instance (e.g., `http://localhost:8080`)
- Other environment variables (`NHX_ACCESS_TOKEN`, etc.) as needed
- Config file (`~/.config/nhx/config.yaml`)
- Command-line flags

## Architecture

```
┌─────────────────────────────────────┐
│            AI Assistant             │
└─────────────────┬───────────────────┘
                  │ MCP Protocol
                  ▼
┌─────────────────────────────────────┐
│       NeMo MCP Server (Core)        │
│  ┌──────────────────────────────┐   │
│  │   Core Tools                 │   │
│  │   - list_workspaces          │   │
│  │   - (more coming soon)       │   │
│  └──────────────────────────────┘   │
│                                     │
│  Uses: nhx.common.mcp utilities     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   NeMo Helix Python SDK     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   NeMo Helix API   │
└─────────────────────────────────────┘
```

### V2 Architecture Alignment

This service follows NeMo Helix v2 patterns:

- **Location**: `services/core/mcp/` (core infrastructure service)
- **Namespace**: `nhx.core.mcp` (follows v2 naming)
- **Shared Utilities**: Uses `nhx.common.mcp` for client factory and error handling
- **Dependencies**: Workspace-scoped dependencies via `uv`

Future expansion will support mounting service-specific MCP servers from:

- `nhx.guardrails.mcp`
- `nemo_customizer` plugin MCP tools (when enabled)
- etc.

## Development

### Run Tests

Manual integration test verifies the MCP server correctly connects to NeMo Helix. NeMo Helix must be running locally:

```bash
# Start platform (core services)
NHX_CONFIG_FILE_PATH=e2e/configs/docker_in_memory.yaml \
  uv run nemo-helix run --service-group core

# Run integration tests
NHX_BASE_URL=http://localhost:8080 \
  uv run pytest services/core/mcp/tests/integration/smoke_test.py -v
```

### Code Quality

```bash
# Lint
uv run ruff check services/core/mcp

# Type check
uv run --frozen --extra cpu ty check services/core/mcp

# Format
uv run ruff format services/core/mcp
```

## Roadmap

**Phase 1: Core Foundation** (Current)

- ✅ `list_workspaces()` - Workspace discovery
- 🔄 `list_models()` - Model discovery
- 🔄 `get_job_status()` - Universal job monitoring
- 🔄 Additional core tools

**Future Phases**:

- Phase 2: Evaluator service MCP
- Phase 3: Data Designer service MCP
- Phase 4: Customization service MCP
- Phase 5: Guardrails & Inference service MCPs

## Contributing

When adding new tools:

1. **Use shared utilities**: Import from `nhx.common.mcp`
2. **Follow error handling patterns**: Use `format_error_response()`
3. **Document tools clearly**: Agents need clear descriptions
4. **Add tests**: Integration tests in `tests/integration/`
5. **Consider agent UX**: Tools should be discoverable and self-documenting

Example:

```python
from nhx.common.mcp import create_nemo_client, format_error_response

@server.tool(description="Clear, concise description for agents")
async def my_tool(param: str) -> dict[str, Any]:
    """
    Detailed docstring explaining what this tool does.

    Args:
        param: Description of parameter

    Returns:
        Dictionary with success and data fields
    """
    try:
        result = await nemo_client.some_operation(param)
        return {"success": True, "data": result}
    except Exception as e:
        return format_error_response(e)
```

## References

- [MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25)
- [FastMCP Documentation](https://gofastmcp.com/)
- [NeMo Helix Docs](https://docs.nvidia.com/nemo-helix)
