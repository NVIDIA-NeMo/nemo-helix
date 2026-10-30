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

# With HTTP transport: a bearer token is required
NHX_MCP_AUTH_TOKEN="$(openssl rand -hex 32)" uv run nemo-mcp --transport streamable-http --port 8080

# Local debugging without a token (loopback hosts only)
uv run nemo-mcp --transport streamable-http --port 8080 --allow-unauthenticated
```

### Security

The HTTP transport acts on the platform with the server's own credentials, so it is locked down by default:

- **Authentication:** `--transport streamable-http` refuses to start unless `NHX_MCP_AUTH_TOKEN` is set (at least 16
  characters, no leading or trailing whitespace). Clients send it as `Authorization: Bearer <token>`. The token is read from the environment, not a
  flag, so it does not appear in the process list. Use a long random value.
- **Transport encryption:** the server serves plain HTTP, so the bearer token is sent unencrypted. For any non-loopback
  `--host`, terminate TLS in front of the server (reverse proxy) or use an encrypted tunnel (SSH, VPN).
- **No anonymous non-loopback binds:** `--allow-unauthenticated` is accepted only when `--host` is a loopback address.
- **Host and Origin checks:** the HTTP transport rejects requests whose `Host` or `Origin` header is not trusted
  (FastMCP `host_origin_protection="auto"`), which blocks DNS rebinding against a loopback-bound server. When a TLS
  proxy forwards to the server over loopback and keeps the public `Host`, requests are rejected with 421 unless that
  hostname is allowed: set `FASTMCP_HTTP_ALLOWED_HOSTS='["mcp.example.com"]'` on the server.
  `"auto"` only checks loopback-bound servers: with a non-loopback `--host` the server does not check `Host` or
  `Origin`, so the bearer token is the only access control there. Keep it secret and rotate it if it leaks.
- **Unauthenticated loopback is still reachable by local processes.** Use `--allow-unauthenticated` only for short
  debugging sessions, and it cannot be combined with `--enable-destructive-tools`.
- **Destructive tools are off by default:** `delete_workspace` is registered only with `--enable-destructive-tools`,
  on every transport. Deleting a workspace cascades to everything in it, and an LLM client can be steered into calling
  it by injected content.
- **Identity:** the token controls who can reach the server. Calls to the platform still use the server's own
  identity, not the caller's. Do not expose the server to users who should not have that level of access.

### Upgrade notes

- `delete_workspace` is no longer registered by default, on every transport (including stdio, e.g. via `.mcp.json`).
  Clients that relied on it must start the server with `--enable-destructive-tools`.
- `--transport streamable-http` now refuses to start without `NHX_MCP_AUTH_TOKEN`. Set the token and send it as
  `Authorization: Bearer <token>`, or use `--allow-unauthenticated` on a loopback host for local debugging.
- `--allow-unauthenticated` is rejected on stdio and when `NHX_MCP_AUTH_TOKEN` is set, instead of being ignored.

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

Unit tests cover the HTTP authentication, the CLI startup rules and the destructive-tool gate. They run in-process and need no platform:

```bash
uv run --frozen pytest services/core/mcp/tests/unit -q
```

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

- Phase 2: Evals service MCP
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
