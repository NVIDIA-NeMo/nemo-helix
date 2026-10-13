<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix Development Tools MCP Server - Setup Complete ✓

## What Was Created

A new MCP (Model Context Protocol) server that provides **narrowly-scoped development tools** for a smoother development experience.

### Files Created
```
tools/mcp-dev-tools/
├── README.md                    # Full documentation
├── SETUP.md                     # This file
├── pyproject.toml               # Package definition
└── nhx_dev_mcp.py               # 18 development tools
```

### Configuration Updated
- `~/.claude.json` - Added `nhx-dev` server to local scope (personal config)
- `.claude/settings.local.json` - Enabled `nhx-dev` server and added permissions

## How It Works

### The Goal
Provide a streamlined development experience with dedicated, purpose-built tools.

### The Solution
This MCP server provides **Python functions** that execute specific commands via `subprocess` - each tool is focused and well-defined.

Each tool is:
- ✅ **Narrowly scoped** - Does exactly one thing
- ✅ **Defined in code** - Pre-defined and reviewable by your team
- ✅ **Safe** - Read-only operations by default, destructive ops excluded
- ✅ **Purpose-built** - Dedicated tools instead of generic shell commands

## Available Tools (18)

### Git Operations (7)
- `git_status()` - Show working tree status
- `git_log(limit=10)` - Show commit history
- `git_branch_list()` - List all branches
- `git_diff_summary()` - Show changes summary
- `git_diff_staged()` - Show staged changes
- `git_diff(staged=False, file_path=None)` - Show full diff
- `git_show(commit="HEAD", stat=False)` - Show commit details

### Testing (4)
- `run_unit_tests()` - Run all unit tests
- `run_integration_tests()` - Run all integration tests
- `run_service_tests(service_name)` - Test specific service
- `run_pytest(path=".", verbose=True, markers=None)` - Run pytest on directory or file

### Build Operations (4)
- `run_precommit()` - Run pre-commit hooks
- `run_ruff_check(path=None)` - Run linter
- `run_ruff_format(path=None, check_only=True)` - Run formatter
- `run_type_check()` - Run type checker

### Project Navigation (2)
- `list_directory(path=".")` - List directory contents
- `find_files(pattern, path=".", file_type=None)` - Find files matching pattern

### Make Targets (1)
- `make_target(target)` - Run allowlisted make targets:
  - test-unit, test-integration, test-all
  - refresh-openapi, update-sdk
  - build-policy, check-policy
  - update-licenses

## Testing the Server

### 1. Quick Verification (Already Done ✓)
```bash
uv run python tools/mcp-dev-tools/test_server.py
```

### 2. Interactive Testing with MCP Inspector
```bash
# Install inspector (if not already installed)
npm install -g @modelcontextprotocol/inspector

# Launch inspector with your server
npx @modelcontextprotocol/inspector uv run nhx-dev-mcp
```

This opens a browser interface where you can:
1. Go to **Tools** → **List Tools**
2. Select any tool (try `git_status` first)
3. Click **Run Tool**
4. See the results

### 3. Use in Claude (Restart Required)
The server is now configured in your local `~/.claude.json` and enabled in `.claude/settings.local.json`.

**Next time you start Claude Code, I will have access to all 18 dedicated development tools!**

To restart Claude:
- Exit this session
- Start a new session in this directory
- The `nhx-dev` MCP server will load automatically from your local config

## Next Steps

### 1. Restart Claude to Load the Server
The MCP server is configured but needs a fresh session to load.

### 2. Test in Real Usage
Once restarted, you can ask me to:
- "Check git status" → I'll use `git_status()` tool
- "Run unit tests" → I'll use `run_unit_tests()` tool

### 3. Expand as Needed
If you need more tools, edit `tools/mcp-dev-tools/nhx_dev_mcp.py`:

```python
@server.tool(description="Your new tool description")
async def your_new_tool(param: str) -> dict[str, Any]:
    """Docstring explaining what this does."""
    return run_command(["your", "command", param])
```

### 4. Share with Your Team
The MCP server code is already in the repo at `tools/mcp-dev-tools/`. Team members can enable it locally:

```bash
python tools/mcp-dev-tools/enable.py
```

This adds the server to their local `~/.claude.json` without modifying the shared `.mcp.json` file.

**Why local-only?** This keeps development tool preferences personal while sharing the tool code itself. Each developer can opt-in when ready.

## Architecture

```
┌─────────────────────────────────────┐
│         Claude Code CLI             │
│                                     │
│                                     │
└─────────────────┬───────────────────┘
                  │
                  │ MCP Protocol
                  │ (stdio transport)
                  │
                  ▼
┌─────────────────────────────────────┐
│   NeMo Helix Development Tools MCP Server          │
│                                     │
│  ✓ Purpose-built development tools  │
│  ✓ Using Python subprocess          │
│  ✓ Narrowly scoped operations       │
│                                     │
│  Tools:                             │
│  - git_status → subprocess.run()    │
│  - run_unit_tests → subprocess.run()│
│  - etc.                             │
└─────────────────┬───────────────────┘
                  │
                  │ subprocess.run()
                  │
                  ▼
┌─────────────────────────────────────┐
│   System Commands                   │
│   (git, make, pytest, docker, etc.) │
└─────────────────────────────────────┘
```

## Key Benefits

1. **Streamlined Workflow** - Development operations with dedicated, focused tools
2. **Audit Trail** - All operations defined in code, reviewable
3. **Team-wide Benefit** - Once in repo, everyone benefits
4. **Safe by Design** - Narrow scope prevents accidental misuse
5. **Easy to Extend** - Add new tools as needs arise

## Troubleshooting

### Server Not Loading
```bash
# Verify installation
cd tools/mcp-dev-tools && uv run nhx-dev-mcp --help

# Check local configuration
claude mcp list

# Check permissions
cat .claude/settings.local.json
```

### Tools Not Working
```bash
# Test individual tool with inspector
npx @modelcontextprotocol/inspector uv run nhx-dev-mcp
```

### Tools Not Being Used
- Confirm you restarted Claude after configuration
- Check that `.claude/settings.local.json` includes "nhx-dev" in enabledMcpjsonServers
- Verify the server is in your local config: `claude mcp list`
- Verify the MCP server loaded successfully at startup

## Questions?

- **README.md** - Full documentation
- **nhx_dev_mcp.py** - All tool implementations
- Ask in your next Claude session once the server loads!

---

**Status: ✓ Ready to use after Claude restart**
