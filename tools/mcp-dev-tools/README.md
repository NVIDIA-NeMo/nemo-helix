<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Helix Development Tools MCP Server

MCP server providing narrowly-scoped development operations for the NeMo Helix project. These tools execute specific commands via Python subprocess, providing a smoother development experience with dedicated, pre-defined operations.

## Overview

This MCP server provides tools for:
- **Git operations**: status, log, branch listing, diffs (summary and full)
- **Testing**: unit tests, integration tests, service-specific tests
- **Build operations**: pre-commit, linting, formatting, type checking
- **Project navigation**: directory listing and file finding
- **Make targets**: safe allowlisted make commands

## Installation

This is a standalone uv script with inline dependencies. No installation needed - it runs directly with `uv run`.

**This MCP server is opt-in** and not enabled by default. To enable it, you need to configure it locally.

### Enabling the MCP Server Locally (Easy Way)

Run the enable script:

```bash
python tools/mcp-dev-tools/enable.py
```

This will automatically:
- Add nhx-dev to your **local config** (`~/.claude.json`) without modifying the committed `.mcp.json`
- Configure permissions in **`.claude/settings.local.json`** (per-developer local override, uncommitted) - this tool modifies your local settings file for machine-specific or temporary permission changes; the team-wide **`.claude/settings.json`** (committed) should be edited manually for permanent permissions that all developers should have
- Add tool preference documentation to `AGENTS.local.md`

Then restart Claude Code to load the server.

### Manual Configuration (Alternative)

If you prefer to configure manually, use the `claude mcp add` command:

```bash
claude mcp add nhx-dev -- sh -c "cd tools/mcp-dev-tools && uv run nhx-dev-mcp"
```

This adds the server to **local scope** (`~/.claude.json`) without affecting the committed `.mcp.json` file.

Then add permissions to `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "mcp__nhx-dev__*"
    ]
  },
  "enabledMcpjsonServers": [
    "nhx-dev"
  ]
}
```

And restart Claude Code to load the server.

**Why local scope?** Using `claude mcp add` (which defaults to local scope) keeps your personal MCP configuration in `~/.claude.json`, separate from the team's shared `.mcp.json` file. This prevents accidental commits of local development tools.

## Usage

### Test Tool Output Format

All test-running tools (`run_unit_tests`, `run_integration_tests`, `run_service_tests`, `run_pytest`) return structured results with:

- `success`: Boolean indicating if tests passed (returncode == 0)
- `stdout`: Full test output
- `stderr`: Error output
- `returncode`: Process exit code
- `command`: Command that was executed
- `summary`: **Parsed test metrics** (when pytest output is detected):
  - `passed`, `failed`, `skipped`, `xfailed`, `warnings`, `subtests_passed`: Test counts
  - `duration`: Test run duration (e.g., "207.29s (0:03:27)")
  - `failed_tests`: List of failed test identifiers
  - `error_tests`: List of errored test identifiers

This structured summary allows Claude to immediately understand test results without parsing megabytes of verbose output.

### Available Tools

#### Git Operations
- `git_status()` - Get repository status
- `git_log(limit=10)` - Show commit history
- `git_branch_list()` - List all branches
- `git_diff_summary()` - Show changes summary
- `git_diff_staged()` - Show staged changes summary
- `git_diff(staged=False, file_path=None)` - Show full diff with actual changes
- `git_show(commit="HEAD", stat=False)` - Show commit details

#### Testing
- `run_unit_tests()` - Run all unit tests
- `run_integration_tests()` - Run all integration tests
- `run_service_tests(service_name)` - Run tests for specific service (e.g., "evaluator", "auth")
- `run_pytest(path=".", verbose=True, markers=None)` - Run pytest on specific directory or file with optional marker filtering

#### Build Operations
- `run_precommit()` - Run pre-commit hooks
- `run_ruff_check(path=None)` - Run linter
- `run_ruff_format(path=None, check_only=True)` - Run formatter
- `run_type_check()` - Run type checker

#### Project Navigation
- `list_directory(path=".")` - List directory contents relative to repo root
- `find_files(pattern, path=".", file_type=None)` - Find files matching pattern (supports wildcards: *, ?)

**Note:** Specialized navigation tools (`list_services`, `list_packages`, `find_api_files`, `find_test_files`) mentioned in initial objectives are not yet implemented. Use the general-purpose `list_directory` and `find_files` tools instead:
- List services: `list_directory("services")`
- List packages: `list_directory("packages")`
- Find API files: `find_files("*api*.py", ".")`
- Find test files: `find_files("test_*.py", ".")`

#### Make Targets
- `make_target(target)` - Run allowlisted make targets:
  - Testing: test-unit, test-integration, test-all, test-policy
  - SDK/API: refresh-openapi, update-sdk
  - Policy: build-policy, check-policy
  - Licenses: update-licenses, check-licenses

## Key Features

### Streamlined Development Workflow
These tools use Python subprocess directly rather than the Bash tool, providing a more focused and efficient development experience.

### Narrowly Scoped
Each tool executes a specific, pre-defined command. No arbitrary command execution.

### Safe by Design
- Make targets use an allowlist
- Read-only operations default to safe parameters
- Destructive operations are excluded

## Configuration

The server is registered in **local scope** using `claude mcp add`. This keeps the configuration in your personal `~/.claude.json` file instead of the shared `.mcp.json` file in the repository.

To manually register the server:

```bash
claude mcp add nhx-dev -- sh -c "cd tools/mcp-dev-tools && uv run nhx-dev-mcp"
```

This creates a local-scoped configuration that won't affect other developers.

## Development

### Testing the Server

Use the MCP Inspector to test tools:

```bash
# Install inspector
npm install -g @modelcontextprotocol/inspector

# Launch with the server
npx @modelcontextprotocol/inspector uv run nhx-dev-mcp
```

### Adding New Tools

1. Add tool function to `nhx_dev_mcp.py` with `@server.tool()` decorator
2. Use `run_command()` helper for subprocess execution
3. Return structured dict with `success`, `stdout`, `stderr`, etc.
4. Document the tool in this README

### Code Quality

```bash
# Lint
uv run ruff check tools/mcp-dev-tools

# Format
uv run ruff format tools/mcp-dev-tools
```

## Architecture

```
┌─────────────────────────────────────┐
│         Claude Code CLI             │
└─────────────────┬───────────────────┘
                  │ MCP Protocol
                  ▼
┌─────────────────────────────────────┐
│   NeMo Helix Development Tools MCP Server          │
│  ┌──────────────────────────────┐   │
│  │  Git Tools                   │   │
│  │  Test Tools                  │   │
│  │  Build Tools                 │   │
│  │  Navigation Tools            │   │
│  │  Make Target Tools           │   │
│  └──────────────────────────────┘   │
│                                     │
│  Uses: Python subprocess            │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Git / Make / uv                   │
└─────────────────────────────────────┘
```

## Why This Approach?

This MCP server provides a better development experience by:

1. **Dedicated Tools**: Uses Python subprocess with focused, task-specific tools instead of generic Bash
2. **Narrowly Scoped**: Each tool does one specific thing
3. **Defined Operations**: Tools are pre-defined in code and reviewable
4. **Team-wide Benefit**: Once added to repo, entire team benefits

This allows AI assistants to perform common development operations with clear, purpose-built tools.
