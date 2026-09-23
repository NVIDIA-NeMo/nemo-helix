<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Workspace Basic CLI Test

This Harbor test verifies that Claude Code can create and list workspaces using the `nemo` CLI.

## Purpose

This test demonstrates that the `nemo` CLI is properly installed and functional in the Harbor test environment, and that Claude Code can successfully use CLI commands to interact with the NeMo Helix.

## Difference from workspace-basic-mcp

- **workspace-basic-mcp**: Uses MCP tools (Model Context Protocol) for workspace operations
- **workspace-basic-cli**: Uses the `nemo` CLI directly via bash commands

Both tests verify the same outcome (workspace creation), but through different interfaces.

**Important:** This test disables MCP by removing `/app/.mcp.json` in the Dockerfile, ensuring Claude Code must use the CLI.

## Test Flow

1. Claude Code receives instructions to create a workspace using the `nemo` CLI
2. Claude executes: `nemo workspaces create harbor-test-workspace --description "..."`
3. Claude verifies by listing: `nemo workspaces list`
4. The verifier pytest script checks that `harbor-test-workspace` exists in the API

## CLI Commands Used

```bash
# Create a workspace
nemo workspaces create harbor-test-workspace --description "Test workspace"

# List workspaces
nemo workspaces list

# Get a specific workspace
nemo workspaces get harbor-test-workspace
```
