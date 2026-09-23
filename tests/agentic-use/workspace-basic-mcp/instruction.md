<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Workspace Creation Test

You have access to a NeMo Helix MCP server with workspace management tools.

## Task

Complete the following workspace operations:

1. Create a workspace named `harbor-test-workspace` with a description of your choice
2. Verify the workspace exists by listing all workspaces

## Available MCP Tools

You should have access to these MCP tools:
- `create_workspace(name, description)` - Create a new workspace
- `list_workspaces()` - List all workspaces
- `delete_workspace(name)` - Delete a workspace

## Success Criteria

The task is complete when:
- The workspace `harbor-test-workspace` has been successfully created
- The workspace appears in the workspace list when you list all workspaces
