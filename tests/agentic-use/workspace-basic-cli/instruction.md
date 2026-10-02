<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Workspace Creation Test (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following workspace operations using the `nemo` CLI:

1. Create a workspace named `harbor-test-workspace` with a description of your choice
2. Verify the workspace exists by listing all workspaces

## Success Criteria

The task is complete when:
- The workspace `harbor-test-workspace` has been successfully created
- The workspace appears in the workspace list when you list all workspaces
