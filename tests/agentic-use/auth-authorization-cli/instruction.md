<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Authorization Flow (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following authorization and role-based access control operations using the `nhx` CLI:

1. Create a workspace named `harbor-auth-test` with description `"Workspace for authorization testing"`
2. List the members of the workspace to confirm you (the workspace creator) are an Admin
3. Add a member `viewer@test.com` with the `Viewer` role to the workspace
4. Add a member `editor@test.com` with the `Editor` role to the workspace
5. Add a member `admin@test.com` with the `Admin` role to the workspace
6. List all members to verify all four members are present with correct roles
7. Update the role of `viewer@test.com` from `Viewer` to `Editor`
8. Delete member `editor@test.com` from the workspace
9. List members to confirm the final state

## Success Criteria

The task is complete when:
- The workspace `harbor-auth-test` has been created
- You confirmed you are Admin of the workspace as its creator
- Members `viewer@test.com`, `editor@test.com`, and `admin@test.com` were added with their respective roles
- `viewer@test.com` was promoted from Viewer to Editor
- `editor@test.com` was removed from the workspace
- Final member list shows the creator as Admin, `viewer@test.com` as Editor, and `admin@test.com` as Admin (with `editor@test.com` removed)
