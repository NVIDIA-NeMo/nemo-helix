<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Auth Service Agentic Flows

The Auth service provides workspace management and authorization capabilities. Workspaces are the fundamental organizational unit in NeMo Helix - all resources belong to a workspace.

**PIC**: Razvan Dinu
**Priority**: High

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 1 | Workspace Management | 1 | `workspace-basic-mcp` | `workspace-basic-cli` | Create, list, get, update, and delete workspaces. All subsequent flows depend on workspaces existing. | POR; tests/e2e/test_workspaces.py |
| 2 | Authorization Flow | 5 | No | `auth-authorization-cli` | Create a workspace, grant roles (Viewer, Editor, Admin) to different principals via role bindings. Verify authorization enforcement: Viewer can read but not write, Editor can create resources, Admin can manage members. | POR; tests/e2e/test_workspaces.py (auth tests) |

---

## Flow Details

### 1. Workspace Management

**Complexity**: 1 (Easy)

**Operations**:
- Create workspace with name and description
- List all workspaces (with optional filtering)
- Get workspace by ID
- Update workspace description
- Delete workspace

**Prerequisites**:
- NeMo Helix running
- API access

**Success Criteria**:
- Workspace CRUD operations complete successfully
- Workspace appears in list after creation
- Workspace no longer appears after deletion

---

### 2. Authorization Flow

**Complexity**: 5 (Advanced)

**Operations**:
1. Create a workspace
2. Create role bindings for different principals:
   - Viewer role
   - Editor role
   - Admin role
3. Verify enforcement:
   - Viewer: Can list/read resources, cannot create/modify
   - Editor: Can create/modify resources, cannot manage members
   - Admin: Full access including member management

**Prerequisites**:
- NeMo Helix with auth enabled
- Multiple test principals (users/service accounts)

**Success Criteria**:
- Role bindings created successfully
- Permission checks pass for allowed operations
- Permission checks fail (403) for disallowed operations
- Admin cannot be removed if last admin

---

## Documentation References

- [Workspaces Concepts](https://docs.nvidia.com/nemo/microservices/latest/get-started/concepts/workspaces.html)
