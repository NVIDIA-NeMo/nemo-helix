<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Secrets Service Agentic Flows

The Secrets service provides secure storage for sensitive data like API keys and credentials. Secrets are foundational for provider API key management in NeMo Helix.

**PIC**: Taylor Mutch
**Priority**: High

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 3 | Secret CRUD Operations | 1 | No | `secrets-crud-cli` | Create a secret with encrypted data, retrieve metadata, update, and delete. Secrets are foundational for provider API key management. | POR; tests/e2e/test_secrets.py |

---

## Flow Details

### 3. Secret CRUD Operations

**Complexity**: 1 (Easy)

**Operations**:
- Create secret with name and encrypted value
- Retrieve secret metadata (value is never exposed via API)
- List secrets in workspace
- Update secret value
- Delete secret

**Prerequisites**:
- NeMo Helix running
- Workspace exists

**Success Criteria**:
- Secret created successfully
- Secret appears in list (metadata only, no value)
- Secret value never exposed in API responses
- Secret can be updated
- Secret can be deleted

**Access Control Notes**:
- Viewer: Can list/retrieve metadata, cannot access values
- Editor: Can create/update/delete, cannot access values
- Admin: Cannot access values
- Service Principal: Can access values (for job execution)

---

## Documentation References

- Reference: docs/set-up/manage-secrets.md (deployment-focused)
- Note: Full Secrets API documentation is new in v2
