<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Entities Service Agentic Flows

The Entities service provides a generic storage layer for metadata about resources in NeMo Helix. Entities can represent models, datasets, and other platform objects.

**PIC**: Max Dubrinsky
**Priority**: High

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 6 | Basic Entity Operations | 1 | No | `entities-basic-cli` | Create, retrieve, query, update, and delete entities via the Entities API. | POR; tests/e2e/test_entities.py |

---

## Flow Details

### 6. Basic Entity Operations

**Complexity**: 1 (Easy)

**Operations**:
- Create entity with type and metadata
- Retrieve entity by ID
- Query entities with filters
- Update entity metadata
- Delete entity

**Prerequisites**:
- NeMo Helix running
- Workspace exists

**Entity Types**:
- Models
- Datasets
- Custom entity types

**Query Capabilities**:
- Filter by name patterns
- Filter by data fields
- Sort by created_at or name
- Pagination support

**Success Criteria**:
- Entity created with correct type and metadata
- Entity can be retrieved by ID
- Query returns matching entities
- Entity metadata can be updated
- Entity can be deleted

---

## Documentation References

- Note: Entities concept has changed in v2 - new docs required
- Old reference: docs/manage-entities/
