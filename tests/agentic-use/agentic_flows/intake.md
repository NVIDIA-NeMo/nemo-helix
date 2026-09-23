<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Intake Service Agentic Flows

The Intake service is the front door for LLM data collection, storing interactions, feedback, and providing APIs for data management. It's a new service in v2.

**PIC**: Ryan Angilly
**Priority**: Low

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 30 | Intake App CRUD Operations | 1 | No | `intake-export-cli` | Create, list, get, update, and delete an Intake application. Apps are the top-level container for collecting data entries. | POR |
| 31 | Intake Entry Submission | 2 | No | `intake-export-cli` | Submit entries to an Intake app. Entries contain the actual data being collected (e.g., user queries, feedback). List and filter entries. | POR |
| 32 | List and Filter Entries | 3 | No | `intake-export-cli` | After submitting entries, list entries with filtering (by app, date range, status). Test pagination and search capabilities. | POR |
| 33 | Export Task to Files Service | 4 | No | `intake-export-cli` | Create an export task to export Intake entries to a fileset in the Files service. Verify exported data format is usable for evaluation. | POR |

---

## Flow Details

### 30. Intake App CRUD Operations

**Complexity**: 1 (Easy)

**Operations**:
- Create app with name and configuration
- List all apps
- Get app by ID
- Update app settings
- Delete app

**App Configuration**:
- Name and description
- Data retention settings
- Schema for entries

**Prerequisites**:
- NeMo Helix running
- Workspace exists

**Success Criteria**:
- App created successfully
- App appears in list
- App settings can be updated
- App can be deleted

---

### 31. Intake Entry Submission

**Complexity**: 2 (Simple)

**Operations**:
1. Create or select app
2. Submit entry with data:
   - Request content
   - Response content
   - Context/metadata
   - Feedback (optional)
3. List entries
4. Filter entries by criteria

**Entry Data Structure**:
- Request: User input/prompt
- Response: Model output
- Context: Additional metadata
- Feedback: User ratings/comments
- External ID: Client-provided identifier

**Prerequisites**:
- App exists
- Entry data prepared

**Success Criteria**:
- Entries submitted successfully
- Entries appear in list
- External IDs work for lookups
- Basic filtering works

---

### 32. List and Filter Entries

**Complexity**: 3 (Moderate)

**Operations**:
1. Submit multiple entries to app
2. Filter by app ID
3. Filter by date range
4. Filter by status
5. Test pagination (page, page_size)
6. Test search capabilities

**Filter Options**:
- app_id
- date_range (start, end)
- status
- external_id
- thread aggregation

**Prerequisites**:
- App with multiple entries
- Entries spanning time range

**Success Criteria**:
- Filters return correct subset
- Pagination works correctly
- Search finds matching entries
- Large result sets handled

---

### 33. Export Task to Files Service

**Complexity**: 4 (Complex)

**Operations**:
1. Select entries to export (via filters)
2. Create export task
3. Specify output format
4. Monitor export progress
5. Verify fileset created in Files service
6. Validate exported data format

**Export Formats**:
- JSONL (for evaluation/training)
- CSV (for analysis)

**Use Cases**:
- Create training dataset from production data
- Export for evaluation pipeline
- Data backup/archival

**Prerequisites**:
- Entries exist to export
- Files service available

**Success Criteria**:
- Export task completes
- Fileset created in Files service
- Data format correct for downstream use
- Entry count matches export count

---

## Documentation References

- Note: Intake is a NEW service in v2 - documentation in progress
- API reference: docs/api/intake.md (points to platform API)
