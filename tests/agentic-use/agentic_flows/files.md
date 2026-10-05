<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Files Service Agentic Flows

The Files service provides artifact storage through filesets. This replaces the old HuggingFace datastore functionality and supports file uploads, downloads, and directory operations.

**PIC**: Matt Grossman
**Priority**: High

---

## Flows

| # | Flow Name | Complexity | MCP Eval | CLI Eval | Description | Source |
|---|-----------|------------|----------|----------|-------------|--------|
| 4 | Fileset CRUD Operations | 1 | No | `files-crud-cli` | Create a fileset, upload a file, list files, download file content, delete file and fileset. This replaces the old datastore functionality. | POR; tests/e2e/test_files.py |
| 5 | Upload Dataset to Files Service | 2 | No | `files-upload-dataset-cli` | Create a fileset, upload training/validation/testing JSONL files (similar to the old HuggingFace datastore flow), and register as a dataset entity. | POR |

---

## Flow Details

### 4. Fileset CRUD Operations

**Complexity**: 1 (Easy)

**Operations**:
- Create fileset with name and description
- Upload file to fileset
- List files in fileset
- Download file content
- Delete file from fileset
- Delete fileset

**Prerequisites**:
- NeMo Helix running
- Workspace exists

**Success Criteria**:
- Fileset created successfully
- File uploaded and appears in list
- File content can be downloaded and matches original
- File can be deleted
- Fileset can be deleted

---

### 5. Upload Dataset to Files Service

**Complexity**: 2 (Simple)

**Operations**:
1. Create a fileset for dataset storage
2. Upload training data (JSONL format)
3. Upload validation data (JSONL format)
4. Upload testing data (JSONL format)
5. Register as dataset entity (optional)

**Prerequisites**:
- NeMo Helix running
- Workspace exists
- JSONL files prepared in correct format

**Data Formats Supported**:
- Prompt/completion format: `{"prompt": "...", "completion": "..."}`
- Chat format: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`

**Success Criteria**:
- Fileset created successfully
- All JSONL files uploaded
- Files can be retrieved and used by other services (Customizer, Evaluator)

---

## Documentation References

- Note: Files service is NEW in v2 - no v1 docs exist
- Old reference to adapt: docs/manage-entities/tutorials/manage-dataset-files.md (uses HF datastore)
- Tutorial reference: docs/get-started/tutorials/customize-eval-loop.md (Section: Upload Datasets)
