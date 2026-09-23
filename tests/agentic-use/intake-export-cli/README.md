<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Intake Export to File - Harbor Eval (CLI)

Tests the agent's ability to use the NeMo Helix Intake service to collect LLM interaction data
and export it to a file. This is a CLI eval - MCP tools are disabled.

## What This Tests

- Discovering and using the Intake REST API (no dedicated CLI or SDK support)
- Creating an intake application
- Submitting LLM interaction entries with proper request/response structure
- Creating an export job to write entries to a local JSONL file
- Monitoring export job completion
- Verifying exported data

## Difficulty: Hard

The agent must discover the Intake API endpoints on its own (no CLI commands exist
for intake operations) and construct correct HTTP requests for apps, entries, and
export jobs.

## Known Limitations

- **Export destination**: The flow spec calls for exporting to the Files service
  (`nds://`), but the quickstart environment does not run the NeMo DataStore backend
  (`http://nemo-data-store:8000`) that the `nds://` scheme requires. This eval uses
  `file://` (local filesystem) instead. A future eval could test the `nds://` path
  once the DataStore is available in quickstart.
- **`status_details.entries_count`**: The export job's reported entry count is
  intermittently `0` due to a platform bug with nested value object persistence.
  The verifier does not assert on this field; actual entry counts are validated
  via the exported file content.
