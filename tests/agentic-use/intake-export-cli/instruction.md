<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Export Intake Entries to File (CLI)

You have access to the `nhx` CLI for NeMo Helix operations, and `curl` for HTTP requests. Note: MCP tools are not available in this environment - you must use the CLI or HTTP API.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The NeMo Helix API server is running at http://localhost:8080.

## Task

The NeMo Helix includes an Intake service for collecting LLM interaction data. Your goal is to create an intake application, populate it with entries, and then export those entries to a file using the Intake export API.

Complete the following operations:

1. Create a workspace named `intake-export-workspace`
2. Create an Intake application named `harbor-test-app` in the `intake-export-workspace` workspace
3. Submit at least 5 entries to the application. Each entry should represent an LLM interaction with request data (containing a user message) and response data (containing an assistant message). Include meaningful, distinct content in each entry.
4. Create an export job that exports **only entries from `harbor-test-app`** to a local JSONL file at `file:///tmp/intake-export.jsonl`. Use the export config's filter capabilities to select entries by app.
5. Check the status of the export job to confirm it completed
6. Verify the exported file exists at `/tmp/intake-export.jsonl` and contains the expected number of entries

## Notes

- The Intake service API is available under the NeMo Helix API server. You may need to explore the API to discover the available endpoints.
- Entries contain `data` (with request/response content) and `context` (with app, task, and optional thread information).
- The export job API accepts an `output_file_url` and a `config` with optional filters.
- The export writes entries in JSONL format, transforming them to include a top-level `messages` array for downstream compatibility.

## Success Criteria

The task is complete when:
- A workspace named `intake-export-workspace` exists
- An intake app named `harbor-test-app` exists in that workspace
- At least 5 entries have been submitted to the app
- An export job has been created and completed successfully
- The file `/tmp/intake-export.jsonl` exists and contains valid JSONL data with the exported entries
