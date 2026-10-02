<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Upload Dataset to Files Service (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following dataset upload operations using the `nhx` CLI:

1. Create a fileset named `harbor-dataset-fileset` with description `Dataset fileset for harbor eval`
2. Create a local training data file `training.jsonl` with at least 3 rows in prompt/completion JSONL format (each line: `{"prompt": "...", "completion": "..."}`)
3. Create a local validation data file `validation.jsonl` with at least 2 rows in the same prompt/completion JSONL format
4. Create a local testing data file `testing.jsonl` with at least 2 rows in the same prompt/completion JSONL format
5. Upload all three JSONL files to the `harbor-dataset-fileset` fileset
6. List the files in the fileset to verify all three uploads succeeded

## Success Criteria

The task is complete when:
- A fileset named `harbor-dataset-fileset` exists with description `Dataset fileset for harbor eval`
- Three JSONL files (`training.jsonl`, `validation.jsonl`, `testing.jsonl`) have been uploaded to the fileset
- Each file contains valid JSONL data with `prompt` and `completion` fields on every line
- `training.jsonl` has at least 3 rows, `validation.jsonl` has at least 2 rows, `testing.jsonl` has at least 2 rows
