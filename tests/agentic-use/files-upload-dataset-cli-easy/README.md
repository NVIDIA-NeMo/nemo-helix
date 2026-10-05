<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Upload Dataset to Files Service (CLI)

Tests the agent's ability to create JSONL dataset files and upload them to the NeMo Helix Files service using the CLI.

## What This Tests

- Creating a fileset via `nemo files filesets create`
- Creating local JSONL files in prompt/completion format
- Uploading multiple files to a fileset via `nemo files upload`
- Listing files to verify uploads via `nemo files list`

## Expected Agent Behavior

1. Create the `harbor-dataset-fileset` fileset
2. Create three local JSONL files (training, validation, testing) with prompt/completion data
3. Upload all three files to the fileset
4. List files to verify the uploads succeeded

## Verification

The verifier checks:
- Fileset exists with correct name and description
- All three JSONL files are present in the fileset
- Each file contains valid JSONL with `prompt` and `completion` fields
- Minimum row counts: training >= 3, validation >= 2, testing >= 2
- Agent trajectory shows the list command was used
