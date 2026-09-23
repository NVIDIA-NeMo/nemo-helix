<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fileset CRUD Operations (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following fileset CRUD operations using the `nhx` CLI:

1. Create a fileset named `harbor-test-fileset` with description `Test fileset for harbor eval`
2. Verify the fileset was created by retrieving it
3. Create a small test file locally, then upload it to the fileset
4. List the files in the fileset to confirm the upload succeeded
5. Download the file from the fileset and verify its contents match the original
6. Delete the file from the fileset
7. Delete the fileset `harbor-test-fileset`
8. Create a new fileset named `harbor-final-fileset` with description `Final fileset for verification`
9. Create a local file called `verify.txt` containing the text `harbor-verification-content` and upload it to `harbor-final-fileset`

## Success Criteria

The task is complete when:
- The fileset `harbor-test-fileset` was created, had a file uploaded, and was then deleted
- A fileset named `harbor-final-fileset` exists with description `Final fileset for verification`
- The file `verify.txt` has been uploaded to `harbor-final-fileset` with content `harbor-verification-content`
