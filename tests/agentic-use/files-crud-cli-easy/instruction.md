<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fileset CRUD Operations (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

## Task

Complete the following fileset CRUD operations using the `nemo` CLI:

1. Create a fileset named `harbor-test-fileset` with description `Test fileset for harbor eval`
2. Verify the fileset was created by retrieving it with `nemo files filesets get harbor-test-fileset`
3. Create a small test file locally, then upload it to the fileset
4. List the files in the fileset to confirm the upload succeeded
5. Download the file from the fileset and verify its contents match the original
6. Delete the file from the fileset
7. Delete the fileset `harbor-test-fileset`
8. Create a new fileset named `harbor-final-fileset` with description `Final fileset for verification`
9. Create a local file called `verify.txt` containing the text `harbor-verification-content` and upload it to `harbor-final-fileset`

Note: If your available tool cannot create local files or run shell redirection, use the NeMo SDK/API fallback:
- Upload content directly with the files upload-content API.
- Download content directly with the files download-content API.
- This fallback is equivalent for steps 3, 5, and 9.

## Available CLI Commands

The `nemo` CLI is available at `/app/.venv/bin/nemo`. You can use these commands:

- `nemo files filesets create <name> --description "<description>"` - Create a new fileset
- `nemo files filesets get <name>` - Retrieve a fileset by name
- `nemo files filesets list` - List all filesets
- `nemo files filesets update <name> --description "<description>"` - Update fileset metadata
- `nemo files filesets delete <name>` - Delete a fileset
- `nemo files upload <local_path> <fileset_name>` - Upload a file to a fileset
- `nemo files download <fileset_name> --remote-path <remote_path> -o <local_path>` - Download a file from a fileset
- `nemo files list <fileset_name>` - List files in a fileset
- `nemo files delete <fileset_name> --remote-path <file_path>` - Delete a file from a fileset
- SDK/API alternatives:
  - Upload: files upload-content operation with `content`, `remote_path`, and `fileset`.
  - Download: files download-content operation with `remote_path` and `fileset`.

Note: The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Success Criteria

The task is complete when:
- The fileset `harbor-test-fileset` was created, had a file uploaded, and was then deleted
- A fileset named `harbor-final-fileset` exists with description `Final fileset for verification`
- The file `verify.txt` has been uploaded to `harbor-final-fileset` with content `harbor-verification-content`
