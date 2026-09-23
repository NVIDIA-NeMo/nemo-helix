<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Secret CRUD Operations (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following secret CRUD operations using the `nhx` CLI:

1. Create a secret named `harbor-test-secret` with the value `my-super-secret-value-123` and description `Test secret for harbor eval`
2. Verify the secret was created by retrieving its metadata
3. List all secrets in the workspace and confirm `harbor-test-secret` appears
4. Update the secret with a new value `updated-secret-value-456` and description `Updated test secret`
5. Verify the update was applied by retrieving the secret metadata again
6. Delete the secret `harbor-test-secret`
7. Create a new secret named `harbor-final-secret` with value `final-value-789` and description `Final secret for verification`

## Success Criteria

The task is complete when:
- A secret named `harbor-test-secret` was created, updated, and then deleted
- A secret named `harbor-final-secret` exists with the description `Final secret for verification`
- The secret `harbor-test-secret` no longer exists (was deleted)
