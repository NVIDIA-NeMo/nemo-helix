<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Auditor Config CRUD Operations (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following auditor configuration CRUD operations using the `nhx` CLI:

1. List the existing global audit configs to understand the available structure (there should be a "default" config)
2. Create an audit config named `harbor-test-config` with description `Test config for harbor eval` that uses the `dan.AutoDANCached` probe
3. Verify the config was created by retrieving it with a get command
4. List all configs in the workspace and confirm `harbor-test-config` appears
5. Update `harbor-test-config` to change its description to `Updated test config`
6. Verify the update was applied by retrieving the config again
7. Delete the config `harbor-test-config`
8. Create a new config named `harbor-final-config` with description `Final config for verification` that uses the `dan.DanInTheWild` probe

## Success Criteria

The task is complete when:
- The config `harbor-test-config` was created, updated, and then deleted
- A config named `harbor-final-config` exists with description `Final config for verification`
- The `harbor-final-config` uses the `dan.DanInTheWild` probe
- The config `harbor-test-config` no longer exists (was deleted)
