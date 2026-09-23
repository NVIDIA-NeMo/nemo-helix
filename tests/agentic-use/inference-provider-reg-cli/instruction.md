<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model Provider Registration (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following model provider registration operations using the `nhx` CLI:

1. Create a secret named `harbor-provider-api-key` with the value `test-api-key-12345` and description `API key for test provider`
2. Register a model provider named `harbor-test-provider` with:
   - Host URL: `https://integrate.api.nvidia.com/v1`
   - API key secret name: `harbor-provider-api-key`
   - Description: `Test inference provider for harbor eval`
3. Verify the provider was registered by listing all providers
4. Retrieve the provider details
5. Delete the provider `harbor-test-provider`
6. Register a final provider named `harbor-final-provider` with:
   - Host URL: `https://integrate.api.nvidia.com/v1`
   - API key secret name: `harbor-provider-api-key`
   - Description: `Final provider for verification`

## Success Criteria

The task is complete when:
- A secret named `harbor-provider-api-key` exists
- The provider `harbor-test-provider` was registered and then deleted (should no longer exist)
- A provider named `harbor-final-provider` exists with host URL `https://integrate.api.nvidia.com/v1` and description `Final provider for verification`
