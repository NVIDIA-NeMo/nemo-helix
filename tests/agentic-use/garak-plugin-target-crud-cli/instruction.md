<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Garak Plugin Target CRUD Operations (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following Garak Plugin target operations using the `nhx` CLI.

1. **Create** a scan target named `harbor-scan-target` that points to a model endpoint. Use the following details:
   - Model: `mock-model-endpoint`
   - Type: `nim`
   - Description: `Initial scan target for harbor testing`

2. **List** all scan targets and confirm `harbor-scan-target` appears in the list.

3. **Get** the scan target `harbor-scan-target` by name and review its details.

4. **Update** the scan target `harbor-scan-target` - change its description to `Updated scan target for harbor testing`.

5. **Delete** the scan target `harbor-scan-target`.

6. **Create** a final scan target named `harbor-scan-target-final` with the following details:
   - Model: `final-model-endpoint`
   - Type: `openai`
   - Description: `Final scan target that persists for verification`

## Success Criteria

The task is complete when:
- All six operations above have been performed successfully
- The target `harbor-scan-target` has been deleted (should no longer exist)
- The target `harbor-scan-target-final` exists with the correct configuration
