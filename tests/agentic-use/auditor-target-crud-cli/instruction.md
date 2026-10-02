<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Auditor Target CRUD Operations (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following Auditor target operations using the `nhx` CLI.

1. **Create** an audit target named `harbor-audit-target` that points to a model endpoint. Use the following details:
   - Model: `mock-model-endpoint`
   - Type: `nim`
   - Description: `Initial audit target for harbor testing`

2. **List** all audit targets and confirm `harbor-audit-target` appears in the list.

3. **Get** the audit target `harbor-audit-target` by name and review its details.

4. **Update** the audit target `harbor-audit-target` - change its description to `Updated audit target for harbor testing`.

5. **Delete** the audit target `harbor-audit-target`.

6. **Create** a final audit target named `harbor-audit-target-final` with the following details:
   - Model: `final-model-endpoint`
   - Type: `openai`
   - Description: `Final audit target that persists for verification`

## Success Criteria

The task is complete when:
- All six operations above have been performed successfully
- The target `harbor-audit-target` has been deleted (should no longer exist)
- The target `harbor-audit-target-final` exists with the correct configuration
