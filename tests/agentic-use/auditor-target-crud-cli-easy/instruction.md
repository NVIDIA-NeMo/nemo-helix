<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Auditor Target CRUD Operations (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following Auditor target operations using the `nemo` CLI.

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

## Available CLI Commands

- `nemo auditor targets create <name> -d '<json>'` - Create an audit target. The JSON body must include `model`, `type`, and optionally `description`.
- `nemo auditor targets list` - List all audit targets
- `nemo auditor targets get <name>` - Retrieve an audit target by name
- `nemo auditor targets update <name> -d '<json>'` - Update an audit target. Include the current `model` and `type` plus any changed fields.
- `nemo auditor targets delete <name>` - Delete an audit target

Example create body:

```json
{"model": "mock-model-endpoint", "type": "nim", "description": "Initial audit target for harbor testing"}
```

### Target Types

The `type` field in the JSON payload specifies the model endpoint type. Common values: `nim`, `openai`.

## Success Criteria

The task is complete when:
- All six operations above have been performed successfully
- The target `harbor-audit-target` has been deleted (should no longer exist)
- The target `harbor-audit-target-final` exists with the correct configuration
