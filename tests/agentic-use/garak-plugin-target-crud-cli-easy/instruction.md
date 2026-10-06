<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Garak Plugin Target CRUD Operations (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Task

Complete the following Garak Plugin target operations using the `nemo` CLI.

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

## Available CLI Commands

- `nemo garak-plugin targets create <name> -d '<json>'` - Create a scan target. The JSON body must include `model`, `type`, and optionally `description`.
- `nemo garak-plugin targets list` - List all scan targets
- `nemo garak-plugin targets get <name>` - Retrieve a scan target by name
- `nemo garak-plugin targets update <name> -d '<json>'` - Update a scan target. Include the current `model` and `type` plus any changed fields.
- `nemo garak-plugin targets delete <name>` - Delete a scan target

Example create body:

```json
{"model": "mock-model-endpoint", "type": "nim", "description": "Initial scan target for harbor testing"}
```

### Target Types

The `type` field in the JSON payload specifies the model endpoint type. Common values: `nim`, `openai`.

## Success Criteria

The task is complete when:
- All six operations above have been performed successfully
- The target `harbor-scan-target` has been deleted (should no longer exist)
- The target `harbor-scan-target-final` exists with the correct configuration
