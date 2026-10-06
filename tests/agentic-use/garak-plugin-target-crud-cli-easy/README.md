<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Garak Plugin Target CRUD Operations - CLI Eval

This Harbor eval tests that a coding agent can perform CRUD (Create, Read, Update, Delete) operations on Garak Plugin targets using the NeMo Helix CLI.

## What It Tests

- Creating a scan target with specific model, type, and description
- Listing scan targets
- Getting a scan target by name
- Updating a scan target's description
- Deleting a scan target
- Creating a final scan target that persists for verification

## Verification

The verifier checks:
1. The original target (`harbor-scan-target`) was successfully deleted
2. The final target (`harbor-scan-target-final`) exists with correct model and type

## Running

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .
harbor run -p tests/agentic-use/garak-plugin-target-crud-cli-easy \
    --agent claude-code \
    --model anthropic/claude-sonnet-4-5
```
