<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Auditor Target CRUD Operations - CLI Eval

This Harbor eval tests that a coding agent can perform CRUD (Create, Read, Update, Delete) operations on Auditor targets using the NeMo Helix CLI.

## What It Tests

- Creating an audit target with specific model, type, and description
- Listing audit targets
- Getting an audit target by name
- Updating an audit target's description
- Deleting an audit target
- Creating a final audit target that persists for verification

## Verification

The verifier checks:
1. The original target (`harbor-audit-target`) was successfully deleted
2. The final target (`harbor-audit-target-final`) exists with correct model and type

## Running

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .
harbor run -p tests/agentic-use/auditor-target-crud-cli-easy \
    --agent claude-code \
    --model anthropic/claude-sonnet-4-5
```
