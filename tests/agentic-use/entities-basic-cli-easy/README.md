<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Entity CRUD Operations - CLI Eval

Tests that a coding agent can perform basic entity CRUD operations using the NeMo Helix CLI.

## What This Tests

- Create an entity with type and metadata
- Retrieve an entity by name
- List entities of a given type
- Update entity metadata
- Delete an entity

## Build and Run

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .

export ANTHROPIC_API_KEY='your-key'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'

harbor run -p tests/agentic-use/entities-basic-cli-easy \
    --agent claude-code \
    --model anthropic/claude-sonnet-4-5
```
