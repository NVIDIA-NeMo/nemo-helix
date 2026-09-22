<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Secret CRUD Operations - CLI Eval

Harbor eval that tests secret CRUD operations using the NeMo Helix CLI.

## What This Tests

- Create a secret with name, value, and description
- Retrieve secret metadata (value is never exposed)
- List secrets in workspace
- Update a secret's value and description
- Delete a secret

## Running

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest . && \
export ANTHROPIC_API_KEY='<your-key>' && \
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com' && \
harbor run -p tests/agentic-use/secrets-crud-cli-easy \
    --agent claude-code \
    --model aws/anthropic/bedrock-claude-sonnet-4-5-v1
```
