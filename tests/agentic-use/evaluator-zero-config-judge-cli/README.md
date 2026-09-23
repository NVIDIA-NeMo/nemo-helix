<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Zero-Config LLM-as-a-Judge Evaluation - CLI Eval

Tests that a coding agent can set up and run an LLM-as-a-Judge evaluation using only minimal configuration (model + scores) via the NeMo Helix CLI, letting the system auto-generate prompt templates, parsers, and structured output schemas.

## What This Tests

- Creating a secret for API authentication
- Uploading a dataset as a fileset
- Creating an LLM-as-a-Judge metric with zero-config (no prompt_template, no explicit parsers)
- Running a synchronous evaluation and examining scored results

## Running

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .

export ANTHROPIC_API_KEY='<your-key>'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'

harbor run -p tests/agentic-use/evaluator-zero-config-judge-cli \
    --agent claude-code \
    --model aws/anthropic/bedrock-claude-sonnet-4-5-v1 \
    --n-tasks 1
```
