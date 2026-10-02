<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# LLM-as-a-Judge Evaluation - CLI Harbor Test (Easy)

Tests configuring and running an LLM-as-a-Judge evaluation job using the NeMo Helix CLI.

## What This Tests

- Creating a secret for the NVIDIA inference API key
- Registering an inference provider in IGW for the judge model
- Uploading a dataset with model outputs to evaluate
- Creating an LLM-as-a-Judge metric with a custom rubric
- Launching a metric evaluation job
- Checking job status

## Running

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .
export ANTHROPIC_API_KEY='your-key'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'
harbor run -p tests/agentic-use/evaluator-llm-judge-cli-easy \
    --agent claude-code \
    --model aws/anthropic/bedrock-claude-sonnet-4-5-v1
```
