<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Simple Custom Evaluation Job - CLI Harbor Test

Tests launching a custom evaluation job with a string-check metric using the NeMo Helix CLI.

## What This Tests

- Creating a workspace and fileset via CLI
- Uploading a dataset to the Files service
- Creating and running an evaluation metric job with the `string-check` metric
- Monitoring job status until completion
- Downloading aggregate and per-row evaluation results

## Running

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .
export ANTHROPIC_API_KEY='your-key'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'
harbor run -p tests/agentic-use/evaluator-simple-job-cli \
    --agent claude-code \
    --model aws/anthropic/bedrock-claude-sonnet-4-5-v1
```
