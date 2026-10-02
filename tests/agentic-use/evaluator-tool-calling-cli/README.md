<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tool Calling Evaluation (BFCL) - CLI Harbor Test

Tests that a coding agent can prepare a BFCL-format dataset, create a tool-calling metric, run a synchronous evaluation, and verify function_name_accuracy and function_name_and_args_accuracy scores using the NeMo Helix CLI.

## What This Tests

- Workspace creation
- BFCL-format dataset preparation and upload (messages, tools, tool_calls, response)
- Tool-calling metric creation with reference template
- Synchronous evaluation producing function_name_accuracy and function_name_and_args_accuracy
- Metric evaluation job creation

## Note on Model Inference

This eval does **not** require a live model endpoint. The tool-calling metric compares pre-recorded model responses (embedded in the dataset) against ground truth tool calls. The synchronous evaluation runs entirely offline — no inference calls are made. This differs from the academic benchmark eval, which creates a job pointing at a model but never executes it.

## Running

```bash
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .

export ANTHROPIC_API_KEY='your-key'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'

harbor run -p tests/agentic-use/evaluator-tool-calling-cli \
    --agent claude-code \
    --model anthropic/claude-sonnet-4-5
```
