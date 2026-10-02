<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guardrails Content Safety CLI Eval

## Overview

This eval tests whether a coding agent can use the NeMo Helix CLI to configure guardrails
for content safety and verify that harmful content is blocked. The agent must create
a guardrail configuration with a self-check input rail and send messages through the
guardrails endpoint.

## Environment Setup

This eval uses a **mock inference backend** instead of a real LLM:

- The Inference Gateway runs in mock provider mode (`igw-mock-` prefix)
- A mock provider is created that always responds "Yes" to any prompt
- When used with a self-check input rail, this causes ALL content to be blocked
  with a blocked guardrail status.

The mock provider setup is handled by `environment/setup-mock.py`, which runs after
the NeMo Helix API is healthy but before the agent starts. The agent is responsible for
creating the guardrail configuration itself.

## What the Agent Should Do

1. Create a guardrail configuration using the mock model (`default/mock-llm`) with a
   self-check input rail
2. Send a harmful message through the guardrails endpoint
3. Confirm the message is blocked

## Verification

The verifier checks:
1. At least one guardrail configuration exists (the agent should have created it)
2. Content sent through guardrails returns a blocked status
3. The guardrails pipeline works end-to-end (even safe content is blocked by the mock)

## Building and Running

```bash
# Build the base Docker image (from repo root)
docker build -f Dockerfile.agentic-base -t nhx-agentic-base:latest .

# Run the eval
harbor run -p tests/agentic-use/guardrails-content-safety-cli-easy \
  --agent claude-code \
  --model <model> \
  --n-tasks 1
```

## Architecture

```
environment/
  Dockerfile       - Extends nhx-agentic-base:latest with mock provider env var + setup script
  setup-mock.py    - Creates mock inference provider after API starts
instruction.md     - Task description for the agent
task.toml          - Harbor configuration (timeouts, resources)
tests/
  test_outputs.py  - Pytest verifier (checks guardrails behavior)
  test.sh          - Runs the verifier
```
