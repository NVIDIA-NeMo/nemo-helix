<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Chat Completions via IGW - CLI Eval

## Overview

This eval tests whether a coding agent can use the NeMo Helix CLI to make chat completion requests through the Inference Gateway (IGW).

## Environment Setup

This eval uses a **mock inference backend** instead of a real LLM:

- The Inference Gateway runs in mock provider mode (`igw-mock-` prefix)
- A mock provider is created that returns a fixed chat completion response
- The mock model `default/chat-model` is available for inference

The mock provider setup is handled by `environment/setup-mock.py`, which runs after
the NeMo Helix API is healthy but before the agent starts.

## What the Agent Should Do

1. Discover the pre-configured mock model (`default/chat-model`)
2. Make a non-streaming chat completion request through the CLI
3. Attempt a streaming chat completion request (note: mock providers do not support streaming, so this request will fail or fall back to non-streaming; the verifier only validates non-streaming responses)

## Verification

The verifier checks:
- Mock provider `igw-mock-chat-model` exists (created by setup script)
- Chat completions return the expected mock response via both provider and model gateway routes
- Response has correct OpenAI-compatible structure

## TODO

- [ ] Revisit this eval to add proper NIM + streaming support. The current mock
  provider approach is useful for validating basic CLI chat completion workflows,
  but does not fully cover what we're trying to accomplish — testing against a
  real NIM backend with true streaming responses.

## Directory Structure

```text
environment/
  Dockerfile       - Extends nhx-agentic-base:latest with mock provider env var + setup script
  setup-mock.py    - Creates mock inference provider after API starts
instruction.md     - Task description for the agent
task.toml          - Harbor configuration (timeouts, resources)
tests/
  test.sh          - Standard shared test runner
  test_outputs.py  - Pytest verification
```
