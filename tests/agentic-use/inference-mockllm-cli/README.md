<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MockLLM Provider in IGW - CLI Eval

## Overview

This eval tests whether a coding agent can configure and use a MockLLM provider through the NeMo Helix Inference Gateway (IGW) using the CLI.

## Environment Setup

- The Inference Gateway runs in mock provider mode (`igw-mock-` prefix enabled)
- MCP tools are disabled; the agent must use the `nhx` CLI
- No mock provider is pre-created; the agent must create and configure it

## What the Agent Should Do

1. Create a mock inference provider named `igw-mock-test-llm` with a deterministic response configured via the `X-Mock-Response` header
2. Register a served model on the provider so it is discoverable via the gateway
3. Make an inference call through the gateway and verify the deterministic response

## Verification

The verifier checks:
- Mock provider `igw-mock-test-llm` exists in the `default` workspace
- Provider has the `X-Mock-Response` header configured in `default_extra_headers`
- Chat completion request returns the expected deterministic mock content

## Directory Structure

```text
environment/
  Dockerfile       - Extends nhx-agentic-base:latest with mock provider env var, disables MCP
instruction.md     - Task description for the agent
task.toml          - Harbor configuration (timeouts, resources)
tests/
  test.sh          - Standard shared test runner
  test_outputs.py  - Pytest verification
```
