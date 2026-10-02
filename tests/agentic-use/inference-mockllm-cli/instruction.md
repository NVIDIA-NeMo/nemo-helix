<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MockLLM Provider in IGW (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Context

This environment has the Inference Gateway (IGW) running with **mock provider mode** enabled. In mock provider mode, any inference provider whose name starts with `igw-mock-` will return mock responses instead of proxying to a real backend. The mock response is controlled by setting the `X-Mock-Response` header (via the provider's `default_extra_headers`) to a JSON string containing the desired response body.

## Task

Using the `nhx` CLI, set up and exercise a MockLLM provider through the Inference Gateway:

1. **Create a mock inference provider** named `igw-mock-test-llm` in the `default` workspace that returns a deterministic chat completion response. The mock response content should be: `"This is a deterministic mock response from the test LLM."`

2. **Register a served model** on the provider so the model entity `default/test-llm` is discoverable through the gateway

3. **Make an inference call** through the Inference Gateway and confirm the response matches the expected deterministic output

## Success Criteria

The task is complete when:
- A mock provider named `igw-mock-test-llm` exists in the `default` workspace
- The provider is configured with the mock response in its headers
- An inference call through the gateway returns the deterministic mock response
