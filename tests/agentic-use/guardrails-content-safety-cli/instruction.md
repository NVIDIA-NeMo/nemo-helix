<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Basic Content Safety Check (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

You have a skill available for `nemo-guardrails` that contains CLI command references and examples. Use it before exploring `--help`.

## Context

A mock inference model (`default/mock-llm`) has been pre-configured in this environment. This model always responds with "Yes" to any prompt, which makes it suitable for use as a guardrails self-check model (it will block all content).

No guardrail configuration exists yet -- you need to create one.

## Task

1. Create a guardrail configuration that uses the `default/mock-llm` model with a self-check input rail to evaluate user messages for content safety
2. Send a harmful or toxic message through the guardrails system (e.g., containing insults or asking about dangerous activities)
3. Confirm that the guardrails blocked the request

## Success Criteria

The task is complete when:
- A guardrail configuration has been created with a self-check input rail
- At least one harmful message was sent through the guardrails system
- The response shows the content was blocked (e.g., `"status": "blocked"` in the output)

Once you see a blocked response, you are done. Do not continue exploring other commands.
