<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guardrails with Custom Configuration - CLI Eval

**Flow**: #21 - Guardrails with Custom Configuration
**Difficulty**: Hard
**Category**: cli-guardrails

## Description

This eval tests the agent's ability to create a custom guardrails configuration with both
input and output rails, perform CRUD operations on it, and exercise the configuration
through guardrails inference endpoints using a **real LLM** for content evaluation.

The guardrail rules use keyword-based checks for deterministic, verifiable testing:
- **Input rail**: blocks any user message that mentions a fruit (apple, banana, etc.)
- **Output rail**: blocks any bot response about baking bread

The agent must:

1. Explore existing guardrail configs to understand the schema
2. Create a custom config with input rails (self check input), output rails (self check output),
   custom keyword-based prompts, and the pre-configured `guardrails-llm` model
3. Retrieve and verify the config
4. Update the config description
5. Test that fruit mentions are blocked by input rails
6. Test that normal messages pass through and return a real response
7. Test that bread-baking responses are blocked by output rails

## Environment

- NeMo Helix API server running on localhost:8080
- Real inference provider (`nvidia-inference`) pre-configured via NVIDIA's inference API
- Model entity `default/guardrails-llm` registered and available
- CLI available at `/app/.venv/bin/nhx`
- CLI auth pre-configured
- MCP tools disabled (CLI only)

## Verification

The verifier tests both config structure AND functional correctness:
- Config has correct prompts mentioning "fruit" (input) and "bread" (output)
- Message mentioning fruit (e.g., "apples") must be blocked by input rail
- Normal message (e.g., "What is the capital of France?") must pass through
- Request eliciting bread-baking response must be blocked by output rail
