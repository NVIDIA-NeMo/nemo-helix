<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guardrails with Custom Configuration (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

The NeMo Helix API server is running locally. You need to set up a real inference provider so guardrails can use an LLM for self-check evaluations. The `ANTHROPIC_API_KEY` environment variable contains an API key that works with NVIDIA's inference API.

Several built-in guardrail configurations exist (e.g., `default`, `self-check`). You can inspect them to understand the expected config structure.

## Task

Perform the following operations:

1. **Set up the inference provider**: Create a secret named `nvidia-api-key` using the value of `$ANTHROPIC_API_KEY`, then create an inference provider named `nvidia-inference` pointing to `https://inference-api.nvidia.com/v1` with that secret. Finally, register a served model so the model entity `default/guardrails-llm` maps to the upstream model `aws/anthropic/bedrock-claude-sonnet-4-5-v1`.
2. Explore the available guardrail config commands and inspect the existing built-in configurations to understand the config data structure
3. Create a custom guardrail configuration named `harbor-custom-config` with description `Custom guardrail config for harbor eval` that includes:
   - The `default/guardrails-llm` model configured as the main model (engine: nim)
   - **Input rails** using the `self check input` flow (pre-processing)
   - **Output rails** using the `self check output` flow (post-processing), with streaming enabled (chunk_size: 200)
   - A `self_check_input` prompt that instructs the model to block any user message that mentions a **fruit** (apple, banana, orange, grape, strawberry, mango, pear, peach, cherry, watermelon, lemon, lime, etc.). The prompt must ask "Should the user message be blocked (Yes or No)?" and expect a Yes/No answer.
   - A `self_check_output` prompt that instructs the model to block any bot response that contains information or instructions about **baking bread**. The prompt must ask "Should the message be blocked (Yes or No)?" and expect a Yes/No answer.
   - General instructions describing the bot's purpose
4. Retrieve the config by name and verify it was created correctly
5. Update the config's description to `Updated custom guardrail config`
6. Retrieve the config again and confirm the description was updated
7. Test the **input rail** by sending a message that mentions a fruit (e.g., "Tell me about the health benefits of apples") through the guardrails check endpoint — verify that the input rails **block** the request
8. Test that normal messages pass through by sending a message that does NOT mention any fruit or bread (e.g., "What is the capital of France?") — verify that the message **passes through** with a success status

## Success Criteria

The task is complete when:
- A guardrail config named `harbor-custom-config` exists with description `Updated custom guardrail config`
- The config has both input rails (`self check input`) and output rails (`self check output`) configured
- The config uses the `default/guardrails-llm` model
- The config has prompts for both `self_check_input` and `self_check_output` tasks
- A message mentioning fruit was **blocked** by the input rail
- A normal message (no fruit, no bread) **passed through** with a success status
