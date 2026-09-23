<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guardrails with Custom Configuration (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Available CLI Commands

### Secrets

- `nemo secrets create <name> --value <value> --description <desc>` - Create a secret

### Inference Provider Commands

- `nemo inference providers create <name> --host-url <url> --api-key-secret-name <secret> --description <desc>` - Create an inference provider
- `nemo inference providers update-status <name> --served-models '<json>'` - Register served models on a provider

Served models JSON format:
```json
[{"model_entity_id": "default/<model-name>", "served_model_name": "<upstream-model-id>"}]
```

### Guardrail Config Commands

- `nemo guardrail configs create <name> --description <desc> --data '<json>'` - Create a guardrail configuration
- `nemo guardrail configs get <name>` - Retrieve a guardrail configuration by name
- `nemo guardrail configs list` - List guardrail configurations in current workspace
- `nemo guardrail configs list-global` - List built-in guardrail configurations (use this to see examples)
- `nemo guardrail configs update <name> --description <desc>` - Update a guardrail configuration

### Guardrail Inference Commands

- `nemo guardrail check --model <model> --messages '<json_array>' --guardrails '{"config_id": "<workspace>/<config_name>"}' --max-tokens 256` - Send a message through guardrails

Messages are passed as a JSON array: `'[{"role": "user", "content": "Your message here"}]'`

The config_id format is `default/<config-name>` (workspace/name).

### Guardrail Config JSON Structure

A full guardrail configuration with both input and output self-check rails:

```json
{
  "models": [
    {
      "type": "main",
      "engine": "nim",
      "model": "default/guardrails-llm"
    }
  ],
  "rails": {
    "input": {
      "flows": ["self check input"]
    },
    "output": {
      "flows": ["self check output"]
    }
  },
  "prompts": [
    {
      "task": "self_check_input",
      "content": "Your task is to check if the user message below should be blocked.\n\nUser message: {{ user_input }}\n\nQuestion: Should the user message be blocked (Yes or No)?\nAnswer:"
    },
    {
      "task": "self_check_output",
      "content": "Your task is to check if the bot response below should be blocked.\n\nBot response: {{ bot_response }}\n\nQuestion: Should the message be blocked (Yes or No)?\nAnswer:"
    }
  ],
  "instructions": [
    {
      "type": "general",
      "content": "Below is a conversation between a user and a bot."
    }
  ]
}
```

## Context

The NeMo Helix API server is running locally. You need to set up a real inference provider so guardrails can use an LLM for self-check evaluations. The `ANTHROPIC_API_KEY` environment variable contains an API key that works with NVIDIA's inference API.

Several built-in guardrail configurations exist. Use `nemo guardrail configs list-global` to inspect them and understand the expected config structure.

## Task

Perform the following operations:

1. **Set up the inference provider**: Create a secret named `nvidia-api-key` using the value of `$ANTHROPIC_API_KEY`, then create an inference provider named `nvidia-inference` pointing to `https://inference-api.nvidia.com/v1` with that secret. Finally, register a served model so the model entity `default/guardrails-llm` maps to the upstream model `aws/anthropic/bedrock-claude-sonnet-4-5-v1`.
2. Inspect the existing built-in configurations with `nemo guardrail configs list-global` to understand the config data structure
3. Create a custom guardrail configuration named `harbor-custom-config` with description `Custom guardrail config for harbor eval` that includes:
   - The `default/guardrails-llm` model configured as the main model (engine: nim)
   - **Input rails** using the `self check input` flow (pre-processing)
   - **Output rails** using the `self check output` flow (post-processing), with streaming enabled (chunk_size: 200)
   - A `self_check_input` prompt that instructs the model to block any user message that mentions a **fruit** (apple, banana, orange, grape, strawberry, mango, pear, peach, cherry, watermelon, lemon, lime, etc.). The prompt must ask "Should the user message be blocked (Yes or No)?" and expect a Yes/No answer.
   - A `self_check_output` prompt that instructs the model to block any bot response that contains information or instructions about **baking bread**. The prompt must ask "Should the message be blocked (Yes or No)?" and expect a Yes/No answer.
   - General instructions describing the bot's purpose
4. Retrieve the config with `nemo guardrail configs get harbor-custom-config` and verify it was created correctly
5. Update the config's description to `Updated custom guardrail config` with `nemo guardrail configs update harbor-custom-config --description "Updated custom guardrail config"`
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
