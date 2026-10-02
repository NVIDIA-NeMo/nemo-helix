<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Basic Content Safety Check (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Available CLI Commands

### Guardrail Config Commands

- `nemo guardrail configs create <name> --data '<json>'` - Create a guardrail configuration
- `nemo guardrail configs list` - List all guardrail configurations
- `nemo guardrail configs get <name>` - Retrieve a guardrail configuration

### Guardrail Check Commands

- `nemo guardrail check --model <model> --messages '<json_array>' --guardrails '{"config_id": "<workspace>/<config_name>"}'` - Send a message through guardrails for content safety check

### Guardrail Config JSON Structure

A self-check input rail configuration looks like:

```json
{
  "models": [
    {
      "type": "main",
      "engine": "nim",
      "model": "default/mock-llm"
    }
  ],
  "rails": {
    "input": {
      "flows": ["self check input"]
    }
  },
  "prompts": [
    {
      "task": "self_check_input",
      "content": "Your task is to check if the user message below complies with safety policy.\n\nUser message: {{ user_input }}\n\nQuestion: Should this message be blocked (Yes or No)?\nAnswer:"
    }
  ]
}
```

### Messages Format

Messages are passed as a JSON array: `'[{"role": "user", "content": "Your message here"}]'`

## Context

A mock inference model (`default/mock-llm`) has been pre-configured in this environment. This model always responds with "Yes" to any prompt, which makes it suitable for use as a guardrails self-check model (it will block all content).

No guardrail configuration exists yet -- you need to create one.

## Task

1. Create a guardrail configuration that uses the `default/mock-llm` model with a self-check input rail to evaluate user messages for content safety
2. Send a harmful or toxic message through the guardrails system (e.g., containing insults or asking about dangerous activities)
3. Confirm that the guardrails blocked the request

Use `nemo guardrail check` to send the message. It will show a blocked result when content is flagged.

## Success Criteria

The task is complete when:
- A guardrail configuration has been created with a self-check input rail
- At least one harmful message was sent through the guardrails system
- The response shows the content was blocked (e.g., `"status": "blocked"` in the output)

Once you see a blocked response, you are done. Do not continue exploring other commands.
