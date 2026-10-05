<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Chat Completions via Inference Gateway (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

## Context

A mock inference model (`default/chat-model`) has been pre-configured in this environment. This model returns a fixed chat completion response to any prompt, so you can test the full chat completions flow without a real LLM backend.

## Task

Using the `nemo` CLI, complete the following:

1. Confirm the mock model is available by listing models or providers in the default workspace
2. Make a non-streaming chat completion request to `default/chat-model` with a user message (e.g., "What is the capital of France?")
3. Make a streaming chat completion request to the same model with a different user message

## Available CLI Commands

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

### Inference Provider/Model Commands

- `nemo inference providers list` - List all registered inference providers
- `nemo inference providers get <name>` - Get provider details

### Chat Completion Commands

- `nemo inference chat completions create --model <model_name> --messages '<json_array>'` - Send a non-streaming chat completion request
- `nemo inference chat completions create --model <model_name> --messages '<json_array>' --stream` - Send a streaming chat completion request

### Messages Format

Messages are passed as a JSON array: `'[{"role": "user", "content": "What is the capital of France?"}]'`

## Notes

- Streaming may not be fully supported by the mock provider - if streaming fails, that is acceptable as long as you attempted it.

## Success Criteria

The task is complete when:
- You have confirmed the mock model is available
- At least one chat completion request returned a response from the model
