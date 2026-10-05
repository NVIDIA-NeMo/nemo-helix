<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference via IGW with Provider (CLI)

You have access to the `nhx` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nhx` CLI is available at `/app/.venv/bin/nhx`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default. CLI auth is pre-configured.

## Context

An inference API key is available in the `ANTHROPIC_API_KEY` environment variable. This key works for NVIDIA's inference API at `https://inference-api.nvidia.com/v1`.

## Task

Using the `nhx` CLI, complete the following steps in order:

1. Create a secret named `nvidia-api-key` with the value from the `ANTHROPIC_API_KEY` environment variable
2. Register a model provider named `nvidia-inference` pointing to `https://inference-api.nvidia.com/v1` using the secret from step 1
3. Verify the provider was registered
4. Make a chat completions request through IGW using the `nvidia-inference` provider with model `aws/anthropic/bedrock-claude-sonnet-4-5-v1`
5. Confirm the response contains generated text

## Success Criteria

The task is complete when:
- A secret named `nvidia-api-key` exists
- A provider named `nvidia-inference` is registered in IGW pointing to `https://inference-api.nvidia.com/v1`
- A chat completions request through IGW has returned a valid response with generated text
