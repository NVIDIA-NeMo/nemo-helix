<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Inference via IGW with Provider (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Context

An inference API key is available in the `ANTHROPIC_API_KEY` environment variable. This key works for NVIDIA's inference API at `https://inference-api.nvidia.com/v1`. Use the value of `ANTHROPIC_API_KEY` wherever an API key is needed.

CLI auth is pre-configured; you do not need to run `nemo auth login`.

## Task

Using the `nemo` CLI, complete the following steps in order:

1. Create a secret named `nvidia-api-key` with the value from the `ANTHROPIC_API_KEY` environment variable and description `NVIDIA inference API key`
2. Register a model provider named `nvidia-inference` with:
   - Host URL: `https://inference-api.nvidia.com/v1`
   - API key secret name: `nvidia-api-key`
   - Description: `NVIDIA inference provider`
3. Verify the provider was registered by listing providers
4. Make a chat completions request through IGW using the `nvidia-inference` provider with model `aws/anthropic/bedrock-claude-sonnet-4-5-v1` and a user message like "What is 2+2?" using `nemo inference gateway provider post`
5. Confirm the response contains generated text

## Available CLI Commands

### Secrets

```bash
# Create a secret
nemo secrets create <name> --value <value> --description <description>

# List secrets
nemo secrets list
```

### Inference Gateway Providers

```bash
# Register a provider
nemo inference providers create <name> --host-url <url> --api-key-secret-name <secret-name> --description <description>

# List providers
nemo inference providers list

# Make an inference call through a provider (use --body, not --input-data)
nemo inference gateway provider post chat/completions <provider-name> --body '{"model": "<model>", "messages": [{"role": "user", "content": "<prompt>"}], "max_tokens": 32}'
```

## Hints

- To read the API key value: `echo $ANTHROPIC_API_KEY`
- The secret value should be the actual key string, not the variable name
- For the inference call, use `--body` to pass the full JSON payload (not `--input-data`)

## Success Criteria

The task is complete when:
- A secret named `nvidia-api-key` exists
- A provider named `nvidia-inference` is registered in IGW pointing to `https://inference-api.nvidia.com/v1`
- A chat completions request through IGW has returned a valid response with generated text
