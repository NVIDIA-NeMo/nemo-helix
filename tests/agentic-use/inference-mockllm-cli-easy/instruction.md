<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MockLLM Provider in IGW (CLI)

You have access to the `nemo` CLI for NeMo Helix operations. Note: MCP tools are not available in this environment - you must use the CLI.

The `nemo` CLI is available at `/app/.venv/bin/nemo`. The CLI connects to the local NeMo Helix API server at http://localhost:8080 by default.

## Context

This environment has the Inference Gateway (IGW) running with **mock provider mode** enabled. In mock provider mode, any inference provider whose name starts with `igw-mock-` will return mock responses instead of proxying to a real backend. The mock response is controlled by setting the `X-Mock-Response` header (via the provider's `default_extra_headers`) to a JSON string containing the desired response body.

## Task

Using the `nemo` CLI, set up and exercise a MockLLM provider through the Inference Gateway:

1. **Create a mock inference provider** named `igw-mock-test-llm` in the `default` workspace. Configure it with:
   - A `host_url` (any value, e.g., `http://mock.local`, since the mock mode bypasses real backends)
   - A `default_extra_headers` field containing an `X-Mock-Response` header with the following JSON chat completion response:

     ```json
     {"id": "chatcmpl-mock-test", "object": "chat.completion", "created": 1700000000, "model": "test-llm", "choices": [{"index": 0, "message": {"role": "assistant", "content": "This is a deterministic mock response from the test LLM."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22}}
     ```

2. **Register a served model** on the provider so the model entity is discoverable. Use the provider status update to register model entity `default/test-llm` with served model name `test-llm`.

3. **Make an inference call** through the Inference Gateway using the provider route and confirm the response matches the expected deterministic output.

## Available CLI Commands

### Create a provider

```bash
nemo inference providers create \
  --workspace default \
  --name igw-mock-test-llm \
  --host-url http://mock.local \
  --default-extra-headers '{"X-Mock-Response": "{\"id\": \"chatcmpl-mock-test\", \"object\": \"chat.completion\", \"created\": 1700000000, \"model\": \"test-llm\", \"choices\": [{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": \"This is a deterministic mock response from the test LLM.\"}, \"finish_reason\": \"stop\"}], \"usage\": {\"prompt_tokens\": 10, \"completion_tokens\": 12, \"total_tokens\": 22}}"}'
```

### Register a served model on the provider

```bash
nemo inference providers update-status \
  --workspace default \
  --name igw-mock-test-llm \
  --input-data '{"served_models": [{"model_entity_id": "default/test-llm", "served_model_name": "test-llm"}]}'
```

### Make an inference call through the provider gateway route

```bash
nemo inference gateway provider post v1/chat/completions \
  --workspace default \
  --name igw-mock-test-llm \
  --body '{"model": "test-llm", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Other useful commands

```bash
# List providers
nemo inference providers list --workspace default

# Get provider details
nemo inference providers get igw-mock-test-llm --workspace default
```

## Hints

- The `--default-extra-headers` value must be a JSON string where the `X-Mock-Response` value is itself a JSON-encoded string (i.e., escaped JSON inside JSON)
- If passing complex JSON on the command line is difficult, you can write the JSON to a file and use `--input-file` or `--input-data` instead
- Use `--body` (not `--input-data`) for the inference gateway POST request

## Success Criteria

The task is complete when:
- A mock provider named `igw-mock-test-llm` exists in the `default` workspace
- The provider is configured with the mock response in its headers
- An inference call through the gateway returns the deterministic mock response
