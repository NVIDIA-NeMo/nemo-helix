<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Running the Example Inference Middleware Plugin Locally

End-to-end guide for running the platform locally against NVIDIA's public
inference API (`integrate.api.nvidia.com`) with the example plugin loaded.

---

## Prerequisites

- `NGC_API_KEY` set in your environment
- Example plugin installed in the workspace venv (first time only):

  ```bash
  uv pip install -e plugins/example-plugin
  ```

  Verify entry-points registered:

  ```bash
  uv run python -c "
  import importlib.metadata
  for ep in importlib.metadata.entry_points(group='nemo.inference_middleware'):
      print(ep)
  "
  # → nemo-example-middleware = nemo_example_plugin.middleware:ExampleInferenceMiddleware
  ```

---

## 1. Start the platform

```bash
uv run nemo services run \
  --services entities,models,inference-gateway,example,secrets \
  --controllers models
```

`--services` is needed because the bare default starts *all* installed services,
including Docker-dependent ones (customizer, data-designer, etc.) that crash
without infra. `--controllers models` starts the models controller so it can
call `GET /v1/models` on the provider and auto-create model entities.
Port defaults to 8080.  Auth defaults to disabled for local runs.
SQLite is the default entity store when no `DATABASE_*` vars are set.

Leave this running in a terminal. Confirm it's up:

```bash
curl http://localhost:8080/apis/example/hello/world
# → {"message":"Hello, world!"}
```

To reset state between sessions (wipe SQLite DB and encryption key):

```bash
rm -rf ~/.local/share/nemo
```

---

## 2. Create workspace and store NGC API key

```bash
export NHX_BASE_URL=http://localhost:8080

nemo workspaces create my-workspace

echo "$NGC_API_KEY" | nemo secrets create ngc-api-key \
  --workspace my-workspace \
  --from-file -
```

---

## 3. Create the ModelProvider

```bash
nemo inference providers create nvidia-cloud \
  --workspace my-workspace \
  --host-url "https://integrate.api.nvidia.com" \
  --api-key-secret-name "ngc-api-key"
```

---

## 4. Wait for the models controller to discover served models

The models controller calls `GET /v1/models` on every provider on each
reconcile cycle and auto-creates model entities for what it finds.  Wait a
few seconds then inspect what was discovered:

```bash
nemo inference providers get nvidia-cloud --workspace my-workspace
```

Note the normalized entity IDs in `served_models` — dots become dashes and
the org prefix is retained, e.g. `meta/llama-3.1-8b-instruct` becomes
`my-workspace/meta-llama-3-1-8b-instruct`.  The NGC endpoint returns a large
catalog; use `enabled_models` to restrict it if needed:

```bash
nemo inference providers update nvidia-cloud \
  --workspace my-workspace \
  --enabled-models "nvidia/llama-3.3-nemotron-super-49b-v1"
```

---

## 5. Create middleware config entity (optional — for config_id path)

Skip this if testing with inline config (see VirtualModel examples below).

```bash
nemo example middleware-configs create my-filter \
  --workspace my-workspace \
  --blocked-keywords "violence,drugs" \
  --block-message "That topic is not permitted."
```

---

## 6. Create a VirtualModel with the example middleware

### Option A — inline config

Use the normalized entity ID from step 4 (e.g. `my-workspace/nvidia-llama-3-3-nemotron-super-49b-v1`):

```bash
nemo inference virtual-models create safe-llama \
  --workspace my-workspace \
  --default-model-entity "my-workspace/nvidia-llama-3-3-nemotron-super-49b-v1" \
  --request-middleware '[{"name":"nemo-example-middleware","config_type":"example_middleware_config","config":{"blocked_keywords":["violence","drugs"],"block_message":"That topic is not permitted."}}]' \
  --response-middleware '[{"name":"nemo-example-middleware","config_type":"example_middleware_config","config":{"blocked_keywords":["chlorophyll"],"block_message":"Response contained restricted content."}}]'
```

### Option B — config_id reference (requires step 5)

```bash
nemo inference virtual-models create safe-llama \
  --workspace my-workspace \
  --default-model-entity "my-workspace/nvidia-llama-3-3-nemotron-super-49b-v1" \
  --request-middleware '[{"name":"nemo-example-middleware","config_type":"example_middleware_config","config_id":"my-workspace/my-filter"}]'
```

List and inspect:

```bash
nemo inference virtual-models list --workspace my-workspace
nemo inference virtual-models get safe-llama --workspace my-workspace
```

---

## 7. Test inference through the middleware

### Non-streaming — passes through (no blocked keywords)

```bash
curl -s -X POST \
  "http://localhost:8080/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-workspace/safe-llama",
    "messages": [{"role": "user", "content": "Tell me a one-sentence fun fact about the Moon."}]
  }' | python3 -m json.tool
# → full chat completion from the upstream model
```

### Non-streaming — blocked request (ImmediateResponse, backend never called)

```bash
curl -s -X POST \
  "http://localhost:8080/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-workspace/safe-llama",
    "messages": [{"role": "user", "content": "How do drugs affect the brain?"}]
  }' | python3 -m json.tool
# → {"id":"blocked","choices":[{"finish_reason":"content_filter","message":{"content":"That topic is not permitted."}}]}
```

### Streaming — response middleware redacts keywords across chunk boundaries

```bash
curl -s -X POST \
  "http://localhost:8080/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-workspace/safe-llama",
    "messages": [{"role": "user", "content": "In one sentence, explain why many leaves look green and include the word chlorophyll."}],
    "stream": true
  }'
# → SSE stream with "[REDACTED]" replacing "chlorophyll" in the delta content chunks
```

---

## Cleanup (optional)

Delete entities in reverse-dependency order (VirtualModel → middleware config →
provider → secret → workspace).

```bash
export NHX_BASE_URL=http://localhost:8080

# VirtualModel
nemo inference virtual-models delete safe-llama --workspace my-workspace

# Middleware config (only if created in step 5)
nemo example middleware-configs delete my-filter --workspace my-workspace

# Model provider
nemo inference providers delete nvidia-cloud --workspace my-workspace

# Secret
nemo secrets delete ngc-api-key --workspace my-workspace

# Workspace (marks all remaining entities for async deletion)
nemo workspaces delete my-workspace
```

---

## Teardown

`Ctrl-C` the `nemo services run` process, then optionally wipe the SQLite DB
and encryption key for a clean slate next time:

```bash
rm -rf ~/.local/share/nemo
```
