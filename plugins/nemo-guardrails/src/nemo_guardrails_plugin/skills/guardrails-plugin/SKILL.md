---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: guardrails-plugin
description: Use when working on guardrailing chat completions through the Inference Gateway API — creating guardrail configs, validating them with the `/checks` endpoint, attaching `nemo-guardrails` middleware to a VirtualModel, or proving guarded behavior.
allowed-tools: Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
  owner: guardrails
  maturity: active
---

# Guardrails Plugin

Use this skill for API-driven input and output rails on chat-completions traffic served by the NeMo Inference Gateway (IGW). Prefer the REST API throughout. Do not translate these operations into `nemo` CLI commands when an HTTP/API tool is available.

## API surfaces

All paths below are relative to the NeMo Platform base URL (locally, `http://localhost:8080`):

| Purpose | Method and path |
|---|---|
| List or create stored configs | `GET/POST /apis/guardrails/v2/workspaces/{workspace}/configs` |
| Read, update, or delete a config | `GET/PATCH/DELETE /apis/guardrails/v2/workspaces/{workspace}/configs/{name}` |
| Validate a config against messages | `POST /apis/guardrails/v2/workspaces/{workspace}/checks` |
| List or create VirtualModels | `GET/POST /apis/inference-gateway/v2/workspaces/{workspace}/virtual-models` |
| Read, update, or delete a VirtualModel | `GET/PATCH/DELETE /apis/inference-gateway/v2/workspaces/{workspace}/virtual-models/{name}` |
| Send OpenAI-compatible inference | `POST /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/chat/completions` |
| List routable VirtualModels | `GET /apis/inference-gateway/v2/workspaces/{workspace}/openai/-/v1/models` |

Use the deployment's normal authorization headers when authentication is enabled. Never print credentials or include them in saved examples.

## Preconditions

Before creating anything:

1. Verify `GET /health/ready` returns HTTP `200` and `{"status":"ready"}`. If it does not, stop and report the observed status. Do not work around an unhealthy platform.
2. Resolve an existing, reachable backend ModelEntity in `<workspace>/<name>` form. The examples below call it `default/<backend-model>`.
3. Use an auto-discovered ModelEntity served through IGW. Do not invent or manually register a served-model identifier.
4. For any existing config or VirtualModel with the requested name, fetch it first. Reuse it only if its definition matches the user's requested policy and routing; otherwise ask before replacing or patching it.

## API workflow

Follow this sequence for any Guardrails task. Derive config names, policies, test messages, workspace, backend model, and VirtualModel name from the user's request.

### 1. Create a stored guardrail config

Request:

```http
POST /apis/guardrails/v2/workspaces/default/configs
Content-Type: application/json
```

```json
{
  "name": "<config-name>",
  "description": "<what this policy protects>",
  "data": {
    "rails": {
      "input": {
        "flows": ["self check input"]
      }
    },
    "prompts": [
      {
        "task": "self_check_input",
        "content": "Check whether the user message violates this policy:\n<POLICY TEXT>\n\nUser message: {{ user_input }}\n\nShould this message be blocked (Yes or No)?\nAnswer:"
      }
    ]
  }
}
```

Require HTTP `201`. On `409`, fetch the named config and compare its `data` with the intended policy. Do not silently reuse a config with different rules. Use `PATCH` only when the user intends to update the existing policy.

The example is an input-only self-check rail. Adapt `rails`, `prompts`, and optional task models to the requested policy. For output protection, configure output flows and later attach the middleware call to `response_middleware`. For both phases, configure both and attach to both middleware lists.

### 2. Validate the config before attachment

Validate representative messages against the stored config:

```http
POST /apis/guardrails/v2/workspaces/default/checks
Content-Type: application/json
```

```json
{
  "model": "default/<backend-model>",
  "messages": [
    {
      "role": "user",
      "content": "<message expected to be blocked or allowed>"
    }
  ],
  "guardrails": {
    "config_id": "default/<config-name>"
  },
  "max_tokens": 256,
  "temperature": 0
}
```

Require HTTP `200` and compare the top-level `status` with the expected outcome:

- A message expected to violate the policy must return `"status": "blocked"`.
- A message expected to comply must return `"status": "success"`.

Treat `unknown`, a missing or unexpected status, a timeout, or any non-`200` response as validation failure. Test at least one blocked case and one allowed control whenever the policy is intended to distinguish them. Stop and report failures; never attach an unvalidated config to a VirtualModel.

### 3. Attach the config to a VirtualModel

Only after validation succeeds, create or update the target VirtualModel. This example attaches an input rail:

```http
POST /apis/inference-gateway/v2/workspaces/default/virtual-models
Content-Type: application/json
```

```json
{
  "name": "<virtual-model-name>",
  "default_model_entity": "default/<backend-model>",
  "models": [
    {
      "model": "default/<backend-model>",
      "backend_format": "OPENAI_CHAT"
    }
  ],
  "request_middleware": [
    {
      "name": "nemo-guardrails",
      "config_type": "guardrail_config",
      "config_id": "default/<config-name>"
    }
  ],
  "response_middleware": [],
  "post_response_middleware": []
}
```

Require HTTP `201` for creation. On `409`, fetch the existing VirtualModel and compare it with the requested routing and middleware. Use `PATCH` when the user intends to modify that VirtualModel; preserve unrelated middleware entries and their order unless the user explicitly asks to replace them.

For an input rail, place the call in `request_middleware`. For an output rail, place it in `response_middleware`. For a config with both input and output flows, put the same call in both lists.

### 4. Verify the inference path

Wait until `GET /apis/inference-gateway/v2/workspaces/default/openai/-/v1/models` includes `default/<virtual-model-name>`. The routing cache refreshes asynchronously, so poll the API rather than assuming the create or update response is immediately routable.

Request:

```http
POST /apis/inference-gateway/v2/workspaces/default/openai/-/v1/chat/completions
Content-Type: application/json
```

```json
{
  "model": "default/<virtual-model-name>",
  "messages": [
    {
      "role": "user",
      "content": "<verification message>"
    }
  ],
  "max_tokens": 256
}
```

Exercise the cases established during `/checks` validation and confirm inference behavior matches the policy. When an input or output rail blocks, the response content is expected to be `I'm sorry, I can't respond to that.` Confirm an allowed control still returns a normal response.

### 5. Read back and report

Fetch the config and VirtualModel after mutation. Report:

- The stored config ID and enabled input/output flows.
- The target VirtualModel and backend ModelEntity.
- Whether the middleware call is attached to request, response, or both phases.
- The `/checks` results and inference verification results.
- Any validation, cache propagation, or backend errors encountered.

## General config and middleware contract

A stored config create body is:

```json
{
  "name": "<config-name>",
  "description": "<optional description>",
  "data": {
    "rails": {},
    "prompts": []
  }
}
```

A Guardrails middleware call is always:

```json
{
  "name": "nemo-guardrails",
  "config_type": "guardrail_config",
  "config_id": "<workspace>/<config-name>"
}
```

Use `config_id` for stored, reusable configs. For development-only inline configs, replace `config_id` with `config`; never provide both.

Request middleware executes input flows. Response middleware executes output flows. A config containing both phases must be attached to both lists; attaching it to only one list silently omits the other phase.

## Rails config rules

The plugin consumes the standard `nemoguardrails` `RailsConfig` shape. The main LLM is the check request's `model` or, on the inference path, the backend ModelEntity selected by the VirtualModel. Entries with `type: "main"` in `models[]` are ignored; omit them.

Add `models[]` only for task LLMs such as `content_safety`, `topic_control`, `jailbreak_detection`, or `embeddings`. Those ModelEntity IDs must also be reachable through IGW.

See [Rails Config Reference](resources/rails-config.md) for full input/output examples, custom policies, and streaming output rails. See [Content Safety with a Task LLM](resources/content-safety.md) for classifier-backed moderation.

## Failure handling and gotchas

- The `guardrails` object in a normal chat-completions body contains runtime options only. It does not select the stored middleware config; the VirtualModel does that.
- The standalone `/checks` endpoint does accept `guardrails.config_id` and must be used before attachment.
- A VirtualModel create or update is persisted before IGW's routing cache sees it. Poll the OpenAI models endpoint; do not use fixed sleeps as proof of readiness.
- A stored config update changes `updated_at`; IGW resolves the new revision on cache refresh. Re-run both blocked and allowed `/checks` assertions after every update.
- Output rails reject `n > 1`. Streaming output rails require `rails.output.streaming.enabled: true` when a streaming block is configured.
- A task LLM or main LLM that is not reachable through IGW makes the workflow invalid. Stop and surface that dependency instead of substituting an arbitrary model.
- When a rail blocks on the inference path, the expected response content is exactly `I'm sorry, I can't respond to that.`

## Python SDK

When the assistant has the Python platform SDK rather than a generic HTTP tool, use the same API resources:

```python
from nemo_platform import NeMoPlatform

client = NeMoPlatform(base_url="http://localhost:8080", workspace="default")
configs = client.guardrail.configs.list()
```

Create configs through `client.guardrail.configs` and VirtualModels through `client.inference.virtual_models`. Preserve the same validation gates and read-back verification described above.
