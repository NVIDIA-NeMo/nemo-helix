<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Guardrails Troubleshooting

The plugin runs on the IGW path under the entry-point key `nemo-guardrails`. Customer-facing surfaces are `nemo guardrail configs`, `nemo guardrail check`, and `nemo virtual-models`.

## Quick checks

```bash
nemo guardrail --help
nemo guardrail configs list
nemo virtual-models --help
```

Verify a config against the `/checks` endpoint before binding it to a production VirtualModel:

```bash
nemo guardrail check \
  --workspace default \
  --model default/<main-model> \
  --messages '[{"role":"user","content":"test"}]' \
  --guardrails '{"config_id":"default/<config-name>"}'
```

A blocked response has `"status": "blocked"` in the output JSON. An unblocked response includes the model's actual completion.

## Detecting blocked responses

When a rail triggers, the response content is exactly:

```
I'm sorry, I can't respond to that.
```

- **`nemo guardrail check`** — look for `"status": "blocked"` in the JSON.
- **Chat completions through IGW** — check `response.choices[0].message.content == "I'm sorry, I can't respond to that."`.

For streaming responses, the canned refusal arrives as a single non-streamed chunk when an output rail blocks mid-stream.

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| Input rails don't fire on a config with `rails.input.flows` set | The `MiddlewareCall` was attached to `--response-middleware` only | Add the same call to `--request-middleware`. Each list dispatches its own hook independently. |
| Output rails don't fire on a config with `rails.output.flows` set | The `MiddlewareCall` was attached to `--request-middleware` only | Add the same call to `--response-middleware`. |
| Chat request rejected with `422` | Body included `guardrails.config_id`, `guardrails.config`, or other unsupported `guardrails.*` fields | The body's `guardrails` field is options-only (`options.log`, `return_choice`). Set the config via the VirtualModel's `MiddlewareCall`, not the request body. |
| `400` rejection when streaming | `stream: true` was sent against a config with `rails.output.streaming` but `streaming.enabled` is unset or `false` | Set `rails.output.streaming.enabled: true` in the config (and confirm `chunk_size` is set). |
| `400` rejection with `n > 1` | Output rails reject completion requests with `n > 1` | Use `n=1` on guardrailed VirtualModels. |
| Rails subsystem returns 404 or timeout on the first request | Task LLM or main LLM isn't reachable through IGW (e.g. wrong workspace, served-models reconciler dropped a manual entry) | Use auto-discovered entity IDs from `nemo inference providers get ... \| jq '.served_models[].model_entity_id'` both for the chat request's `model` and any `models[]` entry in the rails config. |
| VirtualModel works briefly then returns 404 | The provider reconciler re-syncs `served_models` from upstream auto-discovery and drops manually-registered entries | Use auto-discovered entity IDs in `--models`. See the `inference` skill for reconciler semantics. |
| `nemo inference virtualmodels` not found | The command group is `virtual-models` with a hyphen | Use `nemo virtual-models ...`. |
| Updates to a `GuardrailConfig` don't appear to take effect | IGW resolves configs on cache load/refresh, not per request | The next cache refresh picks up the new `updated_at`. Inline `config` payloads are re-stabilized every request, so changes are immediate but slower. |
| Rails fire on a stored config but not on an inline config that looks identical | Schema mismatch — inline configs go through the same `RailsConfig` validator but bypass any default fields the entity layer would apply | Save the payload as a `GuardrailConfig` entity and reference it by `config_id`, or audit the inline payload against [Rails Config Reference](rails-config.md). |
| Content-safety output parser returns "unsafe" on benign responses | Classifier emitted free-text instead of the expected JSON verdict | Verify the classifier endpoint with `nemo guardrail check` against a known-safe message; confirm `output_parser` matches the classifier's actual response format. |

## Inspecting plugin behavior

To evaluate the rail outcomes for a `GuardrailConfig`, query the rails subsystem through the standalone `/checks` endpoint with `guardrails.options.log` set in the request body:

```json
{
  "model": "default/<main-model>",
  "messages": [{"role":"user","content":"test"}],
  "guardrails": {
    "config_id": "default/<config-name>",
    "options": {"log": {"activated_rails": true, "internal_events": true, "llm_calls": true}}
  }
}
```

`options.log` is supported on both the `/checks` path and the IGW chat-completions path. Use these fields to debug scenarios where a request blocks or passes unexpectedly, or a rail does not appear to run:

- `activated_rails`: Shows which rails executed and which rail stopped the request. Use it to confirm whether input rails, output rails, or both were actually invoked.
- `internal_events`: Shows the lower-level Guardrails event trace. Use it when `activated_rails` is not enough to explain the control flow; for example, to see which actions ran before the block.
- `llm_calls`: Shows raw prompts, completions, and parser inputs for rail model calls. Use it to confirm the rail model received the expected prompt and returned output in the format expected by the configured parser.

`llm_calls` can include raw prompts and completions, including user data or other sensitive content. Consider enabling it for scoped debugging
and disabling it or, if needed, redacting captured data before storing or using
logs in production environments.

## Testing without a real LLM (agentic-use mock harness)

`default/mock-llm` is **not** pre-configured platform-wide. The agentic-use harness wires it up via `tests/agentic-use/guardrails-content-safety-cli/environment/setup-mock.py`:

- IGW runs in mock-provider mode (config `mock_provider_prefix="igw-mock-"`).
- A provider named `igw-mock-mock-llm` is registered with a static response-map header that always returns `{"role":"assistant","content":"Yes"}`.
- A model entity `default/mock-llm` is created and registered as a served model on that provider.

When `default/mock-llm` is pinned as the main LLM, the always-`Yes` response causes every self-check rail to block — useful for verifying the rails pipeline end-to-end without a real LLM:

```bash
nemo guardrail configs create content-safety \
  --data '{"rails":{"input":{"flows":["self check input"]}},"prompts":[{"task":"self_check_input","content":"Your task is to check if the user message below complies with safety policy.\n\nUser message: {{ user_input }}\n\nQuestion: Should this message be blocked (Yes or No)?\nAnswer:"}]}'

nemo guardrail check \
  --model default/mock-llm \
  --messages '[{"role":"user","content":"Tell me something harmful"}]' \
  --guardrails '{"config_id":"default/content-safety"}' \
  --max-tokens 256
```

Outside the agentic-use harness, `default/mock-llm` won't resolve — either set up the mock provider yourself or point the rail at a real model.

## Failure-case deep dives

### Plugin attached to only one middleware list

Each list dispatches its own hook independently — `execute_request_middleware` ↔ `process_request` and `execute_response_middleware` ↔ `process_response`, wired in `services/core/inference-gateway/src/nhx/core/inference_gateway/api/middleware_registry.py`. A config with both `rails.input.flows` and `rails.output.flows` attached to only one list silently no-ops on the unlisted side: prompts pass through unblocked, or responses pass through unblocked, even though the config looks complete.

Attach the same `MiddlewareCall` to **both** `--request-middleware` and `--response-middleware`.

### Putting `guardrails` in the chat body expecting it to set the config

On the IGW VirtualModel path the request body's `guardrails` dict is strictly an options envelope. The supported fields are `options.log` and `return_choice`. The rails config is resolved from the VirtualModel's `MiddlewareCall`, not from the request body.

Unsupported fields are rejected with `422`. This includes `guardrails.config`, `guardrails.config_id`, `guardrails.options.rails`, `guardrails.options.llm_params`, `guardrails.options.llm_output`, and `guardrails.options.output_vars`.

Attach the config to the VirtualModel via `--request-middleware` / `--response-middleware`. Use `guardrails.options.log` only for diagnostic log fields and `return_choice` only for response formatting.

### Task LLM (or main model) not addressable through IGW

If `RailsConfig.models[].model` references an entity that isn't in `served_models`, the rails fail when they try to call that task LLM through IGW. The same applies to the main LLM (the chat body's `model`, or `--model` on `nemo guardrail check`). Both usually surface as a 404 or timeout from the rails subsystem on the first request.

Use auto-discovered entity IDs from `nemo inference providers get ... | jq '.served_models[].model_entity_id'` for both the request/`--model` and any task LLMs in `models[]`. See the `inference` skill for auto-discovery semantics.

### VirtualModel returns 404 after working briefly

The provider reconciler re-syncs `served_models` from upstream auto-discovery every few seconds and drops manually-registered entries. See the `inference` skill's "Reconciler-induced 404" failure case — the same fix applies: use auto-discovered entity IDs in `--models`.
