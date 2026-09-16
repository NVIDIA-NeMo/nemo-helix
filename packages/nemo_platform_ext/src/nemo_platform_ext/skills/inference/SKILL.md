---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: inference
description: >
  End-to-end reference for inference on the NeMo platform — registering LLM
  backends as ModelProviders, wiring them to VirtualModels with Switchyard
  middleware (random routing, translate) and `nemo-guardrails`
  content-safety rails, and hitting them via the nemo CLI. Use when the task
  involves registering inference providers, discovering served models,
  creating VirtualModels, configuring switchyard middleware, layering
  guardrails alongside translate (correct middleware ordering for
  OpenAI/Anthropic cross-format setups), making inference calls through IGW,
  or debugging routing and translation failures locally. For platform startup,
  Switchyard install, and DB-reset prerequisites, see the setup playbook
  (`SETUP.md` at the repo root).
preconditions:
  - nemo_setup_complete
user-invocable: true
allowed-tools: Bash, Read, Grep
---

# Inference Reference — ModelProvider + VirtualModel + Switchyard

## Prerequisites

This skill assumes the platform is already running and any required plugins
(`nemo-switchyard`, `nemo-guardrails`) are loaded. For local-platform startup,
Switchyard install, and DB-reset choices, follow the **setup playbook**
(`SETUP.md` at the repo root)
first and then return here.

## API key environment variables

Applies to both branches. Before creating secrets or running notebook/test
harness flows, confirm API key environment variables:

1. Check whether `INFERENCE_NVIDIA_API_KEY` and/or `NVIDIA_API_KEY` are already
   set in the user's shell.
2. Treat `INFERENCE_NVIDIA_API_KEY` as the source of truth for the
   `https://inference-api.nvidia.com/v1` provider secret and notebook/test
   harness flows.
3. If `INFERENCE_NVIDIA_API_KEY` is not set, ask whether the user has the same
   token in another environment variable. Common alias:
   `NVIDIA_INFERENCE_API_KEY`.
4. If the user identifies an alias variable, export it into
   `INFERENCE_NVIDIA_API_KEY` for the current shell/session before continuing.
5. If `NVIDIA_API_KEY` is required for a Build/NIM-style provider or seeding
   path such as `https://integrate.api.nvidia.com`, and is not set, ask
   whether another env var already contains that token and export it into
   `NVIDIA_API_KEY` if the user confirms.

Example alias normalization:

```bash
export INFERENCE_NVIDIA_API_KEY="$NVIDIA_INFERENCE_API_KEY"
```

Use `INFERENCE_NVIDIA_API_KEY` for inference-api secret creation:

```bash
printf '%s' "$INFERENCE_NVIDIA_API_KEY" | nemo secrets create nvidia-inference-key \
  --from-file - --workspace my-workspace
```

## Environment

- **Port**: `8080` (CLI default — do NOT pass a custom `--base-url`)
- **`export NMP_BASE_URL=http://localhost:8080` — required when targeting a local platform.** If your `~/.config/nmp/config.yaml` already points at a remote cluster, the CLI uses that base URL and ignores the local platform entirely. Setting this env var overrides the config file for the current shell session.
- Workspace: `my-workspace` (substitute as needed; `default` also works)
- Backend: `https://inference-api.nvidia.com/v1`

For platform startup (`nemo services run`), Switchyard install, and state reset, see the setup playbook (`SETUP.md` at the repo root).

---

## CLI gotchas (real failures observed)

- **`nemo secrets create`** uses `--from-file` (pipe key in). No `--value` flag.
- **`nemo inference providers create`** takes `<name>` as a **positional** arg.
  Same for `update-status`, `get`, `delete`.
- **`nemo inference virtual-models create`** takes `<name>` as positional.
- **There is no `nemo inference chat completions create` command.** Use
  `nemo inference gateway model post <path> <vm-name> --workspace <ws> --body '<json>'`.
- **`example` is not a valid `--services` arg.** Valid services: `audit`,
  `auth`, `customization`, `data-designer`, `entities`, `evaluation`, `files`,
  `guardrails`, `hello-world`, `inference-gateway`, `intake`, `jobs`, `models`,
  `safe-synthesizer`, `secrets`, `studio`.
- **The reconciler re-syncs `served_models` from backend discovery every few
  seconds and drops manually-registered entries.** Always point VM `models` at
  auto-discovered entity IDs (e.g. `my-workspace/aws-anthropic-claude-opus-4-5`)
  — these survive reconciler cycles. If a VM 404s shortly after working, the
  reconciler overwrote your entry. Re-run `update-status` with the **full list**
  (it replaces, not appends) or switch to auto-discovered IDs.
- **`update-status` replaces the entire `served_models` list** — include all
  entries, not just the new one.

---

## Step 1 — Workspace + Secret + Provider

`my-workspace` is not created automatically. Create it first, then the secret
and provider.

```bash
nemo workspaces create my-workspace

printf '%s' "$INFERENCE_NVIDIA_API_KEY" | nemo secrets create nvidia-inference-key \
  --from-file - --workspace my-workspace

nemo inference providers create nvidia-inference \
  --workspace my-workspace \
  --host-url "https://inference-api.nvidia.com/v1" \
  --api-key-secret-name "nvidia-inference-key"

nemo wait inference provider nvidia-inference --workspace my-workspace
```

Auto-discovery runs within ~3s.

For NVIDIA Build / NIM via `https://integrate.api.nvidia.com`, use
`NVIDIA_API_KEY`:

```bash
printf '%s' "$NVIDIA_API_KEY" | nemo secrets create nvidia-build-key \
  --from-file - --workspace my-workspace

nemo inference providers create nvidia-build \
  --workspace my-workspace \
  --host-url "https://integrate.api.nvidia.com" \
  --api-key-secret-name "nvidia-build-key"

nemo wait inference provider nvidia-build --workspace my-workspace
```

For Anthropic direct (needs auth header rewrite):

```bash
printf '%s' "$ANTHROPIC_API_KEY" | nemo secrets create anthropic-api-key \
  --from-file - --workspace my-workspace

nemo inference providers create anthropic \
  --workspace my-workspace \
  --host-url "https://api.anthropic.com" \
  --api-key-secret-name "anthropic-api-key" \
  --auth-header-format "X-Api-Key: {{ auth_secret }}" \
  --default-extra-headers '{"anthropic-version": "2023-06-01"}'
```

`{{ auth_secret }}` is Jinja2 — substituted at request time. Without it the
gateway defaults to `Authorization: Bearer` which Anthropic rejects.

---

## Step 2 — Discover available models

After the provider is ready, `served_models` contains auto-discovered entries.
**Always use these entity IDs** in VM `--models` — they survive the reconciler.

```bash
# All entity IDs (use these in --models and inference body)
nemo inference providers get nvidia-inference --workspace my-workspace \
  --output-format json | jq -r '.served_models[].model_entity_id'

# Side-by-side: served_model_name → model_entity_id
nemo inference providers get nvidia-inference --workspace my-workspace \
  --output-format json \
  | jq -r '.served_models[] | "\(.served_model_name)  →  \(.model_entity_id)"'

# Filter by vendor prefix
nemo inference providers get nvidia-inference --workspace my-workspace \
  --output-format json \
  | jq -r '.served_models[] | select(.served_model_name | startswith("aws/anthropic")) | .model_entity_id'

# Count
nemo inference providers get nvidia-inference --workspace my-workspace \
  --output-format json | jq '.served_models | length'
```

Entity ID normalization: slashes/dots → dashes, workspace prefix added.
`aws/anthropic/claude-opus-4-5` → `my-workspace/aws-anthropic-claude-opus-4-5`

---

## Step 3 — Body + model entity convention

`body["model"]` must be a **real backend entity ID** that IGW can resolve — not
the VM name. The VM name is resolved via the URL path
(`gateway model post v1/chat/completions <vm-name>`), not the body.

Exception: VMs that **rewrite** `body["model"]` (random_routing,
`ModelFormatLookupProcessor` in translate, native `stage_router` / `llm_classifier`)
— here the initial body model can be the VM name since the rewrite resolves it
to the real entity.

Native `stage_router` and `llm_classifier` require `switchyard_rust` in the IGW
process. Default platform images do not ship it; upsert is HTTP 400 until the
native wheel is installed in an isolated environment (do not overlay it on the
May `switchyard` vendor). Judge calls must use provider-direct URLs from
`get_inference_url_and_model` plus the provider's cached secret, not the VirtualModel gateway URL.

For VMs without a rewriting middleware: always send the real auto-discovered
entity in the body.

### Common failure: HTTP 422 from chat completion

If `nemo agents invoke …`, an `openai`-SDK client, or a `langchain-openai`
client returns HTTP 422 in well under a second with no provider call made,
the gateway rejected the request's `model` field as malformed. The most
common cause is sending the **upstream catalog name** (e.g.
`meta/llama-3.3-70b-instruct` — vendor slash, dotted version) instead of
the **hyphenated entity ID** (`meta-llama-3-3-70b-instruct`).

The entity ID is what `nemo models list` returns and what every gateway
input expects. The slash-with-dots form is `served_model_name` — a
human-display alias from the upstream provider's catalog. It appears in
provider metadata and error messages, but is **never** a valid request
field. (A `workspace/entity-id` prefix is fine — that's a workspace
qualifier, not the upstream alias. The rejected form is specifically
`vendor/upstream.dotted.name`.)

Diagnose and recover:

```bash
# 1. What entity IDs actually exist?
nemo inference providers get nvidia-build --workspace default \
  --output-format json \
  | jq -r '.served_models[] | "\(.served_model_name)  →  \(.model_entity_id)"'

# 2. If you set NEMO_DEFAULT_MODEL, confirm it's the hyphenated entity ID.
echo "$NEMO_DEFAULT_MODEL"

# 3. If a NAT-style deployed agent's resolved config carries the upstream
#    slash form in model_name, re-deploy with NEMO_DEFAULT_MODEL set to the
#    entity ID. The .config.llms.agent.model_name path is specific to NAT
#    workflows (e.g. the calculator-agent example); other workflow types store
#    the model elsewhere.
nemo agents deployments list --workspace default \
  | jq -r '.data[] | "\(.name)  model=\(.config.llms.agent.model_name // "n/a")"'
```

For an OpenAI-SDK or LangChain client, pass the entity ID (with or without
the `workspace/` prefix) as `model=`:

```python
client.chat.completions.create(
    model="default/meta-llama-3-3-70b-instruct",   # OR "meta-llama-3-3-70b-instruct"
    messages=[...],
)
# NOT: model="meta/llama-3.3-70b-instruct"  → 422
```

This is intentional gateway behavior: accepting arbitrary slash-prefixed
names would force IGW to guess provider attribution for every request
instead of resolving it from a registered entity.

---

## Step 4 — VirtualModel patterns

### Random routing — same format (deterministic test: `strong_probability=1.0`)

```bash
nemo inference virtual-models create vm-random-strong --workspace my-workspace \
  --models '[
    {"model":"my-workspace/nvidia-mistralai-mixtral-8x22b-instruct-v01","backend_format":"OPENAI_CHAT"},
    {"model":"my-workspace/nvidia-qwen-qwen3-32b","backend_format":"OPENAI_CHAT"}
  ]' \
  --request-middleware '[{"name":"nemo-switchyard","config_type":"random_routing","config":{
    "strong":{"model":"my-workspace/nvidia-mistralai-mixtral-8x22b-instruct-v01"},
    "weak":{"model":"my-workspace/nvidia-qwen-qwen3-32b"},
    "strong_probability":1.0,
    "rng_seed":42,
    "enable_stats":false
  }}]'
```

### Random routing — cross format (chain routing + translate)

`random_routing` picks the backend; `translate` rewrites the request format.
Order matters: routing first, translate second. Use `response_middleware` too
for full round-trip translation back to the client's format.

```bash
nemo inference virtual-models create vm-random-cross --workspace my-workspace \
  --models '[
    {"model":"my-workspace/aws-anthropic-claude-opus-4-5","backend_format":"ANTHROPIC_MESSAGES"},
    {"model":"my-workspace/nvidia-nvidia-nemotron-nano-31b-v3","backend_format":"OPENAI_CHAT"}
  ]' \
  --request-middleware '[
    {"name":"nemo-switchyard","config_type":"random_routing","config":{
      "strong":{"model":"my-workspace/aws-anthropic-claude-opus-4-5"},
      "weak":{"model":"my-workspace/nvidia-nvidia-nemotron-nano-31b-v3"},
      "strong_probability":0.5,
      "enable_stats":false
    }},
    {"name":"nemo-switchyard","config_type":"translate","config":{"target_format":"auto","enable_stats":false}}
  ]' \
  --response-middleware '[{"name":"nemo-switchyard","config_type":"translate","config":{"target_format":"auto","enable_stats":false}}]'
```

### Translate — cross format, full round-trip ✓

Client sends OpenAI shape, backend is Anthropic, response comes back as OpenAI.
**Must list translate in BOTH `request_middleware` and `response_middleware`.**

```bash
nemo inference virtual-models create vm-translate-cross --workspace my-workspace \
  --models '[{"model":"my-workspace/aws-anthropic-claude-opus-4-5","backend_format":"ANTHROPIC_MESSAGES"}]' \
  --request-middleware '[{"name":"nemo-switchyard","config_type":"translate","config":{"target_format":"anthropic","enable_stats":false}}]' \
  --response-middleware '[{"name":"nemo-switchyard","config_type":"translate","config":{"target_format":"anthropic","enable_stats":false}}]'
```

### Guardrails — adding content safety to a VirtualModel

Guardrail rails are attached as a `nemo-guardrails` MiddlewareCall. Configs are
created separately with `nemo guardrail configs create` — see the
**nemo-guardrails** skill for rails config JSON, prompt templates, task LLMs,
and streaming output rails. This section only covers the VirtualModel wiring.

The same call must appear in `--request-middleware` (input rails),
`--response-middleware` (output rails), or both (full coverage). A call only on
the request side fires input rails only; only on the response side fires output
rails only.

**Output rails only** — block bad bot responses (most common):

```bash
nemo inference virtual-models create vm-guarded --workspace my-workspace \
  --models '[{"model":"my-workspace/<backend-entity-id>","backend_format":"OPENAI_CHAT"}]' \
  --response-middleware '[{
    "name":"nemo-guardrails",
    "config_type":"guardrail_config",
    "config_id":"my-workspace/content-safety"
  }]'
```

**Input + output rails** — full coverage. Include the call in **both** lists:

```bash
nemo inference virtual-models create vm-guarded-full --workspace my-workspace \
  --models '[{"model":"my-workspace/<backend-entity-id>","backend_format":"OPENAI_CHAT"}]' \
  --request-middleware '[{"name":"nemo-guardrails","config_type":"guardrail_config","config_id":"my-workspace/content-safety"}]' \
  --response-middleware '[{"name":"nemo-guardrails","config_type":"guardrail_config","config_id":"my-workspace/content-safety"}]'
```

The guardrails plugin doesn't route — the VirtualModel's `--models` array
decides the upstream backend. Inline configs (no stored entity) are also
supported via `"config":{...}` instead of `"config_id"`.

#### Guardrails + Switchyard `translate` — ordering matters

The `nemo-guardrails` plugin only understands **OpenAI chat completions**
shape. When the backend is in a different format (e.g. `ANTHROPIC_MESSAGES`)
and a `translate` middleware is also in the chain, guardrails must see the
OpenAI form on both sides:

- **Request stack:** `guardrails` BEFORE `translate`. Guardrails inspects the
  OpenAI request from the client, then translate rewrites it to the backend
  format.
- **Response stack:** `guardrails` AFTER `translate`. Translate converts the
  backend response back to OpenAI first, then guardrails runs output rails on
  the OpenAI shape.
- **Inference calls:** keep the request body in OpenAI shape regardless of
  the backend's `backend_format`. The translate middleware handles the
  conversion to/from backend format; sending Anthropic-shaped bodies bypasses
  guardrails' input rails because the plugin can't parse them.

```bash
nemo inference virtual-models create vm-guarded-translate --workspace my-workspace \
  --models '[{"model":"my-workspace/aws-anthropic-claude-opus-4-5","backend_format":"ANTHROPIC_MESSAGES"}]' \
  --request-middleware '[
    {"name":"nemo-guardrails","config_type":"guardrail_config","config_id":"my-workspace/content-safety"},
    {"name":"nemo-switchyard","config_type":"translate","config":{"target_format":"anthropic","enable_stats":false}}
  ]' \
  --response-middleware '[
    {"name":"nemo-switchyard","config_type":"translate","config":{"target_format":"anthropic","enable_stats":false}},
    {"name":"nemo-guardrails","config_type":"guardrail_config","config_id":"my-workspace/content-safety"}
  ]'
```

Same principle applies when `random_routing` mixes OpenAI and Anthropic
backends behind one VirtualModel: put `guardrails` first in the request chain
(before any routing/translate decisions), and put `guardrails` last in the
response chain (after translate has normalized everything back to OpenAI).

If a rails config declares a task LLM via `models[]` (e.g. `content_safety`,
`topic_control`), the task LLM must itself be addressable as OpenAI chat
completions — point `models[].model` at an OpenAI-format entity ID, not at an
Anthropic-format backend. The guardrails plugin doesn't translate task-LLM
calls.

---

## Step 5 — Making inference calls

**Preferred (nemo CLI):**

```bash
nemo inference gateway model post v1/chat/completions <vm-name> \
  --workspace my-workspace \
  --body '{"model":"my-workspace/<entity-id>","messages":[{"role":"user","content":"hi"}],"max_tokens":15}'
```

**Verify routing by running multiple times** (model field in response flips
between backends at the configured probability). Parse with Python — see the
parsing-pitfalls subsection below for why `jq` is the wrong tool here:

```bash
for i in $(seq 1 10); do
  nemo inference gateway model post v1/chat/completions vm-random-cross \
    --workspace my-workspace \
    --body '{"model":"my-workspace/vm-random-cross","messages":[{"role":"user","content":"hi"}],"max_tokens":400}' \
    | python3 -c 'import json,sys; d=json.loads(sys.stdin.read(), strict=False); print(d.get("model"))'
done
```

For reasoning models (e.g. `nemotron-nano-*`), use `max_tokens` ≥ ~200; the
model spends most of its budget on a hidden reasoning trace and produces an
empty `content` if cut off mid-trace (`finish_reason: length`). That is not
an inference error, just a budget shortfall.

**curl (streaming or when you need raw output):**

```bash
curl -s -X POST \
  "http://localhost:8080/apis/inference-gateway/v2/workspaces/my-workspace/openai/-/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"my-workspace/<entity-id>","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

### Parsing pitfalls — `jq` and reasoning-model responses

Reasoning models (`nemotron-nano-*` and similar) sometimes emit responses
that contain raw control characters (literal U+0000–U+001F bytes — typically
unescaped newlines or tabs) inside the `reasoning_content` field. This is
technically invalid JSON. Python's `json.loads` rejects it by default but can
be configured with `strict=False` to accept it. **`jq` does not** — it bails
out with:

```
jq: parse error: Invalid string: control characters from U+0000 through
U+001F must be escaped at line N, column M
```

In a routing-verification loop driven by `jq -r '.model'`, every nemotron
response will look like a failure (empty `model` field) while every opus
response parses fine — making the split look catastrophically broken when
inference is actually working. **Always parse these responses with Python
using `strict=False`** (or a similarly lenient parser) instead of `jq`:

```bash
echo "$resp" | python3 -c '
import json, sys
d = json.loads(sys.stdin.read(), strict=False)
print(d.get("model"), d["choices"][0].get("finish_reason"))
'
```

If you must stay in shell, `jq --slurp -R 'fromjson?'` will tolerate most of
these payloads but is brittle and not worth the effort. Python is the right
tool.

---

## Verifying routing decisions (DEBUG logs)

Start with `LOG_LEVEL=DEBUG`. Filter service output:

```bash
grep -E "RandomRoutingRequestProcessor|picked tier|FormatTranslate|StampOriginalFormat" \
  <service-output-file>
```

Expected output:
```
debug RandomRoutingRequestProcessor: picked tier=strong model=my-workspace/aws-anthropic-claude-opus-4-5
debug RandomRoutingRequestProcessor: picked tier=weak   model=my-workspace/nvidia-nvidia-nemotron-nano-31b-v3
```

`tier=` and `model=` in DEBUG logs match `response.model` returned to the client.

---

## Failure cases

### ❌ Translate in `request_middleware` only — response shape mismatch

The most common mistake. Request translates fine, backend call succeeds, but the
client receives the backend's **native** response shape instead of OpenAI:

| Field | ✓ translate in both | ✗ request only |
|---|---|---|
| `id` prefix | `chatcmpl-` (OpenAI) | `msg_` (Anthropic) |
| Shape | `choices[0].message.content` | `content[0].text` |
| Extra keys | — | `type`, `stop_reason`, `stop_sequence` |

Fix: always add `response_middleware` with the same translate config.

### ❌ Path mismatch without translate → 502

Sending `/v1/messages` traffic to a VM backed by an OpenAI-format model with no
translate fails because IGW forwards the inbound path unchanged. Switchyard's
`PathUpdateProcessor` only rewrites `request.path` when translate is configured.

### ❌ Reconciler-induced 404

VM works briefly then returns "Model entity not found". The reconciler overwrote
`served_models` from upstream auto-discovery. Fix: use auto-discovered entity IDs
in VM `--models` (they survive reconciler cycles) instead of manually-registered
aliases.

### ❌ NVIDIA hub silently accepts format mismatches

OpenAI body at `/v1/chat/completions` routed to an Anthropic backend **without**
translate does not fail at the upstream — NVIDIA hub accepts OpenAI-shaped bodies
for Anthropic models and returns the Anthropic-native response. The visible
failure is the response shape mismatch above (case 1), not a 4xx/5xx.

### ❌ Guardrails after translate (request) or before translate (response)

`nemo-guardrails` parses **OpenAI** chat shape only. With an Anthropic backend
and `request_middleware=[translate, guardrails]`, guardrails sees the
already-translated Anthropic body and either skips its rails or errors on the
unexpected shape. Symmetrically, `response_middleware=[guardrails, translate]`
runs guardrails against the raw Anthropic response. Fix: keep guardrails on the
OpenAI side of translate in both chains — `request=[guardrails, translate]`,
`response=[translate, guardrails]`. Same applies to clients sending
Anthropic-shaped request bodies: input rails won't fire. Send OpenAI shape and
let translate do the conversion.

---

## Troubleshooting

**DB disk I/O error on startup** — use the `nemo-teardown` skill's **stop +
wipe data** flow, then rerun setup. Do not delete live platform data directly.

**`nemo-switchyard` fails to load at startup** — `switchyard.lib` not importable.
Run `uv sync` from the repo root with default groups enabled to install the
plugin and its vendored `switchyard` library
(`plugins/nemo-switchyard/vendor/switchyard/`), then restart services.

**401 from Anthropic** — missing or wrong `--auth-header-format`. Verify:
```bash
nemo inference providers get anthropic --workspace my-workspace --output-format json \
  | jq '.auth_header_format'
```

**No served models after provider creation** — wait ~10s for the first reconciler
cycle, then check:
```bash
nemo inference providers get nvidia-inference --workspace my-workspace \
  --output-format json | jq '.served_models | length'
```

---

## Cleanup

```bash
# Delete only the VirtualModels created by the examples in this skill.
created_vms=(
  vm-random-strong
  vm-random-cross
  vm-translate-cross
  vm-guarded
  vm-guarded-full
  vm-guarded-translate
)

printf 'Delete example resources in my-workspace (VirtualModels: %s; provider: nvidia-inference; secret: nvidia-inference-key; workspace: my-workspace)? Type DELETE to continue: ' "${created_vms[*]}"
read -r confirmation
if [ "$confirmation" != "DELETE" ]; then
  echo "Cleanup cancelled." >&2
  exit 1
fi

for vm in "${created_vms[@]}"; do
  nemo inference virtual-models delete "$vm" --workspace my-workspace
done

nemo inference providers delete nvidia-inference --workspace my-workspace
nemo secrets delete nvidia-inference-key --workspace my-workspace
nemo workspaces delete my-workspace
```
