---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-guardrails
description: >
  NeMo guardrails CLI reference for creating guardrail configs and applying them
  to chat completions through an IGW VirtualModel. Covers config CRUD, the
  standalone `/checks` endpoint for verifying a config, and the
  `nemo-guardrails` inference middleware plugin attached via VirtualModel
  MiddlewareCalls. Use when the task involves guardrail configurations, content
  safety, input/output rails, or `nemo guardrail` / `nemo inference virtual-models` CLI
  commands for guardrailing inference.
preconditions:
  - nemo_setup_complete
  - workspace_exists
  - guardrails_plugin_available
user-invocable: true
allowed-tools: Bash, Read, Grep
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Guardrails CLI Reference

`nemo guardrail` (config CRUD + the `/checks` endpoint) and `nemo inference virtual-models`
with a `nemo-guardrails` MiddlewareCall both back the same `guardrail_config`
entity. The `nemo guardrail chat completions create` and `nemo guardrail
completions create` commands are deprecated; do not use them.

## Environment

- **CLI paths**: `/app/.venv/bin/nemo` and `/app/.venv/bin/nmp` (same binary,
  two names)
- **API server**: `http://localhost:8080` (default, no auth needed for local services)
- **Default workspace**: `default`
- **Workspace override**: `--workspace <name>` flag, or `NMP_WORKSPACE` env var
- **Entity ID**: `<workspace>/<model-entity-name>`, e.g.
  `default/meta-llama-3-70b-instruct`. For backend models behind a provider,
  use the auto-discovered ID from `served_models[].model_entity_id` (see
  `inference` skill).
- **Canned refusal**: when a rail blocks, the response content is exactly
  `"I'm sorry, I can't respond to that."`

## Prerequisites

- For **`nemo guardrail check`** alone: just a config and model entities
  reachable through IGW (so the main LLM and any task LLMs can resolve).
- For **chat-time guardrailing on a VirtualModel**: a real backend model entity
  the VirtualModel can proxy to. Use `nemo-secrets` and `inference`
  skills to register a secret + provider; the `inference` skill is
  the end-to-end reference for the provider/served-models/auto-discovery flow.

---

## CLI gotchas

- **Use `nemo inference virtual-models` (with hyphen)**, not
  `nemo inference virtualmodels`.
- **Names are positional** for `nemo guardrail configs create <name>` and
  `nemo inference virtual-models create <name>`.
- **No `nemo inference chat completions create` command exists.** Use
  `nemo inference gateway model post v1/chat/completions <vm-name>` (matches
  the `inference` skill).
- **`--data` (configs) and `--body` / `--models` / `--*-middleware`
  (VirtualModels) take JSON strings.** Watch shell quoting for embedded
  backticks/quotes — prefer `--input-file` for large payloads.
- **`config_type` is always `"guardrail_config"`** in a `MiddlewareCall` for
  `nemo-guardrails`.
- **Each middleware list dispatches its own hook independently.** If you want
  input rails AND output rails, the call must appear in **both**
  `--request-middleware` AND `--response-middleware`. See `Failure cases`.
- **The `guardrails` field on a chat-completions request body is options-only
  on the IGW path** (e.g. `options.log`, `return_choice`). It does **not**
  set the rails config — that comes from the VirtualModel's `MiddlewareCall`.
  This is the key behavior difference from the deprecated `nemo guardrail chat
  completions create --guardrails '{"config_id":...}'`.

---

## Config CRUD Commands

```bash
# List configs in current workspace
nemo guardrail configs list

# Create a config (positional name + --data JSON)
nemo guardrail configs create <name> \
  --description "<description>" \
  --data '<json>'

# Retrieve a config by name
nemo guardrail configs get <name>

# Update a config (pass only fields to change)
nemo guardrail configs update <name> --description "<new description>"

# Delete a config
nemo guardrail configs delete <name>
```

For large config payloads, use `--input-file config.json` instead of inline
`--data`.

---

## Path A — `nemo guardrail check`

Evaluates a single message against a guardrail config server-side. The
`--guardrails '{"config_id": "..."}'` argument is the resolver — the
Guardrails service uses it to look up the config. `--model` is the **main**
LLM. Self-check rails call this model; content-safety rails call any task LLMs
the config references (for example, `models[type=content_safety]`), which must
also be reachable through IGW.

```bash
nemo guardrail check \
  --model default/<main-model> \
  --messages '[{"role": "user", "content": "Your message here"}]' \
  --guardrails '{"config_id": "default/<config-name>"}' \
  --max-tokens 256
```

---

## Path B — Guardrailed chat via IGW + VirtualModel

Configs are attached to a VirtualModel via a `MiddlewareCall`; chat completions
then route through IGW which executes the plugin's input/output rails.

### Step 1 — Provider + served models

Use the `inference` skill to set up a backend secret + provider and
discover served entity IDs:

```bash
nemo inference providers get <provider-name> --workspace default \
  --output-format json | jq -r '.served_models[].model_entity_id'
```

Reference the auto-discovered entity ID in the VirtualModel `--models` array,
and — if the rails config declares a task LLM via `models[]` — in
`models[].model` too. The reconciler overwrites manually-registered entries.

### Step 2 — Create the guardrail config

This example uses content-safety input and output rails. A content-safety rail
asks a dedicated safety classifier to assess user input or model output against
a safety taxonomy, then uses the parser's verdict to allow or block the turn.
The flows call the model declared in `models[]` via `$model=content_safety`.
The prompt includes the full unsafe-content category taxonomy used by the
content-safety tutorial.

```bash
nemo guardrail configs create content-safety \
  --workspace default \
  --description "NemoGuard content-safety input and output rails" \
  --data '{
    "models": [
      {
        "type": "content_safety",
        "engine": "nim",
        "model": "default/nvidia-llama-3-1-nemotron-safety-guard-8b-v3"
      }
    ],
    "rails": {
      "input": {"flows": ["content safety check input $model=content_safety"]},
      "output": {"flows": ["content safety check output $model=content_safety"]}
    },
    "prompts": [
      {
        "task": "content_safety_check_input $model=content_safety",
        "content": "Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a \"safe\" or \"unsafe\" rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nS1: Violence.\nS2: Sexual.\nS3: Criminal Planning/Confessions.\nS4: Guns and Illegal Weapons.\nS5: Controlled/Regulated Substances.\nS6: Suicide and Self Harm.\nS7: Sexual (minor).\nS8: Hate/Identity Hate.\nS9: PII/Privacy.\nS10: Harassment.\nS11: Threat.\nS12: Profanity.\nS13: Needs Caution.\nS14: Other.\nS15: Manipulation.\nS16: Fraud/Deception.\nS17: Malware.\nS18: High Risk Gov Decision Making.\nS19: Political/Misinformation/Conspiracy.\nS20: Copyright/Trademark/Plagiarism.\nS21: Unauthorized Advice.\nS22: Illegal Activity.\nS23: Immoral/Unethical.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\nuser: {{ user_input }}\n\n<END CONVERSATION>\n\nProvide your safety assessment for the above conversation in the below output JSON format:\n{\"User Safety\": safe or unsafe rating for the user message, \"Response Safety\": safe or unsafe rating for the agent response. Omit if no agent response present. \"Safety Categories\": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}\n\nDo not include anything other than the output JSON in your response.\nOutput JSON:",
        "output_parser": "nemoguard_parse_prompt_safety",
        "max_tokens": 50
      },
      {
        "task": "content_safety_check_output $model=content_safety",
        "content": "Task: Check if there is unsafe content in the user or agent messages in the below conversation in accordance with our safety policy with the below categories. Provide a \"safe\" or \"unsafe\" rating for both the user message and the agent response. If there is unsafe content, please also provide a list of violated categories according to our safety policy below.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nS1: Violence.\nS2: Sexual.\nS3: Criminal Planning/Confessions.\nS4: Guns and Illegal Weapons.\nS5: Controlled/Regulated Substances.\nS6: Suicide and Self Harm.\nS7: Sexual (minor).\nS8: Hate/Identity Hate.\nS9: PII/Privacy.\nS10: Harassment.\nS11: Threat.\nS12: Profanity.\nS13: Needs Caution.\nS14: Other.\nS15: Manipulation.\nS16: Fraud/Deception.\nS17: Malware.\nS18: High Risk Gov Decision Making.\nS19: Political/Misinformation/Conspiracy.\nS20: Copyright/Trademark/Plagiarism.\nS21: Unauthorized Advice.\nS22: Illegal Activity.\nS23: Immoral/Unethical.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\nuser: {{ user_input }}\n\nresponse: agent: {{ bot_response }}\n\n<END CONVERSATION>\n\nProvide your safety assessment for the above conversation in the below output JSON format:\n{\"User Safety\": safe or unsafe rating for the user message, \"Response Safety\": safe or unsafe rating for the agent response. Omit if no agent response present. \"Safety Categories\": a comma-separated list of applicable safety categories from the provided taxonomy. Omit if all safe.}\n\nDo not include anything other than the output JSON in your response.\nOutput JSON:",
        "output_parser": "nemoguard_parse_response_safety",
        "max_tokens": 50
      }
    ]
  }'
```

The `content_safety` task model must be reachable through IGW. Use the
auto-discovered entity ID from Step 1 and update `models[].model` if your
workspace or provider uses a different entity name.

### Step 3 — Create the VirtualModel with a guardrails MiddlewareCall

Three patterns, mirroring the input/output rail combinations.

**Input + output rails** — full content-safety coverage. Because the config
created above declares both `rails.input.flows` and `rails.output.flows`, the
call must appear in **both** lists; otherwise only the listed hook fires:

```bash
nemo inference virtual-models create vm-guarded-full \
  --workspace default \
  --models '[{"model":"default/<backend-model>","backend_format":"OPENAI_CHAT"}]' \
  --request-middleware '[{
    "name":"nemo-guardrails",
    "config_type":"guardrail_config",
    "config_id":"default/content-safety"
  }]' \
  --response-middleware '[{
    "name":"nemo-guardrails",
    "config_type":"guardrail_config",
    "config_id":"default/content-safety"
  }]'
```

**Input rails only** — block bad user prompts before they reach the backend:

```bash
nemo inference virtual-models create vm-guarded-input \
  --workspace default \
  --models '[{"model":"default/<backend-model>","backend_format":"OPENAI_CHAT"}]' \
  --request-middleware '[{
    "name":"nemo-guardrails",
    "config_type":"guardrail_config",
    "config_id":"default/content-safety"
  }]'
```

**Output rails only** — block bad bot responses:

```bash
nemo inference virtual-models create vm-guarded-output \
  --workspace default \
  --models '[{"model":"default/<backend-model>","backend_format":"OPENAI_CHAT"}]' \
  --response-middleware '[{
    "name":"nemo-guardrails",
    "config_type":"guardrail_config",
    "config_id":"default/content-safety"
  }]'
```

**Inline config** — no stored `guardrail_config` entity; the rails config goes
directly inside the `MiddlewareCall`:

```bash
nemo inference virtual-models create vm-guarded-inline \
  --workspace default \
  --models '[{"model":"default/<backend-model>","backend_format":"OPENAI_CHAT"}]' \
  --response-middleware '[{
    "name":"nemo-guardrails",
    "config_type":"guardrail_config",
    "config":{"models":[...],"rails":{...},"prompts":[...]}
  }]'
```

The `--models` array is required and uses the backend model entity ID, not the
VirtualModel name (the guardrails plugin doesn't route — the VirtualModel's
`default_model_entity` / `--models` decide the upstream backend).

### Step 4 — Make a guardrailed chat call

```bash
nemo inference gateway model post v1/chat/completions vm-guarded-full \
  --workspace default \
  --body '{
    "model":"default/<backend-model>",
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":256
  }'
```

`body["model"]` is the backend entity ID and acts as the rails' **main** LLM.
Content-safety flows call the task LLM from `models[type=content_safety]`;
self-check flows, when used, call the main LLM. The VirtualModel name comes
from the URL path.

For streaming, add `"stream": true` to the body — the plugin wraps the response
iterator and runs output rails per chunk if `rails.output.streaming.chunk_size`
is set in the config.

---

## MiddlewareCall JSON Reference

`MiddlewareCall` is defined in
[packages/nemo_platform_plugin/src/nemo_platform_plugin/inference_middleware.py](packages/nemo_platform_plugin/src/nemo_platform_plugin/inference_middleware.py).
For `nemo-guardrails`, the contract is:

```json
{
  "name": "nemo-guardrails",
  "config_type": "guardrail_config",
  "config_id": "<workspace>/<config-name>"
}
```

Or, with an inline rails config:

```json
{
  "name": "nemo-guardrails",
  "config_type": "guardrail_config",
  "config": {"models": [...], "rails": {...}, "prompts": [...]}
}
```

`config_type` is always the string `"guardrail_config"`. Use `config_id` for
stored entities (recommended — easier to update and reuse) and `config` for
ephemeral inline configs.

---

## Config JSON Structure

The **main** LLM is always the model from the chat request body (Path B) or
the `--model` flag (Path A) — both Path A and Path B ignore any
`models[type=main]` entry on the config and use the request's model instead.
Self-check rails (`self check input`/`self check output`) call the main LLM,
so they don't need a `models[]` entry at all. Add `models[]` only for
**task LLMs** (`content_safety`, `topic_control`, `jailbreak_detection`,
`embeddings`, `vision_rails`, etc.) that flows explicitly reference via `$model=<type>`.

### Content safety with a task LLM

This pattern uses a dedicated content-safety classifier (for example,
`default/nvidia-llama-3-1-nemotron-safety-guard-8b-v3`) to assess both user
input and model output against a safety taxonomy. The `content_safety` flows
reference the task LLM via `$model=content_safety`, so the `models[]` entry
**is** required here.

```json
{
  "models": [
    {
      "type": "content_safety",
      "engine": "nim",
      "model": "default/nvidia-llama-3-1-nemotron-safety-guard-8b-v3"
    }
  ],
  "rails": {
    "input":  {"flows": ["content safety check input $model=content_safety"]},
    "output": {"flows": ["content safety check output $model=content_safety"]}
  }
}
```

See [Step 2](#step-2--create-the-guardrail-config) for the complete runnable
config, including the full prompt taxonomy and output parsers.

### Self-check rails

Self-check rails are most useful for tests and lightweight custom policies.
For production safety, prefer content-safety rails with a dedicated classifier
and taxonomy. Self-check runs against the main/general chat model and expects
the model to answer `Yes` to block or `No` to allow. If the model returns
anything else (for example, reasoning text, a refusal, or a truncated answer),
the parser can default-deny and block the message. See
[Self-check example](#self-check-example) for a custom-policy example.

### Streaming output rails

```json
"output": {
  "flows": ["content safety check output $model=content_safety"],
  "streaming": {"enabled": true, "chunk_size": 200}
}
```

`streaming` is an `OutputRailsStreamingConfig` object, not a boolean. Other
fields: `enabled` (default `true`), `context_size` (default `50`),
`stream_first` (default `true`). When unset, the plugin runs output rails on
the assembled response only.

### Field reference

**models[]** (optional — only for task LLMs)
- `type` — task to use the model for. Examples: `"main"`, `"content_safety"`, `"topic_control"`,
  `"jailbreak_detection"`, `"embeddings"`, `"vision_rails"`. Entries with `type: "main"` are
  ignored — omit them.
- `engine` — free-form string (commonly `"nim"`); the schema is permissive.
- `model` — `<workspace>/<model-entity-name>`. Must be reachable through IGW;
  use auto-discovered entity IDs from
  `nemo inference providers get ... | jq '.served_models[].model_entity_id'`.

Flows reference task LLMs by appending `$model=<type>` (e.g.
`content safety check input $model=content_safety`). Self-check flows
(`self check input`/`self check output`) run against the main LLM and need
no `models[]` entry.

**rails**
- `rails.input.flows`: enables input rails (e.g.
  `["content safety check input $model=content_safety"]`)
- `rails.output.flows`: enables output rails (e.g.
  `["content safety check output $model=content_safety"]`)
- `rails.output.streaming`: optional `{"chunk_size": N}` for streaming output checks

**prompts[]**
- `task`: the task this prompt template is applied to. Common values:
  `"self_check_input"`, `"self_check_output"`,
  `"content_safety_check_input $model=content_safety"`,
  `"content_safety_check_output $model=content_safety"`. The `$model=<type>`
  suffix binds the prompt to a task LLM declared in `models[]`.
- `content`: prompt template. Use `{{ user_input }}` to embed the user
  message and `{{ bot_response }}` to embed the model's response
  (`{{ user_input }}` is available in both input and output rail prompts;
  `{{ bot_response }}` is only meaningful in output rail prompts). For
  self-check tasks, end with a "Yes or No" question — the model answers
  `Yes` to block, `No` to allow. Content-safety tasks instead emit a JSON
  verdict consumed by `output_parser`.
- `output_parser`: optional parser name. Use this when the model returns
  a safety verdict that must be converted to an allow/block decision.
  `is_content_safe` is the generic parser for simple `safe` / `unsafe` or
  `yes` / `no` outputs. For content-safety checks using NemoGuard models, use
  `"nemoguard_parse_prompt_safety"` for the input rail and
  `"nemoguard_parse_response_safety"` for the output rail.
  For content-safety checks using Nemotron reasoning models, use
  `"nemotron_reasoning_parse_prompt_safety"` for the input rail and
  `"nemotron_reasoning_parse_response_safety"` for the output rail.

**instructions[]** (optional)
- `type`: `"general"`
- `content`: system-level context for the conversation.

---

## Detecting Blocked Responses

When a rail triggers, the response content is exactly:

```
I'm sorry, I can't respond to that.
```

- **Path A (`nemo guardrail check`)**: look for `"status": "blocked"` in the
  output JSON.
- **Path B (chat completions through IGW)**: check
  `response.choices[0].message.content == "I'm sorry, I can't respond to that."`.

For streaming responses, the canned refusal arrives as a single non-streamed
chunk when an output rail blocks mid-stream.

---

## Self-check Example

Use self-check for tests or custom lightweight policies. For production safety,
prefer the content-safety workflow above. These prompts run against the main
chat model, so keep the prompt strict and prefer non-reasoning models for this
pattern unless you also raise the prompt's `max_tokens` budget enough for the
final `Yes` / `No` verdict.

```bash
nemo guardrail configs create custom-blocking \
  --description "Block fruit mentions in input, bread baking in output" \
  --data '{
    "rails":{
      "input":{"flows":["self check input"]},
      "output":{"flows":["self check output"]}
    },
    "prompts":[
      {"task":"self_check_input","content":"Check if the user message mentions any fruit (apple, banana, orange, grape, strawberry, mango, pear, peach, cherry, watermelon, lemon, lime, etc.).\n\nUser message: {{ user_input }}\n\nQuestion: Should the user message be blocked (Yes or No)?\nAnswer:"},
      {"task":"self_check_output","content":"Check if the bot response contains information about baking bread.\n\nBot response: {{ bot_response }}\n\nQuestion: Should the message be blocked (Yes or No)?\nAnswer:"}
    ],
    "instructions":[{"type":"general","content":"Below is a conversation between a user and a bot."}]
  }'
```

Then verify via `nemo guardrail check`, or attach to a VirtualModel as in
[Path B](#path-b--guardrailed-chat-via-igw--virtualmodel).

---

## Failure cases

### Plugin in only one of `--request-middleware` / `--response-middleware`

Each list dispatches its own hook independently
(`execute_request_middleware` ↔ `process_request`, `execute_response_middleware`
↔ `process_response` — see
[services/core/inference-gateway/src/nmp/core/inference_gateway/api/middleware_registry.py](services/core/inference-gateway/src/nmp/core/inference_gateway/api/middleware_registry.py)).

If your config has both `rails.input.flows` and `rails.output.flows` but you
attach the call to only one list, the unlisted side will silently no-op. Symptom:
prompts pass through unblocked, or responses pass through unblocked, even
though the config looks complete.

Fix: attach the same call to both `--request-middleware` and
`--response-middleware`.

### Putting `guardrails` field in the chat body expecting it to set the config

On the **IGW VirtualModel path**, the request body's `guardrails` dict is
strictly an options envelope. Supported fields are `options.log` and
`return_choice`. The rails config is resolved from the VirtualModel's
`MiddlewareCall`, not from the request body.

Unsupported fields are rejected with a `422` validation error. This includes
`guardrails.config`, `guardrails.config_id`, `guardrails.options.rails`,
`guardrails.options.llm_params`, `guardrails.options.llm_output`, and
`guardrails.options.output_vars`.

Symptom: you set `--body '{"guardrails":{"config_id":"default/foo"},...}'`
expecting `default/foo` to be applied, but the request fails validation.

Fix: attach the config to the VirtualModel via `--request-middleware` /
`--response-middleware` instead of trying to override per-request. Use
`guardrails.options.log` only for diagnostic log fields and `return_choice`
only for response formatting.

### Task LLM (or main model) not addressable

If `RailsConfig.models[].model` references an entity that isn't in
`served_models`, the rails will fail when they try to call that task LLM
through IGW. Same applies to the main LLM (the request body's model on Path
B, `--model` on Path A) — if it isn't reachable through IGW, self-check
flows fail. Both usually show up as a 404 or timeout from the rails
subsystem during the first request.

Fix: use auto-discovered entity IDs (from
`nemo inference providers get ... | jq '.served_models[].model_entity_id'`)
both for the request/`--model` and for any task LLMs in `models[]`. See
`inference` skill for the auto-discovery semantics.

### VirtualModel 404 after working briefly

The provider reconciler re-syncs `served_models` from upstream auto-discovery
every few seconds and drops manually-registered entries. See
`inference` skill's "Reconciler-induced 404" failure case — the same
fix applies (use auto-discovered entity IDs in `--models`).
