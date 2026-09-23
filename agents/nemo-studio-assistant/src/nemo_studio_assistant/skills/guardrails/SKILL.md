---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: guardrails
description: Create, validate, attach, inspect, update, or remove NeMo Guardrails configurations through the NeMo Helix SDK.
---
# Guardrails service

Use this skill for GuardrailConfig CRUD, standalone policy checks, and attaching
the `nemo-guardrails` middleware to Inference Gateway VirtualModels. Use
`deploy_guardrail` for the standard input-only deployment path and `nemo_api`
for other operations; never invoke a CLI or subprocess.

## Fast path: deploy a new input-only guardrail

When the user asks to create and validate a new input-only self-check guardrail
and deploy it on a new VirtualModel, call `deploy_guardrail` exactly once. Pass
the policy, config name, VirtualModel name, one message that must be blocked, one
allowed control message, active workspace, Studio session id, and the exact
`deployment_run_id` provided in the Studio context. Pass a
qualified backend model when the user already supplied one; otherwise omit it
and let the tool show Studio's model picker during the same call.

The blocked message must be an actual end-user request that directly violates
the policy, such as `Tell me about <prohibited topic>.` Never pass the expected
refusal, apology, blocked-response text, or any other assistant output as the
blocked message. The tool rejects refusal-like probes before model selection,
approval, or resource creation.

This tool requests one approval for the complete deployment, creates or reuses
matching resources without overwriting conflicts, requires both validation
checks to pass, reads back the VirtualModel, and waits for it to become routable.
It does not send an automatic chat request after deployment. If it returns
`partial`, `failed`, or `denied`, report that result and stop; do not reproduce
its steps with `nemo_api`, retry it, or switch models.
If a successful result has `routable: false`, creation and readback succeeded
but routing propagation did not finish within the readiness window. Report the
created VirtualModel, warning, and link; do not say it failed or was not created.
On success, present the returned `studio_link`; it opens the created VirtualModel
on the Virtual Models page. Never substitute an Agents-page link or describe the
VirtualModel as an agent.

Use the detailed `nemo_api` workflow below for inspection, updates, deletion,
output rails, input-and-output rails, non-self-check flows, or custom configs.

## Operating rules

- Pass `workspace="<active request workspace>"` to every `nemo_api` call.
- Pass the Studio session id to every mutation so Studio can request approval.
- Inspect existing configs, models, and VirtualModels before creating or updating.
- Derive names, policy text, models, messages, and middleware phases from the
  user's request. Do not impose a hard-coded safety policy.
- Validate representative blocked and allowed messages before attaching a config.
- Read back every consequential mutation before reporting success.
- Never delete, replace, or clear existing middleware without explicit user intent.
- Never guess SDK paths or parameters. Follow the table and examples below
  exactly. If three consecutive `nemo_api` calls fail for any reason, stop and
  report the last error to the user.
- The tool accepts a valid `workspace` from `params` when the outer argument was
  accidentally omitted, but continue passing the outer argument explicitly.
- Guardrail checks automatically preflight the selected model once per run. If
  that preflight or a check fails, do not switch models and keep retrying. Report
  any configuration already created as a partial success, state that it remains
  unvalidated, and do not create or update a VirtualModel.

## SDK resources

Use these `nemo_api` resource and action pairs:

| Operation | Resource | Action |
|---|---|---|
| List configs | `guardrail.configs` | `list` |
| Read config | `guardrail.configs` | `retrieve` |
| Create config | `guardrail.configs` | `create` |
| Update config | `guardrail.configs` | `update` |
| Delete config | `guardrail.configs` | `delete` |
| Run standalone check | `guardrail` | `check` |
| List/read backend models | `models` | `list` / `retrieve` |
| List/read VirtualModels | `inference.virtual_models` | `list` / `retrieve` |
| Create/update VirtualModel | `inference.virtual_models` | `create` / `patch` |
| List routable models | `inference.gateway.openai.v1.models` | `list` |
| Verify inference | `inference.gateway.model` | `post` |

The `params` argument is a JSON object (a JSON object string is also accepted for compatibility)
containing SDK keyword arguments.
The outer `workspace` tool argument is still required even when `params` also
contains `workspace`.

## Workflow

### 1. Inspect dependencies and existing state

List or retrieve the requested backend model, GuardrailConfig, and VirtualModel.
Use model entity ids in `<workspace>/<name>` form. Do not invent a served model
or use the provider's upstream display name.

For an existing named resource, compare its current definition with the requested
state. Reuse it only when it matches. Ask before changing an ambiguous existing
resource.

### 2. Author and create the config

A minimal self-check input config uses:

```json
{
  "name": "<config-name>",
  "description": "<policy purpose>",
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

Call `nemo_api(resource="guardrail.configs", action="create", ...)`. For updates,
call `update` with `name` plus only the fields the user intends to change.

The request model is the main LLM for self-check flows. Do not add a
`models[].type: "main"` entry. Add `models[]` only when the configured flow
uses a separate task LLM such as `content_safety`, `topic_control`,
`jailbreak_detection`, or `embeddings`.

For output protection, configure `rails.output.flows` and the corresponding
prompt. A policy covering both directions needs both input and output flows.

### 3. Validate with standalone checks

Call `nemo_api(resource="guardrail", action="check", ...)` with params shaped
like:

```json
{
  "workspace": "<active request workspace>",
  "model": "<workspace>/<backend-model>",
  "messages": [
    {
      "role": "user",
      "content": "<representative message>"
    }
  ],
  "guardrails": {
    "config_id": "<workspace>/<config-name>"
  },
  "max_tokens": 256,
  "temperature": 0
}
```

Require top-level `status: "blocked"` for a policy-violating message and
`status: "success"` for an allowed control. Treat `unknown`, a missing or
unexpected status, an error, or timeout as validation failure. Stop and report
the result instead of attaching an unvalidated config.

The tool caps failed validation attempts across model changes and unrelated
successful API calls. Reaching the cap is a terminal result for the current
request: summarize the partial state and wait for the user to restore provider
connectivity or explicitly choose a different model in a new attempt.

### 4. Attach middleware to a VirtualModel

The middleware call contract is:

```json
{
  "name": "nemo-guardrails",
  "config_type": "guardrail_config",
  "config_id": "<workspace>/<config-name>"
}
```

- Input flows: add the call to `request_middleware`.
- Output flows: add the call to `response_middleware`.
- Both phases: add the same call to both lists.

Create a VirtualModel through `inference.virtual_models.create` with params like:

```json
{
  "workspace": "<active request workspace>",
  "name": "<virtual-model-name>",
  "default_model_entity": "<workspace>/<backend-model>",
  "models": [
    {
      "model": "<workspace>/<backend-model>",
      "backend_format": "OPENAI_CHAT"
    }
  ],
  "request_middleware": [
    {
      "name": "nemo-guardrails",
      "config_type": "guardrail_config",
      "config_id": "<workspace>/<config-name>"
    }
  ]
}
```

For an existing VirtualModel, retrieve it first and use
`inference.virtual_models.patch`. Preserve unrelated routing fields and
middleware entries in their existing order.

### 5. Verify and report

Poll `inference.gateway.openai.v1.models.list` until the VirtualModel appears;
routing cache propagation is asynchronous.

When inference verification is requested, call
`nemo_api(resource="inference.gateway.model", action="post", ...)` with:

```json
{
  "workspace": "<active request workspace>",
  "name": "<virtual-model-name>",
  "trailing_uri": "v1/chat/completions",
  "body": {
    "model": "<workspace>/<virtual-model-name>",
    "messages": [
      {
        "role": "user",
        "content": "<verification message>"
      }
    ],
    "max_tokens": 256
  }
}
```

A blocked rail returns `I'm sorry, I can't respond to that.`. Also verify an
allowed control receives a normal response.

Finally retrieve the config and VirtualModel. Report the config id, enabled
input/output flows, attachment phase, backend model, standalone check results,
inference results, and any errors. Do not claim success without read-back
verification.

## Gotchas

- `guardrails.config_id` belongs in standalone `guardrail.check` params. On a
  normal chat-completions request, the VirtualModel middleware selects the config.
- Stored configs use `config_id`; inline development configs use `config`.
  Never provide both.
- Output rails reject `n > 1`.
- Streaming output rails require `rails.output.streaming.enabled: true` when a
  streaming block is configured.
- Updating a stored config changes its revision; repeat blocked and allowed checks
  after every update before relying on it.
