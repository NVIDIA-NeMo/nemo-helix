<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Switchyard Inference Middleware Plugin

Request-routing middleware backed by
[Switchyard 0.3.0](https://github.com/NVIDIA-NeMo/Switchyard).

The plugin distribution is `nemo-switchyard-plugin`; VirtualModels reference
its `nemo-switchyard` middleware entry point. The API image installs the
separate `nemo-switchyard==0.3.0` distribution that provides
`switchyard_rust`.

## Supported requests and algorithms

The middleware supports OpenAI Chat Completions requests and runs only in
`request_middleware`. Protocol translation and response middleware are not
supported.

## Upgrading from the legacy middleware

Before upgrading an existing Switchyard VirtualModel:

1. Remove `nemo-switchyard` entries from `response_middleware`.
2. Remove middleware calls whose `config_type` is `translate`.
3. Ensure clients and every routed backend use the OpenAI Chat Completions
   request/response shape.
4. Update or recreate the VirtualModel with one of the supported request
   middleware configurations below.

Perform the VirtualModel migration together with the plugin upgrade. Legacy
translation and response-middleware configurations are rejected rather than
silently ignored.

| `config_type` | Purpose | Required configuration |
| --- | --- | --- |
| `random_routing` | Weighted choice between two models | `strong`, `weak`, `strong_probability` |
| `stage_router` | Route between capable and efficient stages | `confidence_threshold`, `models.capable`, `models.efficient` |
| `llm_classifier` | Capability classification with a judge model | `base_threshold`, `models.judge`, `models.capable`, `models.efficient` |

`llm_classifier` supports capability mode only.

## Random routing

The existing random-routing JSON shape is preserved:

```json
{
  "request_middleware": [
    {
      "name": "nemo-switchyard",
      "config_type": "random_routing",
      "config": {
        "strong": {"model": "workspace/model-a"},
        "weak": {"model": "workspace/model-b"},
        "strong_probability": 0.5,
        "rng_seed": 7
      }
    }
  ]
}
```

## Stage routing

```json
{
  "name": "nemo-switchyard",
  "config_type": "stage_router",
  "config": {
    "picker": "efficient_first",
    "confidence_threshold": 0.5,
    "models": {
      "capable": ["workspace/model-a"],
      "efficient": ["workspace/model-b"]
    }
  }
}
```

## Capability classifier

```json
{
  "name": "nemo-switchyard",
  "config_type": "llm_classifier",
  "config": {
    "mode": "capability",
    "base_threshold": 0.5,
    "models": {
      "judge": ["workspace/judge"],
      "capable": ["workspace/model-a"],
      "efficient": ["workspace/model-b"]
    }
  }
}
```

Judge calls are sent directly to the configured model provider with provider
credentials from the Inference Gateway model cache. Caller authorization
headers are never forwarded.

## Optimization strategy

The plugin also ships the `switchyard` strategy for `nemo agents optimize`.
Given a platform agent, a list of acceptable models and a list of routing
strategies, the job creates one VirtualModel per (model pair × routing
strategy) and saves a copy of the agent's `nemo-agents-spec-v1` config per
combination, pointed at that VirtualModel, to the job's results. The stored
agent is never modified and no VirtualModel is deleted.

`models` is ordered from most capable to most efficient: every pair `(i < j)`
routes with `models[i]` as the capable/strong model and `models[j]` as the
efficient/weak one. VirtualModels are named
`<agent>-<routing-strategy>-<digest>` in the submission workspace, where
`<digest>` is a short hash of the pair's models and middleware config. Re-running
the same request reuses the same VirtualModels; changing the model list or a
routing setting creates new ones alongside the old. An existing VirtualModel
with a matching name is reused only when its models and middleware config match
the request; otherwise the run fails and asks you to delete it before re-running.

| Field | Default | Meaning |
| --- | --- | --- |
| `agent` | required | Platform agent to route, `name` or `workspace/name`. |
| `models` | required | At least two model entity refs, most capable first. |
| `routing_strategies` | `["random_routing"]` | Any of `random_routing`, `stage_router`, `llm_classifier`. |
| `strong_probability` | `0.5` | `random_routing`: share of requests sent to the capable model. |
| `confidence_threshold` | `0.5` | `stage_router` confidence threshold. |
| `base_threshold` | `0.5` | `llm_classifier` capability threshold. |
| `judge_model` | capable model of the pair | `llm_classifier` judge. |

```bash
uv run nemo agents optimize run-strategy --strategy switchyard --agent my-agent \
  --spec '{"models": ["default/model-a", "default/model-b"], "routing_strategies": ["random_routing", "stage_router"]}'

# The results download as a tarball holding one agent-<virtual-model>.yaml per combination
# plus switchyard-result.json, which maps each VirtualModel to its models and middleware config.
uv run nemo jobs results download switchyard --job <job-name> -o switchyard.tar.gz
mkdir -p switchyard-results && tar -xzf switchyard.tar.gz -C switchyard-results --strip-components=1
# Pick a VirtualModel name from switchyard-result.json, e.g. my-agent-random-routing-3f2a9c1e.
uv run nemo agents create --name my-agent-random-routing-3f2a9c1e \
  --agent-config ./switchyard-results/agent-my-agent-random-routing-3f2a9c1e.yaml
```

Each rewritten config points its `models.default` and harness model blocks at
`<workspace>/<virtual-model>` with no `base_url`, so the platform binds the
Inference Gateway URL and credential when the agent runs.

## Verification

```bash
uv run --frozen pytest plugins/nemo-switchyard/tests -v
```
