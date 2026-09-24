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

## Verification

Run unit tests in the workspace:

```bash
uv run --frozen pytest plugins/nemo-switchyard/tests -v
```

Run the native wheel integration tests in an isolated environment:

```bash
plugins/nemo-switchyard/scripts/run_native_tests.sh
```

Run all native routing types through a live local Inference Gateway and write
a shareable Markdown report:

```bash
uv run --frozen python plugins/nemo-switchyard/scripts/smoke_native_routing.py \
  --nemo .venv/bin/nemo \
  --output plugins/nemo-switchyard/scripts/switchyard-routing-smoke-report.md
```
