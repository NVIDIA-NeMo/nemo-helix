<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model Configs

`model_configs` is a YAML list that defines the model aliases the plugin resolves through NeMo Helix. `selected_models` can bind those aliases to Anonymizer library roles. The [NVIDIA NeMo Anonymizer library docs](https://github.com/NVIDIA-NeMo/Anonymizer/tree/main/docs) and library skills own the full role list and model-selection semantics.

## When is `model_configs` required?

| Surface                  | Status                  | `model_configs` required?                                                                                  |
|--------------------------|-------------------------|------------------------------------------------------------------------------------------------------------|
| `sdk.anonymizer.preview` | Available (SDK/service) | **Yes** — needed so requests route through the NeMo Helix Inference Gateway instead of build.nvidia.com directly. |
| `nemo anonymizer preview` | Available (plugin svc) | **Yes** — same service path as SDK preview.                                                                |
| `nemo anonymizer run`    | Available (Jobs worker) | **Yes** — the job routes through the NeMo Helix Inference Gateway.                                      |
| Strategy is `Substitute` | n/a                     | Provide a `replacement_generator`-capable alias.                                                           |
| Mode is `rewrite`        | n/a                     | Provide aliases for the Anonymizer library rewrite roles.                                                  |

`selected_models` is **only** legal alongside `model_configs` — passing overrides without a pool raises `selected_models requires model_configs so aliases can be resolved.`

## `ModelConfig` shape

```yaml
model_configs:
  - alias: gliner-pii-detector       # name your role bindings will reference
    provider: nvidia-build           # name of a NeMo Helix inference provider in the target workspace
    model: nvidia/gliner-pii         # provider-specific model id
    # inference_parameters:          # optional, provider-dependent (temperature, max_tokens, ...)
    #   temperature: 0.0
```

The `provider` field is resolved at request time against NeMo Helix. The string format is `provider-name` or `workspace/provider-name`. For provider discovery or creation, refer the user to the platform inference/model-provider docs or the relevant inference/model skill; keep this skill's executable commands limited to Anonymizer workflows.

## `selected_models` (role bindings)

`selected_models` is a partial mapping that overrides the Anonymizer library's bundled defaults. Each section is optional; omitted sections fall back to defaults. Use Anonymizer library role names exactly.

```yaml
selected_models:
  detection:
    entity_detector: gliner-pii-detector
    entity_validator: gpt-oss-120b                    # scalar alias
    # entity_validator: [gpt-oss-120b, nemotron-30b]  # list also accepted
  replace:
    replacement_generator: gpt-oss-120b
  rewrite:
    rewriter: gpt-oss-120b
    evaluator: nemotron-30b-thinking
    repairer: gpt-oss-120b
    judge: nemotron-30b-thinking
```

Only emit a section if you actually want to override its defaults — overrides are merged on top of the bundled YAML defaults at parse time.

## Common patterns

**Default model pool** (`sdk.anonymizer.preview`, `nemo anonymizer preview`, `nemo anonymizer run`) — provide the aliases used by the Anonymizer library defaults:

```yaml
model_configs:
  - {alias: gliner-pii-detector, provider: nvidia-build, model: nvidia/gliner-pii}
  - {alias: gpt-oss-120b, provider: nvidia-build, model: openai/gpt-oss-120b}
  - {alias: nemotron-30b-thinking, provider: nvidia-build, model: nvidia/nemotron-3-nano-30b-a3b}
```

`Redact`/`Annotate`/`Hash` don't require a replacement model, but the default detection selection references the detection aliases above. `Substitute` and `rewrite` can use the same pool unless the user wants to pin different aliases through `selected_models`.

**Picking a specific validator** — keep library defaults but pin the validator:

```yaml
selected_models:
  detection:
    entity_validator: gpt-oss-120b
```
