<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model metadata plugin

Fills in `metadata` on Model entities that a ModelProvider serves: display name, description, capabilities, input and output modalities, parameter count, context length, and maximum output tokens.

A background controller (`model-metadata`) checks every Model entity that lists a provider. It asks each configured source about the model's upstream id (the provider's `served_model_name`) and keeps, for each field, the value from the first source that has one. The result is written with `PATCH /v2/workspaces/{workspace}/models/{name}`.

## Sources

| Source | What it reads | Fields it usually fills |
| --- | --- | --- |
| `provider` | The provider's own `GET /v1/models`, through the inference gateway provider route, so the provider's credentials are used | OpenRouter and similar: name, description, modalities, context length, max output, tool calling. vLLM: context length |
| `ngc` | The NGC catalog entry behind a build.nvidia.com model (`/v2/endpoints/{org}/{name}/spec` and `/v2/endpoints/{org}/{name}`), only for providers on `integrate.api.nvidia.com`. The endpoint name comes from the model id, and an entry whose spec names a different model is ignored | Description, capabilities, modalities, parameter count, context length |

## Behavior

- Metadata with an empty `sources` list counts as set by hand, and the controller never overwrites it. Send `"metadata": null` in a `PATCH` to clear it and hand the model back to the controller.
- Fetched metadata is refreshed after `refresh_interval_seconds`. A model no source knows is retried after the same interval.
- When a source cannot answer (network error, rate limit, server error), the model is left as it is and retried on the next cycle.
- A model's providers are tried in order; the first one that yields metadata is used.
- Models without a provider, such as fileset-only checkpoints, are skipped.

## Configuration

Set under `platformConfig.model_metadata` in Helm, or with `NEMO_MODEL_METADATA_*` environment variables, for example `NEMO_MODEL_METADATA_SOURCES='["provider"]'`.

| Field | Default | Meaning |
| --- | --- | --- |
| `sources` | `["provider", "ngc"]` | Sources in priority order. Remove one to stop calling it, for example on air-gapped installs. |
| `interval_seconds` | `300` | Seconds between reconcile cycles |
| `refresh_interval_seconds` | `86400` | Seconds before metadata, or a lookup that found nothing, is fetched again |
| `request_timeout_seconds` | `15` | Timeout for each request |
| `ngc_url` | `https://api.ngc.nvidia.com` | NGC catalog API URL |
| `ngc_org` | `qc69jvmznzxy` | NGC org that publishes the build.nvidia.com endpoints |
