<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NGC plugin

Supplies model metadata for build.nvidia.com models. It registers the `ngc` source under the `nemo.models.metadata_sources` entry point, which the Models service's `model-metadata` controller uses for every ModelProvider on `integrate.api.nvidia.com`.

For each model it reads the public NGC endpoint spec and catalog entry, `GET /v2/endpoints/{org}/{name}/spec` and `GET /v2/endpoints/{org}/{name}`, without credentials. The endpoint name comes from the model id's last segment (dots kept, then turned into underscores, then dashes), and an endpoint whose spec names a different model is ignored. NGC answers 403 for retired or private endpoints, which counts as not found.

Fields it fills: description, capabilities (tool calling, reasoning, structured output), input and output modalities, parameter count, and context length.

## Configuration

Set under `platformConfig.ngc` in Helm, or with `NEMO_NGC_*` environment variables.

| Field | Default | Meaning |
| --- | --- | --- |
| `url` | `https://api.ngc.nvidia.com` | NGC catalog API URL |
| `org` | `qc69jvmznzxy` | NGC org that publishes the build.nvidia.com endpoints |

To stop calling NGC, uninstall the plugin or leave `ngc` out of `NEMO_PLUGIN_MODEL_METADATA_SOURCES_ALLOWLIST`.
