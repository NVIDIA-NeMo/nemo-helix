<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Data Designer Plugin

A NeMo Helix plugin that brings Data Designer into the platform.

## Validate a Config

`nemo data-designer validate` checks whether a Data Designer config is fit to be executed on the platform.

```bash
nemo data-designer validate config.yaml
```

JSON output (`--output json`) emits a structured `ValidationReport` for CI / automation use.

### Programmatic use

The same logic is exposed on the SDK via `DataDesignerResource.validate(config_builder, *, workspace=None)` and its async sibling. Both return a `ValidationReport` from `nemo_data_designer_plugin.sdk.validation`.

## Nemotron Personas Filesets

Data Designer workloads that include `PersonSampler` columns require Nemotron Personas filesets to exist in the `system` workspace for each requested locale. These filesets can be created using the CLI.

Use an existing NGC API key secret:

```bash
nemo data-designer personas make-fileset \
  --locale en_US \
  --api-key-secret system/ngc-api-key
```

Create a new secret from an environment variable, then bind the fileset to it:

```bash
nemo data-designer personas make-fileset \
  --locale en_US \
  --api-key-secret system/my-ngc-key \
  --api-key-env-var NGC_API_KEY
```
