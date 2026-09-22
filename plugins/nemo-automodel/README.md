<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-automodel-plugin

Automodel training contributor under `/apis/customization/v2/workspaces/{workspace}/automodel/`.

Requires **`nemo-customizer-plugin`** at runtime (router + `client.customization` SDK) and **`nhx-automodel`** (compiler/tasks). The Automodel plugin does not declare a pyproject dependency on the customizer plugin — install both via root `enabled-plugins`:

```bash
uv sync --group enabled-plugins
```

## CLI

Verbs are mounted directly on the contributor (no `jobs` subgroup):

```bash
nemo customization automodel explain
nemo customization automodel submit path/to/job.json
nemo customization automodel submit path/to/job.json -w acme-corp
nemo customization automodel submit path/to/job.json --cluster my-cluster
```

Other customization backends may still use `nemo customization <backend> jobs submit ...`.

Job JSON uses the simplified `AutomodelJobInput` schema (see `nemo_automodel_plugin/schema.py`). Submit posts to `/apis/customization/v2/workspaces/{workspace}/automodel/jobs`.

Optional `integrations` (W&B / MLflow) use the shared `IntegrationsSpec` from `nemo_helix_plugin.integrations`. Example: `plugins/nemo-automodel/tests/fixtures/integrations_wandb_mlflow.json`. Field reference: customizer skill `references/hyperparameters.md` § **Integrations (automodel + unsloth)**.
