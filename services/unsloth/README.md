<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nhx-unsloth

Task package for **container-submit Unsloth SFT** under the NeMo Helix customization router.

This package owns the heavy code that runs *inside* the platform's GPU containers:

- **Canonical schemas** (`nhx.unsloth.schemas`) — `UnslothJobOutput` and shared sub-shapes consumed by both compile-time and runtime code.
- **Training driver** (`nhx.unsloth.tasks.training.backends.unsloth_sft.train_sft`) — runs SFT inside the training container's baked venv (`unsloth` + `torch` + `transformers` + `trl` + `peft` + `bitsandbytes`). Heavy imports are localized to the function body so the parent process can import this module without dragging in the ML stack.
- **Container entrypoints**:
  - `nhx.customization_common.tasks.file_io` / `model_entity` — shared CPU steps (run from `nhx-customizer-tasks`; compiler passes `--service-source unsloth --service-name unsloth`).
  - `nhx.unsloth.tasks.training` — runs `train_sft` against the paths the file_io step populated (`nhx-unsloth-training` image).
- **Compile glue** (`nhx.unsloth.compile.platform_job_config_compiler`) — turns a canonical `UnslothJobOutput` into a 4-step `PlatformJobSpec`. Invoked by the plugin's `UnslothJob.compile`.

The thin contributor wrapper that registers Unsloth with the customization hub lives in `plugins/nemo-unsloth/`. That plugin owns submitter-facing schema (`UnslothJobInput`), the `UnslothContributor`, the `UnslothJob` lifecycle (`to_spec` + `compile`), the SDK shapes, and CLI overrides (`submit` reshaped, `run` disabled).

## Layout

```
services/unsloth/
├── pyproject.toml            # nhx-unsloth + [integrations]/[unsloth] extras for container images
├── README.md                 # this file
├── docker/                   # Dockerfile for nhx-unsloth-training
└── src/nhx/unsloth/
    ├── schemas.py            # canonical UnslothJobOutput + sub-shapes
    ├── compile.py            # public compile entry
    ├── config.py             # NHX_UNSLOTH_* env-var configuration
    ├── images.py             # image-ref resolution (nhx-unsloth-training etc)
    ├── download.py           # fileset download helper (sync)
    ├── upload.py             # fileset upload helper (sync)
    ├── model_entity.py       # output entity / adapter creation
    ├── platform_client.py    # async helpers for to_spec validation
    ├── app/
    │   ├── constants.py
    │   └── jobs/
    │       ├── compiler.py             # 4-step PlatformJobSpec builder
    │       ├── context.py              # NHXJobContext (env-var-driven)
    │       ├── file_io/schemas.py      # FileIOTaskConfig
    │       ├── model_entity/schemas.py # ModelEntityTaskConfig
    │       └── training/
    │           ├── compiler.py         # GPU training PlatformJobStep
    │           └── schemas.py          # TrainingStepConfig
    └── tasks/
        └── training/
            ├── __main__.py         (entrypoint: reads step config, calls train_sft)
            └── backends/
                └── unsloth_sft.py  (train_sft)
```

CPU `file_io` / `model_entity` runners live in `packages/nhx_customization_common`
(`nhx.customization_common.tasks.*`) and ship in the shared `nhx-customizer-tasks`
image — not under `services/unsloth/src/nhx/unsloth/tasks/`.

## Why a service package, not just a plugin module?

Two reasons:

1. **Container-process boundary.** The training driver runs inside `nhx-unsloth-training`. CPU tasks run from `nhx-customizer-tasks`. The plugin (compile-time) and containers (runtime) share schemas (`UnslothJobOutput`, `FileIOTaskConfig`, `ModelEntityTaskConfig`, `TrainingStepConfig`) via `nhx-customization-common` and this package.
2. **Image isolation.** The plugin process stays lightweight (no `unsloth` / `torch`). Heavy ML deps are installed only inside `nhx-unsloth-training` (see `docker/unsloth/README.md`). CPU steps use the lighter `nhx-customizer-tasks` image.

## Status

Container submit is the **only** supported execution path. The GPU image (`docker/Dockerfile.nhx-unsloth-training`) installs the ML stack via the canonical `uv pip install unsloth --torch-backend=auto`, then layers the platform glue on top with the `[integrations]` extra for W&B/MLflow. The `[unsloth]` extra in `pyproject.toml` is now a thin alias for `unsloth[huggingface]` for any caller that wants to install the same ML stack outside the image.

See `docker/README.md` for build / push / GPU smoke-test instructions.
