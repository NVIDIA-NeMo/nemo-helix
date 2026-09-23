<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nhx-platform (legacy task entrypoint)

> **Status: legacy.** This package is kept for backwards compatibility with
> existing task container images and is on track to be dropped. New code
> should not depend on it.

## What this is

A thin shim that exposes a single console script, `nemo-helix`, with one
supported subcommand:

```bash
nemo-helix run task --task <python.module>
```

It executes the named Python module via `runpy` with optional `--env` and
`--config` (the latter is forwarded as the `NEMO_JOB_STEP_CONFIG` environment
variable). Anything else (`run --services …`, `run --config …`,
`run --quickstart`, etc.) is intentionally rejected with a non-zero exit code
that points callers at `nemo services run`.

## Why it still exists

A handful of task container images and seed jobs invoke `nemo-helix run task`
as their entrypoint:

- `nhx-customizer-tasks` — shared CPU task image (`nhx.customization_common.tasks.file_io` /
  `model_entity`, plus `model_spec` and the LoRA sidecar). Local compose example:
  `services/automodel/src/nhx/automodel/tasks/docker/docker-compose.yaml`.
- `services/platform-seed` — recommended invocation in its README is
  `nemo-helix run task --task nhx.platform_seed`.

The package is wired into container builds via the `nhx-task-runtime`
dependency group in the root `pyproject.toml`.

## What replaced it

All actual platform/service startup has moved to `nemo services run`,
implemented by `packages/nhx_platform_runner/`. See `CONTRIBUTING.md` for the
current local-dev workflow.

## When to delete this package

Once nothing invokes `nemo-helix run task` anymore — i.e. once the task
images and `platform-seed` switch to invoking the target module directly
(`python -m nhx.platform_seed`, etc.) or move to a different launcher — this
whole directory can be removed along with:

- the `nhx-task-runtime` dependency group in the root `pyproject.toml`,
- the `"packages/nhx_platform"` entry in `[tool.uv.workspace] members`,
- the `[tool.hatch.build.targets.wheel] packages = ["packages/nhx_platform/"]`
  line at the bottom of the root `pyproject.toml`.

## Layout

```text
config/         # local-dev platform config (NHX_CONFIG_FILE_PATH default)
src/nhx/platform/main.py
tests/test_main.py
```

The `config/` files (`local.yaml`, `local.env`) are not Python — they are the
default config consumed by `nemo services run` during local development and
referenced from several Makefiles and run scripts in the repo.

`local.env` sets SQLite for the entity store (`~/.local/share/nemo/nhx-platform.db`)
so no PostgreSQL is required. Source it before starting services:

```bash
set -a && source packages/nhx_platform/config/local.env && set +a
uv run nemo services run --host 127.0.0.1 --port 8080
```
