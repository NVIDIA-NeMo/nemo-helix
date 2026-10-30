<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Customization contributor guide

Register a training backend under **`nemo.customization.contributors`** (not `nemo.services`).

## Contract

Implement `CustomizationContributor`:

- `name` — must match the entry-point key (e.g. `automodel`)
- `get_routers()` — `RouterSpec` list with a **unique** prefix under `v2/workspaces/{workspace}/<backend>/`
- `get_cli()` — optional `typer.Typer` mounted at `nemo customization <name>`
- `get_cli_summary()` — optional `CustomizationCLISummary` added to `nemo customization --help`; the router lists discovered backends in name order and never names one itself
- `get_sdk_resources()` — optional sync/async resource classes for `client.customization.<name>`; the Customizer hub constructs them with a typed Customizer SDK context that contains the typed Customizer client, typed Jobs client, and active workspace (do not register a separate `nemo.sdk` entry point; the Customizer hub owns `nemo.sdk` → `customization` and composes backends)

## Help text

A user should be able to choose a backend from `nemo customization --help` alone, so
each backend provides two layers of help:

- **Overview** — `get_cli_summary()` returns a `CustomizationCLISummary` with the
  fields `trains`, `runs_on`, `job_json`, `use_when` and `command`. Keep each field to
  one or two short sentences. `render()` wraps them to fit the 80-column console the
  router's help is printed through.
- **Detail** — the `help` on the `typer.Typer` returned by `get_cli()`, and the help on
  its `submit` command. This is where the job JSON fields, runtime constraints, and
  pointers to a more suitable backend belong.

Keep the two layers consistent with the other backends: the overview fields answer the
same questions in the same order, and the detailed help follows the same section order.

## pyproject.toml

```toml
[project.entry-points."nemo.customization.contributors"]
automodel = "nemo_automodel_plugin.contributor:AutomodelContributor"
```

## Jobs

Use `add_job_routes(YourJob, service_name="customization", ...)` so Jobs records use `source=customization`.

## Skills

The plugin ships two skills under `src/nemo_customizer/skills/`: `nemo-customizer` (submit and
monitor training jobs) and `nemo-gym-environments` (build GRPO reward environments). They are
installed into coding agents without the repository, so every path they reference must resolve
inside the installed skill. Repository paths belong here, not in the skills.

When a schema or validator changes, update the skill text from these sources. `nemo customization
<plugin> explain` prints the same schemas from an installed build.

| Topic | Source |
|---|---|
| Submit schemas | `plugins/nemo-automodel/src/nemo_automodel_plugin/schema.py`, `plugins/nemo-unsloth/src/nemo_unsloth_plugin/schema.py`, `plugins/nemo-rl/src/nemo_rl_plugin/schema.py` |
| automodel schema → compiler mapping | `services/automodel/src/nhx/automodel/adapter.py`, `services/automodel/src/nhx/automodel/api/v2/jobs/schemas.py` |
| automodel dataset schema detection | `DatasetSchema` in `services/automodel/` dataset preparation |
| unsloth canonical schema and training driver | `services/unsloth/src/nhx/unsloth/schemas.py`, `services/unsloth/src/nhx/unsloth/tasks/training/backends/unsloth_sft.py` |
| rl canonical schema and DPO config builder | `services/rl/src/nhx/rl/schemas.py`, `services/rl/src/nhx/rl/tasks/training/backends/nemo_rl/dpo_config.py` |
| Gym environment manifest and dataset rows | `services/rl/src/nhx/rl/schemas/environment.py` |
| Environment validation, converter, `build_policy_model_yaml` | `services/rl/src/nhx/rl/tasks/environment/validate.py`, `convert.py`, `package.py`, `__main__.py` |
| GRPO example scripts | `scripts/grpo-examples/` |

`tests/test_skill_job_examples.py` runs every job JSON example in both skills, fragments such as
`"batch": {...}` included, through the backends' `--job-json` validators. When a schema change
fails it, update the example in the skill.

The JSON files under `plugins/nemo-*/tests/fixtures/` are smoke-test inputs. They set whatever
makes a test cheap (`max_steps: 20`, a 0.6B model) and cover one path, so never take field sets,
defaults or recommended values from them, and keep the skills from pointing agents at them.
