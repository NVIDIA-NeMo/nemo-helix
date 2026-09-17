<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Customization contributor guide

Register a training backend under **`nemo.customization.contributors`** (not `nemo.services`).

## Contract

Implement `CustomizationContributor`:

- `name` — must match the entry-point key (e.g. `automodel`)
- `get_routers()` — `RouterSpec` list with a **unique** prefix under `v2/workspaces/{workspace}/<backend>/`
- `get_cli()` — optional `typer.Typer` mounted at `nemo customization <name>`
- `get_cli_summary()` — optional `CustomizationCLISummary` folded into `nemo customization --help`; the router lists discovered backends in name order and never names one itself
- `get_sdk_resources()` — optional sync/async resource classes for `client.customization.<name>`; the Customizer hub constructs them with a typed Customizer SDK context that contains the typed Customizer client, typed Jobs client, and active workspace (do not register a separate `nemo.sdk` entry point; the Customizer hub owns `nemo.sdk` → `customization` and composes backends)

## Help text

`nemo customization --help` has to let someone pick a backend without reading any
docs, so each backend owns two layers of help:

- **Overview** — `get_cli_summary()` returns a `CustomizationCLISummary` with
  `trains`, `runs_on`, `job_json`, `pick_when` and `command`. Keep each field to one
  or two short sentences. The router prints it through an 80-column console, so lines
  longer than that re-wrap and lose their indent.
- **Detail** — the `help` on the `typer.Typer` from `get_cli()`, and the `submit`
  command's own help. This is where the job JSON fields, runtime constraints and
  "use another backend instead" advice belong.

## pyproject.toml

```toml
[project.entry-points."nemo.customization.contributors"]
automodel = "nemo_automodel_plugin.contributor:AutomodelContributor"
```

## Jobs

Use `add_job_routes(YourJob, service_name="customization", ...)` so Jobs records use `source=customization`.
