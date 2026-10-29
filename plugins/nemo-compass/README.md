<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Compass plugin for Helix

NeMo Helix plugin for analyzing agent telemetry and persisting actionable insights.

Analysis uses [NeMo Compass](https://github.com/NVIDIA-NeMo/nemo-compass).
NeMo Helix supplies authenticated trace and model access and stores the resulting insights.

## Prerequisites

Use a running NeMo Helix with Intake, Compass, Jobs, and a configured
`agents.execute` runtime. The target agent must already have traces in Intake.
Run `nemo setup` to select default and fast Platform models, or supply
`--default-model` and `--fast-model` when submitting analysis.

For Compass analysis, use **NVIDIA Nemotron 3.5 Super VL** instead of Nemotron
3.5 Lightning 30B, which has known tool-calling failures. Selecting Nemotron 3.5
Lightning 30B produces a nonblocking warning; Helix keeps your selected model. The CLI and Studio recognize
this model's full name in model references, including normalized Helix names. Jobs also check the resolved served-model names,
so aliases are checked at runtime, with warnings in job logs and completed analysis reports.

## Install from the monorepo

```bash
uv sync
```

The plugin is installed by default through the root workspace's `enabled-plugins` group.

## Install from a package index

```bash
uv tool install --python 3.13 "nemo-helix[all]"
```

The `nemo-compass-plugin` extra includes `nemo-compass==0.2.0`, which bundles Compass's
analysis code and `trace_ingest` loaders. Compass execution requires Python
3.12 or 3.13; Compass does not yet support Python 3.14.

## CLI

Submit analysis through NeMo Helix. Replace `<run-name>` with the name
returned by `create`:

```bash
uv run nemo compass analysis-runs create --agent research-agent --workspace default --wait
uv run nemo compass analysis-runs list --agent research-agent
uv run nemo compass analysis-runs get "<run-name>"
```

AnalysisRuns execute through `agents.execute`; the Compass plugin stores the
resulting insights in NeMo Helix. `--wait` exits non-zero unless the job
completes successfully. A run and its backing job share one name.

`create` uses those configured models unless you pass `--default-model` and
`--fast-model`. Add `--ethos ETHOS.md` to supply the agent's intended behavior,
`--since <ISO-8601 timestamp>` to set a lower time bound, or
`--evaluation-id <id>` to select one evaluation.

Provide `--agent` and pass the workspace and Ethos explicitly when needed.
`--base-url` defaults to `NHX_BASE_URL`, then `http://localhost:8080`.

### Telemetry requirement

The normalized `agent_name` on the agent's traces must match `--agent`.
For OTLP, set `gen_ai.agent.name` on every span. Intake also normalizes
`llm.agent.name` and `agent.name`; ATIF maps its required `agent.name` automatically.

### Recurrence thresholds

Compass only files insights for tool failures that recur. A tool issue must
appear in at least 3 independent cases, where a case is the evaluation test
case name or, without one, the trace ID. A failure pattern must appear in at
least 3 traces. Runs over a few sessions, such as runs scoped by `--since` or
`--evaluation-id`, can find a real failure that is below these thresholds.

When that happens, the run summary says some observations did not meet the
thresholds. The `analysis-report` result lists the closest ones (up to 5) and
any Compass Signals that were skipped, with their reason:

```text
- below threshold: tool-issues knowledge_search explicit_tool_failure (2 of 3 independent cases)
- skipped: ethos-divergence (No ethos document)
```

### Periodic analysis

```bash
uv run nemo compass analysis enable --agent research-agent
uv run nemo compass analysis status
uv run nemo compass analysis disable --agent research-agent
```

`analysis enable` stores the effective default/fast model pair for scheduled
jobs. The pair comes from `nemo setup`; pass `--default-model` and `--fast-model`
(Model Entity refs such as `default/<model>`) to set it without `nemo setup`, or
to override it. Re-run `enable` after changing the pair.

## API and SDK

The service is mounted under:

```text
/apis/compass/v2/workspaces/{workspace}
```

The plugin SDK is available as `client.compass`, including:

- `client.compass.insights`
- `client.compass.analysis_configs`
- `client.compass.analysis_runs` — `create`, `list_runs`, `get`, and `wait`
  (polls a run until its backing job is terminal)
- `client.compass.analysis_run_statuses`

## Configuration

Periodic analysis is a *deployment* setting, not a per-run one. Like every other
NeMo plugin it is a `NemoConfig`, so it can be set either in the `compass:`
section of the platform config file or through the environment, and the
environment wins. All settings live under `analysis`, with the
`NEMO_COMPASS_` environment prefix.

| Variable | Config key | Default | Meaning |
|---|---|---|---|
| `NEMO_COMPASS_ANALYSIS_ENABLED` | `analysis.enabled` | `true` | Whether the periodic analysis controller runs at all. |
| `NEMO_COMPASS_ANALYSIS_FREQUENCY` | `analysis.frequency` | `daily` | Cadence for each opted-in agent: `daily` or `weekly`. |
| `NEMO_COMPASS_ANALYSIS_TIMEZONE` | `analysis.timezone` | `UTC` | IANA name (e.g. `America/Denver`) the schedule is interpreted in. Converted to the server clock at evaluation time, so runs hold their local hour across DST. An unknown name fails validation. |
| — (see below) | `analysis.run_at_hour` | `0` | Local hour-of-day, 0–23, that scheduled runs fire. |
| — (see below) | `analysis.run_on_weekday` | `monday` | Day scheduled runs fire. Used only when frequency is `weekly`. |
| — (see below) | `analysis.job_profile` | `default` | Jobs execution profile for scheduled Compass jobs. |
| — (see below) | `analysis.base_url` | unset | NeMo Helix base URL passed to Compass jobs. When unset, jobs use their active platform context. |

```bash
export NEMO_COMPASS_ANALYSIS_FREQUENCY=weekly
export NEMO_COMPASS_ANALYSIS_TIMEZONE=America/Denver
```

### Setting the underscored fields

Only single-word fields — `enabled`, `frequency`, `timezone` — have a working
flat environment variable. The plugin sets `env_nested_delimiter="_"`, so a
name like `NEMO_COMPASS_ANALYSIS_RUN_AT_HOUR` is parsed as the nested path
`analysis.run.at.hour`, which does not exist. **The variable is ignored
silently: no error, and the default stays in effect.** Prefer the config file
for these. To set them from the environment anyway, assign the whole `analysis`
object as JSON — unlisted keys keep their defaults:

```bash
export NEMO_COMPASS_ANALYSIS='{"run_at_hour": 17, "run_on_weekday": "friday", "job_profile": "gpu"}'
```

### NeMo Compass self-observability

Whenever a platform base URL is available, NeMo Compass exports its own traces
to Intake's workspace-scoped OTLP endpoint. No opt-in flag or environment
variable is required. Set `NEMO_COMPASS_OBSERVABILITY=false` to opt
out. The endpoint must be HTTPS unless it is loopback.

## Development

```bash
uv run pytest plugins/nemo-compass/tests
uv run ruff check plugins/nemo-compass
```

## Evaluation

Compass-only evaluation is in [`evaluation/`](evaluation/). It can replay pinned
Intake traces or run Tau2 benchmarks before invoking the shared Python analysis runner.
