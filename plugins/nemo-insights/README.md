<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Insights

NeMo Platform plugin for analyzing agent telemetry and persisting actionable insights.

Analysis uses [trace-intel](https://github.com/NVIDIA-NeMo/labs-trace-intel).
NeMo Platform supplies authenticated trace and model access and stores the resulting insights.

## Install from the monorepo

```bash
uv sync
```

The plugin is installed by default through the root workspace's `enabled-plugins` group.

## CLI

Submit analysis through NeMo Platform:

```bash
uv run nemo insights analysis-runs create --agent research-agent --workspace default --wait
uv run nemo insights analysis-runs list --agent research-agent
uv run nemo insights analysis-runs get <run-name>
```

AnalysisRuns execute through `agents.execute`; the Insights plugin stores the
resulting insights in NeMo Platform. `--wait` exits non-zero unless the job
completes successfully. A run and its backing job share one name.

Run `nemo setup` first to select the default and fast NeMo Platform models.
`create` uses those configured models unless you pass `--default-model` and
`--fast-model`. Add `--ethos ETHOS.md` to supply the agent's intended behavior,
`--since <ISO-8601 timestamp>` to set a lower time bound, or
`--evaluation-id <id>` to select one evaluation.

Provide `--agent` and pass the workspace and Ethos explicitly when needed.
`--base-url` defaults to `NMP_BASE_URL`, then `http://localhost:8080`.

### Telemetry requirement

The normalized `agent_name` on the agent's traces must match `--agent`.
For OTLP, set `gen_ai.agent.name` on every span. Intake also normalizes
`llm.agent.name` and `agent.name`; ATIF maps its required `agent.name` automatically.

### Periodic analysis

```bash
uv run nemo insights analysis enable --agent research-agent
uv run nemo insights analysis status
uv run nemo insights analysis disable --agent research-agent
```

`analysis enable` stores the effective default/fast model pair for scheduled
jobs. Re-run `enable` after changing the pair with `nemo setup`.

## API and SDK

The service is mounted under:

```text
/apis/insights/v2/workspaces/{workspace}
```

The plugin SDK is available as `client.insights`, including:

- `client.insights.insights`
- `client.insights.analysis_configs`
- `client.insights.analysis_runs` — `create`, `list_runs`, `get`, and `wait`
  (polls a run until its backing job is terminal)
- `client.insights.analysis_run_statuses`

## Configuration

Periodic analysis is a *deployment* setting, not a per-run one. Like every other
NeMo plugin it is a `NemoConfig`, so it can be set either in the `insights:`
section of the platform config file or through the environment, and the
environment wins. All settings live under `analyst`, with the
`NEMO_INSIGHTS_` environment prefix.

| Variable | Config key | Default | Meaning |
|---|---|---|---|
| `NEMO_INSIGHTS_ANALYST_ENABLED` | `analyst.enabled` | `true` | Whether the periodic analysis controller runs at all. |
| `NEMO_INSIGHTS_ANALYST_FREQUENCY` | `analyst.frequency` | `daily` | Cadence for each opted-in agent: `daily` or `weekly`. |
| `NEMO_INSIGHTS_ANALYST_TIMEZONE` | `analyst.timezone` | `UTC` | IANA name (e.g. `America/Denver`) the schedule is interpreted in. Converted to the server clock at evaluation time, so runs hold their local hour across DST. An unknown name fails validation. |
| — (see below) | `analyst.run_at_hour` | `0` | Local hour-of-day, 0–23, that scheduled runs fire. |
| — (see below) | `analyst.run_on_weekday` | `monday` | Day scheduled runs fire. Used only when frequency is `weekly`. |
| — (see below) | `analyst.job_profile` | `default` | Jobs execution profile for scheduled analyst jobs. |
| — (see below) | `analyst.base_url` | unset | NeMo Platform base URL passed to analyst jobs. When unset, jobs use their active platform context. |
| — (see below) | `analyst.inference_api_key_secret_name` | unset | NeMo Platform secret whose value is exposed to analyst jobs as `INFERENCE_API_KEY`. Temporary until FP-202 moves analyst model execution to platform-registered models. |

```bash
export NEMO_INSIGHTS_ANALYST_FREQUENCY=weekly
export NEMO_INSIGHTS_ANALYST_TIMEZONE=America/Denver
```

### Setting the underscored fields

Only single-word fields — `enabled`, `frequency`, `timezone` — have a working
flat environment variable. The plugin sets `env_nested_delimiter="_"`, so a
name like `NEMO_INSIGHTS_ANALYST_RUN_AT_HOUR` is parsed as the nested path
`analyst.run.at.hour`, which does not exist. **The variable is ignored
silently: no error, and the default stays in effect.** Prefer the config file
for these. To set them from the environment anyway, assign the whole `analyst`
object as JSON — unlisted keys keep their defaults:

```bash
export NEMO_INSIGHTS_ANALYST='{"run_at_hour": 17, "run_on_weekday": "friday", "job_profile": "gpu"}'
```

### NeMo Analyst self-observability

Whenever a platform base URL is available, the NeMo Analyst exports its own traces
to Intake's workspace-scoped OTLP endpoint. No opt-in flag or environment
variable is required. Set `NEMO_INSIGHTS_ANALYST_OBSERVABILITY=false` to opt
out. The endpoint must be HTTPS unless it is loopback.

## Development

```bash
uv run pytest plugins/nemo-insights/tests
uv run ruff check plugins/nemo-insights
```

## Evaluation

The analyst-only evaluation is in [`evaluation/`](evaluation/). It can replay pinned
Intake traces or run Tau2 benchmarks before invoking the shared Python analysis runner.
