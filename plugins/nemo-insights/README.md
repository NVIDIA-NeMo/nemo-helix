<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Insights

NeMo Helix plugin for analyzing agent telemetry and persisting actionable insights.

## Install from the monorepo

```bash
uv sync
```

The plugin is installed by default through the root workspace's `enabled-plugins` group.

## CLI

From an agent directory, Insights discovers `optimizer.yaml` in the current
directory or its parents. Start by checking the profile and its environment,
then run the analyst:

```bash
cd <agent-directory>
uv run nemo agents analyst doctor
uv run nemo agents analyst run
```

Run `nemo setup` first to select the default and fast Platform Model Entities.
The Analyst uses the default model for analysis and the fast model for context
summarization; an existing context without `fast_model` reuses `default_model`.
Provider credentials remain in Platform Secrets.

The profile contract consumed by Insights is deliberately small:

```yaml
agent: research-agent
ethos: ETHOS.md  # optional
workspace: default         # optional; defaults to "default"
```

Only `agent`, `ethos`, and `workspace` are consumed by Insights.
Unknown experiment-owned fields are ignored, while the reserved `profile_dir`
field is rejected. `agent` is required. Relative `ethos` paths are
resolved relative to the profile. When it is omitted, Insights looks for
`ETHOS.md`, then `README.md`, beside the profile.

An adjacent `.env` is loaded when a profile is found, without replacing
variables already set in the shell. For this shared profile workflow,
`NHX_BASE_URL` is the only base-URL environment variable. Resolution order is
explicit command-line flags, then profile values (for `agent`, `ethos`,
and `workspace`) or `NHX_BASE_URL` (for the base URL), then the built-in
defaults. `--base-url` takes precedence over `NHX_BASE_URL`.

### Telemetry requirement

The analyst scopes Intake span queries to the configured `agent`. The normalized
`agent_name` on each span must therefore match `agent` in `optimizer.yaml` or
`--agent`. For OTLP, always set `gen_ai.agent.name` on every span; Intake also
normalizes `llm.agent.name` and `agent.name` from instrumentation that emits
those conventions. ATIF maps its required `agent.name` automatically.

### Where insights are written

Insights always go to the platform, through the Insights plugin API.

Pass `--insights-file-output <path>` to also keep a local copy: the platform is
written first and the file mirrors what it stored, platform ids included, so a
later run's updates land in both stores. Each run merges into the file rather
than overwriting it. Because the platform is the source of truth, a file that
cannot be written is reported as a warning on the run report instead of failing
the run.

```bash
uv run nemo agents analyst run                                  # platform only
uv run nemo agents analyst run --insights-file-output out.yaml  # platform + local mirror
```

```bash
uv run nemo agents analyst run \
  --agent research-agent \
  --workspace default \
  --base-url http://localhost:8080

uv run nemo insights analysis enable --agent research-agent
uv run nemo insights analysis status
uv run nemo insights analysis disable --agent research-agent
```

`nemo agents analyst run` runs the Analyst locally, in your shell. To have the
platform run it as a job instead, submit an *analysis run*:

```bash
uv run nemo insights analysis-runs create --agent research-agent --wait
uv run nemo insights analysis-runs list --agent research-agent
uv run nemo insights analysis-runs get <run-name>
```

A run and the `agents.execute` job backing it share one name, so `get` returns
both together. `--wait` polls to a terminal job state and exits non-zero unless
the job completed. `create` fills the default/fast model pair from your CLI
config unless you pass `--default-model` / `--fast-model`; the request must
carry it because the Platform process cannot read that file.

`analysis enable` stores the effective default/fast pair in the server-side
analysis config so scheduled jobs do not depend on the operator's local CLI
file. Re-run `enable` after changing the pair with `nemo setup`. Existing
enabled records created before model-pair persistence must also be re-enabled.

`--base-url` defaults to `NHX_BASE_URL`, then `http://localhost:8080`.

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
| — (see below) | `analyst.base_url` | unset | Platform base URL passed to analyst jobs. When unset, jobs use their active platform context. |
| — (see below) | `analyst.inference_api_key_secret_name` | unset | Platform secret whose value is exposed to analyst jobs as `INFERENCE_API_KEY`. Temporary until FP-202 moves analyst model execution to platform-registered models. |

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

### Analyst self-observability

Whenever a platform base URL is available, the Analyst exports its own traces
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
Intake traces or run Tau2 benchmarks before invoking `nemo agents analyst run`.

## What consumes an Insight

Insights is the analysis half of a two-plugin loop. The
[NeMo Experimentalist](../nemo-experimentalist/README.md) plugin consumes what
the analyst produces and uses it to improve the agent against Harbor-compatible
train and validation datasets:

```text
nemo agents analyst run → Platform Insight ID (or --insights-file-output mirror)
                       → nemo agents experimentalist doctor
                       → nemo agents experimentalist run
```

The Experimentalist accepts either a Platform Insight ID or a local mirror
file, so `--insights-file-output` is the option to reach for when you want a
run that does not have to resolve an ID against Platform. Insights does not
propose or evaluate agent changes itself; the Experimentalist does not analyze
traces or host an Insight API.

[Insight-driven optimization](../../docs/agents/insight-driven-optimization.mdx)
walks the full loop end to end.
