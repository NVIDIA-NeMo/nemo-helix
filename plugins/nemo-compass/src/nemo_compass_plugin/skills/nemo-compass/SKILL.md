---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-compass
description: >-
  Analyze an agent's production traces to find recurring failure patterns and
  record each as an Insight. Surveys spans, evaluator scores, and user feedback
  across many sessions, clusters similar failures, then files every finding as a
  titled Insight carrying the trace IDs that evidence the problem. Answers why
  an agent keeps failing, where it gets things wrong, and the recurring
  problems hiding in production traces.
triggers:
  - nemo-compass
  - analyze my agent's traces
  - why my agent keeps failing
  - generate insights for my agent
  - find recurring failure patterns
  - run Compass
  - my agent keeps getting things wrong
not-for:
  - nemo-intake (use to instrument an agent, ingest telemetry, or query raw spans; this skill interprets telemetry that already landed)
  - nemo-experiments-upload (use to upload traces and evaluation results into Intake; this skill reads them back out)
  - nemo-evals (use to author evaluations and metrics; this skill analyzes production behavior)
compatibility: >-
  nemo-helix >= 0.1.0; requires the Compass plugin, a reachable platform
  with Intake telemetry for the target agent, and a model the platform can call
  on Compass's behalf. Requires Jobs and a configured agents.execute runtime.
maturity: beta
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Compass

## Before running

Compass reads telemetry; it cannot create it. Confirm all three:

- The target agent already has traces in Intake. No traces means no Insights.
- The platform is reachable at `NHX_BASE_URL`.
- The CLI has default and fast Platform models configured through `nemo setup`,
  or the submission supplies explicit model references. The Platform must have
  Jobs and an `agents.execute` runtime available to execute the analysis.

Compass runs as the workspace's Compass agent, `compass-analyst`. Check that it
exists, and create it if it doesn't:

```bash
nemo agents get compass-analyst --workspace "<workspace>" || \
  nemo agents create compass.analyst --workspace "<workspace>"
```

An Ethos is optional. It gives Compass the agent's intent, constraints, and
success criteria. Code and traces don't contain that context. Without it,
Compass can only judge an agent against itself. Compass reads it from
`ETHOS.md` in the analyzed agent's `<agent-name>-ethos` fileset, so put it there
before running:

```bash
nemo files filesets create "<agent-name>-ethos" --workspace "<workspace>" --exist-ok
nemo files upload ETHOS.md "<agent-name>-ethos" --workspace "<workspace>"
```

## What it produces

Analyze an agent's behavior from its own telemetry and record what recurs as
Insights.

An Insight is a persistent, named description of one recurring problem, and it
is the unit of work the rest of the optimization loop runs on. Each carries:

- `title` — a sentence naming the failure, such as "Retrieval drops relevant
  context near the token limit"
- `description` — the failure mode, the tool or model call it affects, and the
  conditions that trigger it
- `evidence` — supporting trace IDs with optional URLs and span references, so a developer can
  audit the reasoning and build regression tests

Compass targets at least three representative traces per Insight and appends
evidence to an existing Insight rather than filing a near-duplicate. It judges
behavior rather than status or scores, so it finds failures in sessions that
reported success and passed their evaluations. Two well-evidenced Insights are
worth more than ten vague ones, so a run that files nothing is a valid outcome.

## Run it

Replace the quoted placeholders below with the target agent and workspace.

```bash
nemo agents execute --agent compass-analyst --workspace "<workspace>" \
  --spec '{"context": {"agent": "<agent-name>"}}'
nemo jobs watch "<job-name>" --workspace "<workspace>"
```

`execute` prints the created job; pass its `name` to `nemo jobs watch`. Add
`"since"` (an ISO-8601 lower time bound) or `"evaluation_id"` (to select an
evaluation) to the context. Expect several minutes. `nemo jobs watch` exits
non-zero unless the job completes successfully.

The CLI fills in the configured default and fast models. Run `nemo setup` if
they are missing, or set `"default_model"` and `"fast_model"` in the context.

## Where Insights are stored

Compass creates and updates insights in NeMo Helix after analysis.
Stored insights appear in Studio's optimizer view for the workspace.

## Verify

Replace the quoted placeholders with the job name and workspace:

```bash
nemo jobs get-status "<job-name>" --workspace "<workspace>"
nemo jobs results download analysis-report --job "<job-name>" --workspace "<workspace>" -o analysis-report.txt
```

Confirm the job completed and inspect its analysis report. A completed run may
produce no new insights. When it records insights, read them back and check
for a clear title, an actionable description, and non-empty `evidence`.
Listing all insights for the agent can include earlier runs.

```bash
curl --fail-with-body \
  "$NHX_BASE_URL/apis/compass/v2/workspaces/<workspace>/insights/<insight-id>"
```

On an authenticated platform pass the token through curl's config, not argv.
Replace `<url>` with the insight URL above:

```bash
printf 'header = "Authorization: Bearer %s"' "$(nemo auth token)" | curl --fail-with-body -K - "<url>"
```

## When it finds nothing

Besides a real "nothing worth filing", three things produce an empty result.
Scoping: Compass reads only what the context's `agent` and the job's workspace
together select, and `agent_name` is carried on agent-level spans, not on their model and
tool children. Volume: too few traces looks the same as a healthy agent. And
telemetry that captures only the shape of a run, spans without the inputs and
outputs, leaves nothing to judge however many spans there are.
