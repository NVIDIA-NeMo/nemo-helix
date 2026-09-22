---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-analyst
description: >-
  Analyze an agent's production traces to find recurring failure patterns and
  record each as an Insight. Surveys spans, evaluator scores, and user feedback
  across many sessions, clusters similar failures, then files every finding as a
  titled Insight carrying the trace IDs that evidence the problem. Answers why
  an agent keeps failing, where it gets things wrong, and the recurring
  problems hiding in production traces.
triggers:
  - nemo-analyst
  - analyze my agent's traces
  - why my agent keeps failing
  - generate insights for my agent
  - find recurring failure patterns
  - run the analyst
  - my agent keeps getting things wrong
not-for:
  - nemo-intake (use to instrument an agent, ingest telemetry, or query raw spans; this skill interprets telemetry that already landed)
  - nemo-experiments-upload (use to upload traces and evaluation results into Intake; this skill reads them back out)
  - nemo-evaluator (use to author evaluations and metrics; this skill analyzes production behavior)
compatibility: >-
  nemo-platform >= 0.1.0; requires the Insights plugin, a reachable platform
  with Intake telemetry for the target agent, and a model the platform can call
  on the Analyst's behalf. Requires Jobs and a configured agents.execute runtime.
maturity: beta
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Analyst

Analyze an agent's behavior from its own telemetry and record what recurs as
Insights.

## What it produces

An Insight is a persistent, named description of one recurring problem, and it
is the unit of work the rest of the optimization loop runs on. Each carries:

- `title` — a sentence naming the failure, such as "Retrieval drops relevant
  context near the token limit"
- `description` — the failure mode, the tool or model call it affects, and the
  conditions that trigger it
- `trace_refs` — the Intake trace IDs cited as evidence, so a developer can
  audit the reasoning and build regression tests

The Analyst targets at least three representative traces per Insight and appends
evidence to an existing Insight rather than filing a near-duplicate. It judges
behavior rather than status or scores, so it finds failures in sessions that
reported success and passed their evaluations. Two well-evidenced Insights are
worth more than ten vague ones, so a run that files nothing is a valid outcome.

## Before running

The Analyst reads telemetry; it cannot create it. Confirm all three:

- The target agent already has traces in Intake. No traces means no Insights.
- The platform is reachable at `NMP_BASE_URL`.
- The CLI has default and fast Platform models configured through `nemo setup`,
  or the submission supplies explicit model references. The Platform must have
  Jobs and an `agents.execute` runtime available to execute the analysis.

An `ETHOS.md` file is optional. It gives the Analyst the agent's intent,
constraints, and success criteria. Code and traces don't contain that context.
Without it, the Analyst can only judge an agent against itself.

## Run it

```bash
nemo insights analysis-runs create --agent <agent-name> --workspace <workspace> --wait
```

Add `--ethos ETHOS.md` to supply the agent's intended behavior, `--since` for
an ISO-8601 lower time bound, or `--evaluation-id` to select an evaluation.
The command submits an AnalysisRun backed by an `agents.execute` job. Expect
several minutes. `--wait` exits non-zero unless the job completes successfully.

The CLI supplies the configured default and fast models. Run `nemo setup` if
they are missing, or pass `--default-model` and `--fast-model` explicitly.

## Where Insights are stored

The Insights plugin creates and updates insights in NeMo Platform after analysis.
Stored insights appear in Studio's optimizer view for the workspace.

## Verify

Inspect the submitted run and its backing job:

```bash
nemo insights analysis-runs get <run-name> --workspace <workspace>
```

Confirm the job completed and inspect its analysis report. A completed run may
produce no new insights. When it records insights, read them back and check
for a clear title, an actionable description, and non-empty `trace_refs`.
Listing all insights for the agent can include earlier runs.

```bash
curl --fail-with-body \
  "$NMP_BASE_URL/apis/insights/v2/workspaces/<workspace>/insights/<insight-id>"
```

On an authenticated platform pass the token through curl's config, not argv:

```bash
printf 'header = "Authorization: Bearer %s"' "$(nemo auth token)" | curl -K - <url>
```

## When it finds nothing

Besides a real "nothing worth filing", three things produce an empty result.
Scoping: the Analyst reads only what `--agent` and `--workspace` together
select, and `agent_name` is carried on agent-level spans, not on their model and
tool children. Volume: too few traces looks the same as a healthy agent. And
telemetry that captures only the shape of a run, spans without the inputs and
outputs, leaves nothing to judge however many spans there are.
