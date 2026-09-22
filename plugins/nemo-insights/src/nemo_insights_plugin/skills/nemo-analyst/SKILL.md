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
  problems hiding in production traces. Produces the Insight that the
  Experimentalist later acts on.
triggers:
  - nemo-analyst
  - analyze my agent's traces
  - why my agent keeps failing
  - generate insights for my agent
  - find recurring failure patterns
  - run the analyst
  - my agent keeps getting things wrong
not-for:
  - nemo-experimentalist (use to act on an Insight and change the agent; this skill produces the Insight it consumes)
  - nemo-intake (use to instrument an agent, ingest telemetry, or query raw spans; this skill interprets telemetry that already landed)
  - nemo-experiments-upload (use to upload traces and evaluation results into Intake; this skill reads them back out)
  - nemo-explore (use to design an agent that does not exist yet; this skill needs a running agent with traces)
  - nemo-evaluator (use to author evaluations and metrics; this skill analyzes production behavior)
compatibility: >-
  nemo-platform >= 0.1.0; requires the Insights plugin, a reachable platform
  with Intake telemetry for the target agent, and a model the platform can call
  on the Analyst's behalf. No Docker or datasets needed.
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
- The Analyst has a model to run on. It is an LLM agent itself, and how that is
  configured is changing, so let pre-flight tell you whether it is satisfied —
  it names what is missing and how to set it. Don't reach for the
  Experimentalist's configuration; that is a different contract.

An `ETHOS.md` file is optional. It gives the Analyst the agent's intent,
constraints, and success criteria. Code and traces don't contain that context.
Without it, the Analyst can only judge an agent against itself.

## Pre-flight

```bash
nemo agents analyst doctor
```

Only two results block a run: no usable model configured, and an
`optimizer.yaml` that is missing or unparseable — the second only if you intend
to run without `--agent`. Doctor takes no `--agent` flag, so it always checks
for a profile and always reports a red line when there is none; when you pass
`--agent`, that line is noise. Platform reachability and the workspace probe
only ever warn.

## Run it

```bash
nemo agents analyst run --agent <agent-name> --workspace <workspace>
```

Add `--ethos ETHOS.md` to tell it what the agent is supposed to do,
and `--verbose` to stream its tool calls and reasoning to stderr. Expect several
minutes; it surveys many sessions before drilling into any of them.

From an agent directory, an `optimizer.yaml` profile supplies `agent`,
`workspace`, and `ethos`, so the flags above become optional:

```bash
nemo agents analyst run
```

The profile is discovered by walking up from the current directory. Only those
three fields are read from it; other keys belong to the Experimentalist and are
ignored.

## Where Insights are stored

Insights always go to the platform. `--insights-file-output` additionally
mirrors what the platform stored, platform IDs included, merging into that file
on each run; a mirror that cannot be written warns rather than failing the run.

```bash
nemo agents analyst run --agent <agent-name> --insights-file-output .nemo-optimizer/insights.yaml
```

That path is what the Experimentalist reads by default, so it is the
conventional choice when handing off locally.

## Verify

Do not report success on an exit code. The run prints a line per operation —
`- created: <title> [<insight-id>] (<n> trace refs)`, `- updated: <insight-id>
(<n> trace refs)`, or `- no insights created or updated`, which is a successful
run too. Read back by id whatever it says it wrote, and check each carries a
clear title, an actionable description, and non-empty `trace_refs`. Listing by
`?agent=` also returns earlier runs, so it attests the store, not this run:

```bash
curl --fail-with-body \
  "$NMP_BASE_URL/apis/insights/v2/workspaces/<workspace>/insights/<insight-id>"
```

On an authenticated platform pass the token through curl's config, not argv
where any process on the host can read it:

```bash
printf 'header = "Authorization: Bearer %s"' "$(nemo auth token)" | curl -K - <url>
```

Stored Insights also appear in Studio's optimizer view for the workspace.

## When it finds nothing

Besides a real "nothing worth filing", three things produce an empty result.
Scoping: the Analyst reads only what `--agent` and `--workspace` together
select, and `agent_name` is carried on agent-level spans, not on their model and
tool children. Volume: too few traces looks the same as a healthy agent. And
telemetry that captures only the shape of a run, spans without the inputs and
outputs, leaves nothing to judge however many spans there are.

## Hand off

Once an Insight exists, the Experimentalist acts on it:

```bash
nemo agents experimentalist run
```

For the full data model, the Analyst's tool set, periodic analysis via
`nemo insights analysis enable`, and the rest of the loop, see
[Insight-Driven Optimization](https://github.com/NVIDIA-NeMo/nemo-platform/blob/main/docs/agents/insight-driven-optimization.mdx).
