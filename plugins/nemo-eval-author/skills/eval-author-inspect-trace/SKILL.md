---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: eval-author-inspect-trace
description: >-
  Use when eval-author has routed to this sub-flow for one Intake trace. Do not
  use for instrumenting agents, ingesting telemetry, importing a trace store,
  verifying ingest, or querying Intake outside Eval Author; those belong to
  nemo-intake.
triggers:
  - eval-author routed to inspect-trace
  - inspect the intake trace eval-author named
  - continue eval-author intake trace inspection
not-for:
  - eval-author (use for the standard, the boundaries, and to pick a sub-flow)
  - eval-author-discover (use to establish whether a repository's Harbor suite is runnable)
  - eval-author-audit (use to validate an existing finite audit.md coverage denominator)
  - nemo-intake (use to instrument agents, ingest telemetry, or query Intake outside Eval Author)
  - nemo-experimentalist (use for an optimization experiment across many trials)
compatibility: >-
  A working nemo CLI invocation, an explicit workspace, and read access to
  Intake on a configured local or remote NeMo Platform instance.
maturity: alpha
license: Apache-2.0
user-invocable: false
allowed-tools: Bash, Read, Write, Grep, Glob
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Eval Author: inspect an Intake trace

Read one Intake trace, explain its trajectory, and save an evidence-backed
report. Start this sub-flow only after `eval-author` selects it. For
instrumentation, ingest, or a general Intake query, use `nemo-intake`. Read
`eval-author` for the shared standard and boundaries.

## Requirements

Use a working `nemo` CLI invocation with an explicit workspace and read access to
Intake. The configured NeMo Platform instance can be local or remote. Use the
active context, a caller-supplied context, or `NMP_BASE_URL` and
`NMP_ACCESS_TOKEN`.

### Resolve the CLI invocation

Resolve the CLI invocation once before the first Intake read. If the caller
provides an invocation, verify that caller-supplied invocation first. For a
caller-supplied `uv run nemo` invocation, add `--no-sync`.

Otherwise, probe these candidates in order with the `--version` flag:

1. Use `nemo` when `command -v nemo` succeeds.
2. Use `.venv/bin/nemo` when that executable exists in the source checkout.
3. Use `uv run --no-sync nemo` when `uv` and the prepared environment provide
   the CLI.

Continue after a launcher error while another candidate remains. If no
candidate works, quote the launcher errors and stop.

Use the resolved invocation for every command. The examples use `nemo` as the
logical prefix; replace only that prefix. In the report, record each expanded
command instead of the logical example.

Don't install the CLI, authenticate, change contexts, start services, or repair
the platform. Don't run `uv sync`, use `uvx`, or activate an environment. Don't
print, copy, or save credentials. The first Intake read validates access. If it
exits nonzero, quote the CLI error and stop.

## Select the trace

Accept a bare Intake trace ID or `intake://traces/TRACE_ID`. Reject every other
URI scheme. If no trace is named, list recent candidates:

```bash
nemo intake traces list --output-format=json --workspace="WORKSPACE" --mode=preview --sort=-started_at --page-size=20
```

If the request names an agent, list recent traces that contain its spans:

```bash
nemo intake spans groups list --output-format=json --workspace="WORKSPACE" --by=trace_id --filter.agent-name="AGENT_NAME" --sort=-started_at --page-size=20
```

For a time-bounded agent search, add the command's JSON `--filter` value with
`started_at.gte` and `started_at.lte`. Don't encode the filter. If the command
returns no candidates, report the empty result and stop. Don't widen the search
without user direction.

## Read the trace structure

Read the server-computed trace rollups:

```bash
nemo intake traces get --output-format=json --workspace="WORKSPACE" --mode=detailed "TRACE_ID"
```

Then read every span without payloads:

```bash
nemo intake spans list --output-format=json --workspace="WORKSPACE" --filter.trace-id="TRACE_ID" --mode=summary --sort=started_at --all-pages
```

Use Intake's root status, duration, token, cost, span-count, error-count, model,
and provider rollups. Use span timestamps and parent IDs when order matters.
Don't recreate a timeline object or assume that an error span made the trace
fail.

## Read selective detail

Read bounded previews for a server-side selection:

```bash
nemo intake spans list --output-format=json --workspace="WORKSPACE" --filter.trace-id="TRACE_ID" --filter.status=error --mode=preview --sort=started_at --page-size=20
```

Use `--filter.kind` or `--filter.parent-span-id` for equivalent selections. Read
an exact payload only when its text affects the assessment:

```bash
nemo intake spans get --output-format=json --workspace="WORKSPACE" "SPAN_ID"
```

Require the returned `trace_id` to equal the selected trace ID. Otherwise,
discard the span and report the mismatch. Don't recreate payload-length flags.

## Read evaluator evidence

Collect unique session IDs from the summary spans. Query each relevant session:

```bash
nemo intake evaluator-results list --output-format=json --workspace="WORKSPACE" --filter.session-id="SESSION_ID" --sort=created_at --all-pages
```

Keep only results whose `span_id` belongs to the selected trace. Quote evaluator
comments when they affect the outcome.

## Assess the evidence

Use exact payload text only when it changes the assessment. For every important
moment, name the span ID, input or attempt, output, and relationship to nearby
spans. Separate observations from interpretations.

Assign one outcome:

- `success`: Direct evidence shows that the requested work completed.
- `failure`: Direct evidence shows that the work didn't complete or was wrong.
- `unknown`: Evidence is missing, incomplete, or contradictory.

Start every finding with one category:

- `behavior`: What the agent did without judging it as defective.
- `issue`: A result that harmed correctness, safety, cost, or latency.
- `recovery`: A later action that handled an earlier problem.
- `uncertainty`: A question that the evidence can't settle.

Don't invent an `issue` for a successful trace. Tie every important claim to a
span ID, evaluator result ID, or source symbol.

Inspect local source only after trace evidence identifies behavior that source
can explain. Name the repository-relative path and symbol. Source can't override
recorded trace evidence. Report conflicts as `uncertainty`.

## Save the report

Use the trace ID returned by Intake. Replace characters outside letters,
numbers, `.`, `_`, and `-` with `_`. Save the report as
`.eval-author/traces/intake-TRACE_ID.md`, replacing any report for the same trace.

Put only source metadata and exact read commands in JSON-compatible YAML front
matter. Record the workspace and a named CLI context when used. Never record a
token. Then include outcome, summary, important moments, findings, and optional
source evidence:

```markdown
---
{"source":{"kind":"intake","trace_id":"TRACE_ID","workspace":"WORKSPACE","context":"CONTEXT_OR_NULL"},"commands":["EXACT_READ_COMMAND"]}
---
# Trace TRACE_ID
## Source
<SOURCE_METADATA>
## Outcome
<OUTCOME_AND_DECISIVE_EVIDENCE>
## Summary
<TRACE_SUMMARY>
## Important moments
- `<SPAN_ID>`: <TRANSITION_AND_SIGNIFICANCE>
## Findings
- **<FINDING_CATEGORY>**: <CLAIM_AND_EVIDENCE_IDS>
## Source evidence
- `<PATH>:<SYMBOL>`: <SOURCE_BACKED_EXPLANATION>
```

Omit **Source evidence** when you didn't inspect source. Before completion,
verify every important moment and finding names evidence, the outcome uses an
allowed value, and only the report changed. Leave the report uncommitted and
state its path.
