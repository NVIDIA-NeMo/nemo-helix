---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: eval-author
description: >-
  Work on evaluation suites in a user's repository, derive an environment from
  trace evidence, or understand an agent run from NeMo Intake. Owns the evidence
  standard that every Eval Author sub-flow
  follows. Use when the user asks "help me with my evals",
  "what's the state of the eval suite here?", "what happened in this trace?", or
  when you need to pick between the Eval Author sub-flows. Routes to a sub-flow
  and changes none of the user's source. The selected sub-flow uses the
  provider's supported tools and saves its findings under `.eval-author/`.
triggers:
  - help me with the evals in this repo
  - what is the state of the eval suite here
  - I inherited a repo with Harbor tasks in it
  - work on my evaluation suite
  - what happened in this agent trace
  - create an evaluation environment from a trace
  - which eval author step do I need
not-for:
  - eval-author-discover (use to run the discovery pass and get a runnable verdict)
  - eval-author-audit (use to generate, validate, measure, or aggregate audit.md coverage)
  - eval-author-task-create (use to propose dataset improvements; when task creation is requested, create and prove an eligible Harbor task)
  - eval-author-inspect-trace (use after this skill selects the trace sub-flow)
  - eval-author-trace-environment (use to derive a Harbor environment from canonical ATIF evidence)
  - nemo-intake (use to instrument agents, ingest telemetry, or query Intake outside Eval Author)
  - mlflow-to-atif (use to convert MLflow traces into canonical ATIF files)
  - nemo-experimentalist (use to run insight-driven optimization end to end, which drives the Eval Author agent itself)
  - nemo-evaluator (use to run an existing benchmark rather than work on a repository's own suite)
compatibility: >-
  The router is read-only. Discovery, audit, and task creation use the local
  checkout. Task execution requires Harbor and may require Docker and provider
  credentials. Trace inspection requires the nemo CLI, an explicit workspace,
  and read access to configured Intake.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: [Read, Grep, Glob]
---

# Eval Author

Work on repository-owned evaluation suites and understand agent traces. Route
each request to the narrow sub-flow that owns it.

A report that a downstream model trusts has to be right. A plausible report is
worse than no report when somebody acts on it.

## The standard

**Every fact you record comes from authoritative evidence, not a guess.**

The authority depends on the sub-flow:

- For suite discovery, Harbor's validators judge runnability. A file's presence
  doesn't prove that Harbor accepts it.
- For audit-spec validation, the bundled schema and validator judge the finite
  `audit.md` coverage denominator.
- For task creation, Harbor's Oracle judges task solvability and verifier
  correctness; measured ATIF proves whether repeated runs close the selected gap.
- For trace inspection, Intake establishes what happened. Local source code can
  explain behavior, but it can't replace recorded trace evidence.
- For trace-derived environments, canonical ATIF establishes the request and
  Harbor's NOP and Oracle runs prove the generated environment.

No sub-flow reimplements a provider's rules. When evidence can't settle a claim,
the report marks the claim unproven or uncertain.

## Vocabulary

The sub-flows share this language, and reports use it verbatim.

| Term | Meaning |
|---|---|
| Check | One named result: `pass`, `warn`, or `fail`. Carries a message and, when it fails, a hint |
| Required | A failing required check blocks the suite. Report the suite as not ready |
| Advisory | A warning worth surfacing that blocks nothing |
| Rung | One step of a provider's validation ladder, ordered so a lower rung's failure often clears once a higher one is fixed |
| Proven | A provider judged this check. An unproven check is an observation and never evidence |
| Provider | The evaluation framework that owns the rules. Harbor today |
| Finding | One trace claim categorized as `behavior`, `issue`, `recovery`, or `uncertainty`, with evidence IDs |
| Outcome | The trace assessment: `success`, `failure`, or `unknown` |

## Sub-flows

Read the sub-flow's own `SKILL.md` and follow it. This file carries the standard
and the boundaries; the sub-flow carries the steps.

| Sub-flow | Use it to |
|---|---|
| `eval-author-discover` | Establish whether a repository's evaluations run, name the rung that fails, and get the exact command to run them |
| `eval-author-audit` | Generate and validate a finite `audit.md` coverage denominator, measure and aggregate trace coverage, and report findings |
| `eval-author-task-create` | Propose concrete dataset improvements from audit findings; when task creation is requested, create one eligible Harbor task and prove it with Oracle and repeated measured runs |
| `eval-author-inspect-trace` | Understand one Intake trace without presuming that the trace contains a failure. Not user-invocable; this skill selects it |
| `eval-author-trace-environment` | Normalize one trace to ATIF, make a privacy-reviewed candidate decision, and build a private Harbor task when evidence supports it |

`eval-author-audit` works one level above tasks: it generates and validates the
coverage denominator, measures traces against it, and aggregates deterministic
coverage reports. `eval-author-task-create` owns the proposal step: prioritize
dataset improvements from those findings across tools, capabilities, and failure
cases while preserving their measured or unmeasured status. Proposal-only
requests stop there. Its subsequent task-creation path consumes only actionable
tool gaps and uses Harbor's native task scaffolder rather than guessing a task
layout. For a requested full workflow, proceed from audit to proposals even when
there are no eligible tool gaps; an audit-only request ends with the findings.

## Boundaries

These hold for every sub-flow. They exist because the repository belongs to the
user, not to you.

- **Propose, never mutate customer source.** Read the user's source and report on
  it. Do not edit, move, or reformat any of it, including its `.gitignore`. The
  only files you add belong under `.eval-author/`, which is theirs to commit or
  ignore.
  `eval-author-discover` scripts write nothing; `eval-author-audit` writes only
  requested audit artifacts; `eval-author-task-create` writes only drafts,
  proposals, job outputs, and measurements there; `eval-author-trace-environment`
  writes only private, gitignored task workspaces there.
- **A missing tool is a finding, not a task.** When the provider is not installed,
  say so and stop short of proving anything. Report what you found regardless, and
  do not install the provider into the user's environment.
- **Do not run without approval.** Discovery proves an existing suite can run and
  hands over the command. Task creation may run Oracle locally, then starts
  real-agent jobs only when the user explicitly asked for or approved that spend.
- **Trusted repositories only.** Validating a config can execute repository code,
  because an agent named by import path gets imported. If the repository is not
  trusted, say so and stop.
- **Intake reads are narrow.** Only `eval-author-inspect-trace` reads Intake. It
  uses read-only `nemo intake` commands against the configured instance and
  workspace. No sub-flow discovers accounts, ingests data, uploads files, or
  changes a remote resource.

## Reporting

Lead with the verdict or outcome, then the evidence.

For dataset proposals, follow `eval-author-task-create` Step 1: lead with concrete
recommendations and their evidence, then supporting coverage counts and limits. An empty
task-generation selection does not establish that the dataset needs no changes.

State whether the findings are proven, whether the suite is ready, and the names
of the checks that failed. Never describe a suite as ready while a required check
fails, and never present an observation as proof. When a sub-flow could not reach
its provider, the only honest headline is that nothing was proven.

For a trace, use `success`, `failure`, or `unknown`. Tie key moments and findings
to span IDs, evaluator result IDs, or source symbols. A healthy trace doesn't
need an issue finding.
