---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: eval-author
description: >-
  Build first evals from a required Ethos or work on existing evaluation suites
  in a user's repository, derive an environment from
  trace evidence, or understand an agent run from NeMo Intake. Owns the evidence
  standard that every Eval Author sub-flow
  follows. Use when the user asks "help me with my evals",
  "what's the state of the eval suite here?", "what happened in this trace?", or
  when you need to pick between the Eval Author sub-flows. Routes to a sub-flow
  and changes none of the user's source. The selected sub-flow uses the
  provider's supported tools and saves its findings under `.eval-author/`.
triggers:
  - help me build evals for my agent
  - my agent has no evals yet
  - help me with the evals in this repo
  - what is the state of the eval suite here
  - I inherited a repo with Harbor tasks in it
  - work on my evaluation suite
  - what happened in this agent trace
  - create an evaluation environment from a trace
  - which eval author step do I need
not-for:
  - eval-author-first-eval (use to establish Ethos and build first evals without prior coverage or traces)
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
- For first evals, Ethos establishes intended behavior. NOP and Oracle check
  basic task wiring and verifier behavior; the user's agent run establishes
  a baseline. Working setup does not establish evaluation quality or coverage,
  and a plan alone is not proof of runnability.
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
| `eval-author-first-eval` | Establish required Ethos, plan cases even without Harbor, and set up a small working suite while teaching the user how to run and extend it |
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

### Users with no evals

For a user asking to build their first evals, read
[`eval-author-first-eval`](../eval-author-first-eval/SKILL.md). Do not route an
explicit “no evals yet” request through discovery just to produce a missing-config
failure. The first-eval flow requires Ethos and follows
[Local Ethos](references/local-ethos.md) to save it in the repo when needed.
No platform service or upload is involved. Harbor is required for scaffolding
and execution, not for
establishing Ethos or planning cases.

If suite existence is unclear, inspect briefly: absent Harbor configuration
does not prove that no evals exist. Preserve other frameworks and standalone
tasks. For an inventory-only request, report absence and offer first-eval
authoring; do not start creating evals without that intent.

## Boundaries

These hold for every sub-flow. They exist because the repository belongs to the
user, not to you.

- **Propose, never mutate customer source.** Read the user's source and report on
  it. Do not edit, move, or reformat any of it, including its `.gitignore`. The
  evaluation artifacts you add belong under `.eval-author/`, which is theirs to
  commit or ignore. The sole additional write scope is the requested local
  `ETHOS.md`: follow [Local Ethos](references/local-ethos.md) to create or make
  user-requested edits to it in the repo. Preserve existing Ethos content and
  custom sections; downstream audit measurement does not rewrite it.
  `eval-author-discover` scripts write nothing; `eval-author-audit` writes only
  requested audit artifacts; `eval-author-task-create` writes only drafts,
  proposals, job outputs, and measurements there; `eval-author-trace-environment`
  writes only private, gitignored task workspaces there.
  `eval-author-first-eval` writes plans, drafts, configs, and jobs there. Its
  Ethos is a repository-owned document, saved and checked locally. Never upload
  it, create a Fileset, or require NeMo services or account configuration for
  local first-eval or audit authoring.
- **A missing tool is a finding, not a task.** When the provider is not installed,
  say so and stop short of proving anything. Report what you found regardless, and
  do not install the provider into the user's environment.
  For first evals, retain useful planning progress and provide installation
  guidance while leaving scaffolding and execution blocked until verification.
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

## Communicating with the user

### Onboarding and intent questions

When helping someone create their first evals, lead with what they will gain:
a starter eval set for their agent, with customer scenarios and criteria for
scoring its responses. Explain that they can rerun these evals after changes to
see whether performance improves or regresses. Call each scenario an “eval case.”
Follow the first-eval opening with a short bulleted requirements list,
explaining each requirement's purpose and observed status. Explain Ethos
in ordinary language before asking for intent: it records what the agent should
do, its boundaries, and what success means, so the evals have a target.
The opening need not call Ethos “local” or say the requested skill explains it.
Then ask a concrete question grounded in the agent's workflows. The user's
answer is part of designing the evals, not a validation verdict.

This introduction belongs in the message containing the first intent question,
even if an earlier progress message already mentioned the workflow. Preserve it
when following the local Ethos procedure: its intent questions and content
review remain part of the first-eval onboarding conversation.

An ordinary intent question gathers information needed to do the requested work.
It is not a request for permission to continue or, by itself, a blocked-work
report. Explain why the answer helps define the evals, then end with the
question. Do not append skill names, file paths, requirement quotes, or interview
mechanics just to justify asking it. Missing Ethos being actively created through
the available interview is part of onboarding; unavailable prerequisites or
required approval are separate situations. If the host explicitly requires a
skill disclosure, preserve it, but do not add one solely because a question awaits
an answer or because the interview asks questions one at a time.

Report Harbor version, importability, and readiness when those facts help the
user resolve a setup problem or understand execution. They are not the opening
headline for a first-eval conversation. Follow the first-eval skill's opening
example and explain Harbor's role before reporting its status.

### Validation and result reports

For completed discovery, validation, or execution reports, lead with the verdict
or outcome, then the evidence. This format does not apply to onboarding or an
intent question while establishing Ethos.

For dataset proposals, follow `eval-author-task-create` Step 1: lead with concrete
recommendations and their evidence, then supporting coverage counts and limits. An empty
task-generation selection does not establish that the dataset needs no changes.

State whether the findings are proven, whether the suite is ready, and the names
of the checks that failed. Never describe a suite as ready while a required check
fails, and never present an observation as proof. When a sub-flow could not reach
its provider, the validation report must say that nothing was proven. This does
not prevent a first-eval conversation from explaining the planned value and
making design progress within its prerequisite rules.

For a trace, use `success`, `failure`, or `unknown`. Tie key moments and findings
to span IDs, evaluator result IDs, or source symbols. A healthy trace doesn't
need an issue finding.
