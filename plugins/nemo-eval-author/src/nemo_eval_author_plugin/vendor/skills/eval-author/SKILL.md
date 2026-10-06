---
name: eval-author
description: >-
  Build first evals from a required Ethos, work on existing
  evaluation suites in a user's repository, derive an environment from trace
  evidence (experimental), or understand an agent run from NeMo Intake. Owns the evidence
  standard that every Eval Author sub-flow
  follows. Use when the user asks "help me with my evals",
  "what's the state of the eval suite here?", "what happened in this trace?", or
  when you need to pick between bootstrapping evals, auditing existing evals,
  finding out whether evals exist, or proposing improvements from an audit.
  Establishes local Ethos when the selected flow needs it, without changing the
  agent's implementation. The selected sub-flow uses the provider's supported
  tools and saves findings under `.eval-author/`.
triggers:
  - help me build evals for my agent
  - my agent has no evals yet
  - audit my existing evals
  - I am not sure whether this repo has evals
  - suggest new evals based on this audit
  - help me with the evals in this repo
  - what is the state of the eval suite here
  - I inherited a repo with Harbor tasks in it
  - work on my evaluation suite
  - my evals do not use Harbor
  - what happened in this agent trace
  - create an evaluation environment from a trace
  - which eval author step do I need
not-for:
  - eval-author-first-eval (use to establish Ethos and build first evals without prior coverage or traces)
  - eval-author-discover (use to run the discovery pass and get a runnable verdict)
  - eval-author-audit (use to generate, validate, measure, or aggregate audit.md coverage)
  - eval-author-task-create (use to propose dataset improvements; when task creation is requested, create and prove an eligible Gym or Harbor task)
  - eval-author-inspect-trace (use after this skill selects the trace sub-flow)
  - eval-author-trace-environment (experimental; use to derive a Harbor or native Gym task from canonical ATIF evidence)
  - eval-author-environment (use after an authoring sub-flow hands over its cases to build and prove their task environment)
  - nemo-intake (use to instrument agents, ingest telemetry, or query Intake outside Eval Author)
  - mlflow-to-atif (use to convert MLflow traces into canonical ATIF files)
  - gym-to-atif (use to convert Gym Responses traces into canonical ATIF files)
  - nemo-experimentalist (use to run insight-driven optimization end to end, which drives the Eval Author agent itself)
  - nemo-evaluator (use to run an existing benchmark rather than work on a repository's own suite)
compatibility: >-
  The core can save the local Ethos; execution is delegated to sub-flows.
  Discovery, audit, and task creation use the local checkout.
  Task execution requires the selected Gym or Harbor runtime and may require Docker and provider
  credentials. Trace inspection requires the nemo CLI, an explicit workspace,
  and read access to configured Intake.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: Read Write Grep Glob
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Eval Author

## Purpose

Work on repository-owned evaluation suites and understand agent traces. Route
each request to the narrow sub-flow that owns it.

## Choose the entry route

Choose the route before showing the onboarding checklist, gathering Ethos, or
checking Harbor. Use the requested outcome, prior answers, and supplied material;
do not ask the user to repeat a known starting point. These are alternative entry
routes, not set stages that everyone must complete:

| Starting situation and request | Route | First action and deliverable |
|---|---|---|
| The user has no evals and wants to bootstrap them | [First eval](../eval-author-first-eval/SKILL.md) | Use the authoring welcome below, then build a suite at the agreed scope through the shared milestones. The user's statement settles the eval starting point; no discovery pass is required to prove absence. |
| The user wants a coverage audit, with existing evals or a specification to guide future evals | [Audit](../eval-author-audit/SKILL.md) | Show the audit's five-step checklist, including generating and reviewing `audit-coverage-report.md`, and ask whether to begin; wait before Ethos work or repository inspection. Reuse an answered opening and prior progress on return. Explain what the evidence can establish; do not substitute a readiness report for the requested audit. |
| The user is unsure whether or where evals exist | [Discovery inventory](../eval-author-discover/SKILL.md#inventory-an-uncertain-starting-point) | Report what evals exist, what they test, and their documented run instructions before Ethos or runtime setup. Finish discovery, then ask whether the user wants an audit; do not infer that next step from a general request for help. |
| The user wants new eval recommendations based on an audit | [Dataset proposals](../eval-author-task-create/SKILL.md#step-1-propose-dataset-improvements) | Reuse the audit and its evidence for ranked proposals. A proposal-only request stops before scaffolding or execution. |
| The user wants a task derived from a recorded interaction | [Trace environment](../eval-author-trace-environment/SKILL.md) | Prepare private trace evidence and assess candidacy, then construct the selected Harbor or native Gym output. No audit gap or Harbor setup is required to start this route. |

For a fresh audit, the next deliverable is its opening checklist with descriptions
and a question that ends the turn. Load the audit skill and return that opening
before inspecting the repository, starting Ethos work, or delegating a review.
The user's audit request establishes permission to work; it does not replace the
guided conversation's initial handoff. Put it in the final response, rather than
showing it in commentary and continuing behind an asynchronous question.

Adaptation is temporarily disabled. Do not route to `eval-author-adapt` or load
its archived instructions. For an explicit request to convert existing evals to
Harbor, explain that conversion is currently unavailable. Do not silently treat
that request as first-eval authoring or treat existing material as absent.

An explicit audit or proposal request keeps that purpose even when an input is
missing: locate the missing source or follow the selected sub-flow's prerequisite
guidance, then resume that request. For an audit, confirm the trace source and
location with the user before searching for or reading traces; reuse an explicit
selection already supplied. A missing report does not mean the user has no
evals. A generic request such as “help me with my evals” with no established
starting point uses discovery inventory first. If the material is known but the
desired outcome is unclear, ask one focused question about that outcome.

Discovery inventory ends with its findings and run guidance before offering
another flow. When evals exist, ask whether the user wants them audited and wait;
“help me with my evals” alone does not select audit. With confirmed absence,
offer first-eval; with an unresolved location, explain the search limits and ask
the next source question. An inventory-only request ends with the findings.

After the user requests or accepts a next step, carry source paths, run guidance,
search limits, prior answers, and existing intent or setup evidence into that
route. First-eval starts at its earliest unfinished milestone. Reuse an explicit
request for discovery followed by audit or another flow: present the discovery
summary first, then continue the already-requested work without asking again.

Explicit readiness, repair, trace, and other supported narrow requests keep
their existing sub-flow and scope. On return visits, resume the requested flow
and reuse its completed work. Never turn an audit or recommendation request into
the full authoring experience merely because it is the user's first visit.

A request to add examples to a starter suite created by first-eval returns to
that flow's scope stage, reusing its saved plan and source selection. Review the
affected behaviors' breadth, difficulty, grading, and cost before expansion;
do not restart onboarding or require an audit unless coverage measurement is
requested. Audit-based gap creation still uses `eval-author-task-create`.

## Show the path ahead

Use this opening for a fresh authoring request after selecting first-eval.
Audit, discovery inventory, and proposal routes use their own entry
instructions. Once authoring is selected, give the opening immediately; no further
skill, reference, repository inspection, or Harbor probe is needed to show the
plan. Load supporting instructions only as the current stage needs them, from
this same local tree.

Briefly acknowledge the experience, then render this checklist with literal
`- [ ]` checkboxes. For a fresh start, mark **Understand your agent** as
**We're here** and leave all five steps unchecked. On a handoff, preserve completed
internal stages and map the current stage to its visible step using the grouping
below. Plain bullets are not the checklist. Shorten descriptions as useful, but
retain the labels:

- [ ] **Understand your agent** — establish its purpose, boundaries, and what success looks like.
- [ ] **Get the evaluation runtime ready** — explain the evaluation framework and verify the tooling.
- [ ] **Draft your first evals** — agree on examples, coverage scope, expected outcomes, and grading criteria.
- [ ] **Get the evals running** — prepare the environment, connect the agent, and validate the setup.
- [ ] **Run and review** — evaluate the agent, explain results, and show how to rerun.

The five visible steps group the existing ten internal stages:

| Visible step | Internal stages, in their existing order |
| --- | --- |
| Understand your agent | 1. Establish the agent’s Ethos |
| Get the evaluation runtime ready | 2. Get the evaluation runtime ready |
| Draft your first evals | 3. Understand the evaluation starting point → 4. Define the evaluation scope → 5. Prepare cases and grading |
| Get the evals running | 6. Prepare the execution environment → 7. Connect the agent → 8. Validate the evals |
| Run and review | 9. Evaluate the agent → 10. Review results and explain reruns |

Render only the five visible steps in the user checklist. This grouping changes
presentation only: preserve every internal stage's order, work, check-ins, and
authorization requirements. Mark a visible step `[x]` only when all its internal
stages are complete; mark the step containing the current internal stage
**We're here**. Reuse a starting point already settled by the user or discovery;
first-eval also settles which traces, if any, to use before completing that
internal stage. This does not complete **Draft your first evals**.

End with “Ready to get started?” and wait. Do not follow the opening with tool
calls in the same turn. After acceptance, read
[Milestone check-ins](references/milestone-checkins.md) for stage transitions,
progress, and resumption; begin at the earliest unfinished milestone, normally
Ethos. The opening answer does not complete that stage or waive its check-in.

On return visits, reuse an answered opening and resume the saved stage. Narrow
inventory, readiness, audit, and trace requests use their scoped sub-flow instead
of this onboarding experience. If onboarding later finds an existing Harbor suite,
tailor the remaining work to that suite without restarting the completed stages.

A report that a downstream model trusts has to be right. A plausible report is
worse than no report when somebody acts on it.

## The standard

**Every fact you record comes from authoritative evidence, not a guess.**

The authority depends on the sub-flow:

- For suite discovery, the selected provider's validators and execution checks
  judge readiness: Gym for native manifests and Harbor for its configs and tasks.
  File presence alone does not prove runnability.
- For first evals, Ethos establishes intended behavior. Gym's positive/negative
  controls or Harbor's NOP and Oracle check basic task wiring and verifier
  behavior; the user's agent run establishes a baseline. Working setup does not
  establish evaluation quality or coverage,
  and a plan alone is not proof of runnability.
- For task environments, the environment's Harbor smoke runs or Gym native
  controls establish that its dependencies, starting data, reset, and end-state
  export work. Build files and an environment plan are not that proof.
- For audit-spec validation, the bundled schema and validator judge the finite
  `audit.md` coverage denominator.
- For task creation, Gym's positive/negative controls and native execution, or
  Harbor's Oracle, check task behavior. Measured ATIF proves whether repeated
  real-agent runs close the selected gap; reference controls do not measure agent ability.
- For trace inspection, Intake establishes what happened. Local source code can
  explain behavior, but it can't replace recorded trace evidence.
- For experimental trace-derived environments, canonical ATIF establishes the request.
  Harbor's NOP and Oracle runs or Gym's native validation and repeated controls
  establish the generated task's behavior within the reported proof limits.

No sub-flow reimplements a provider's rules. When evidence can't settle a claim,
the report marks the claim unproven or uncertain.

Before executing, recovering, or reporting evaluation runs, follow
[Execution recovery](references/execution-recovery.md). Preserve the selected
execution path and original verifiers; diagnostic results cannot complete the
requested evaluation.

## Vocabulary

The sub-flows share this language, and reports use it verbatim.

| Term | Meaning |
|---|---|
| Check | One named result: `pass`, `warn`, or `fail`. Carries a message and, when it fails, a hint |
| Required | A failing required check blocks the suite. Report the suite as not ready |
| Advisory | A warning worth surfacing that blocks nothing |
| Rung | One step of a provider's validation ladder, ordered so a lower rung's failure often clears once a higher one is fixed |
| Proven | A provider judged this check. An unproven check is an observation and never evidence |
| Provider | The evaluation framework that owns the rules. Gym and Harbor are supported and compatible; follow the shared provider-selection rule |
| Finding | One trace claim categorized as `behavior`, `issue`, `recovery`, or `uncertainty`, with evidence IDs |
| Outcome | The trace assessment: `success`, `failure`, or `unknown` |

## Sub-flows

Read the selected sub-flow's own `SKILL.md` when its stage begins and follow it.
This file carries the standard and boundaries; the sub-flow carries the steps.

| Sub-flow | Use it to |
|---|---|
| [`ethos`](../ethos/SKILL.md) | Explore intended behavior and write or update ETHOS.md through the [Local Ethos](references/local-ethos.md) handoff |
| `eval-author-first-eval` | Establish required Ethos, plan cases without an installed runtime, and build or expand a starter suite at the agreed breadth while teaching the user how to run it |
| `eval-author-discover` | Establish whether a repository's evaluations run, name the rung that fails, and get the exact command to run them |
| `eval-author-audit` | Generate and validate a finite `audit.md` coverage denominator, measure and aggregate trace coverage, and report findings |
| `eval-author-task-create` | Propose concrete dataset improvements from audit findings; when task creation is requested, create one eligible Gym or Harbor task and prove it with native controls and repeated measured runs |
| `eval-author-inspect-trace` | Understand one Intake trace without presuming that the trace contains a failure. Not user-invocable; this skill selects it |
| `eval-author-trace-environment` | **Experimental.** Normalize one trace to ATIF, make a privacy-reviewed candidate decision, and build a private Harbor or native Gym task when evidence supports it |
| `eval-author-environment` | Build and prove the environment a Harbor or Gym task runs in: provider fit, how each dependency is provided, starting data and reset, end-state export, and isolation. Not user-invocable; first-eval and task-create hand it their agreed cases |

`eval-author-audit` works one level above tasks: it generates and validates the
coverage denominator, measures traces against it, and aggregates deterministic
coverage reports. `eval-author-task-create` owns the proposal step: prioritize
dataset improvements from those findings across tools, capabilities, and failure
cases while preserving their measured or unmeasured status. Proposal-only
requests stop there. Its subsequent task-creation path consumes only actionable
tool gaps and uses the selected Gym or Harbor native scaffolder rather than guessing a task
layout. For a requested full workflow, proceed from audit to proposals even when
there are no eligible tool gaps; an audit-only request ends with the findings.

## Select the evaluation provider

Use the same rule for first-eval, audit-derived task creation, and trace-derived
task creation: honor the user's explicit Gym or Harbor choice; otherwise preserve
the existing suite's provider. When neither is specified, default to Harbor.
Both providers are supported, and Gym is compatible with Harbor. If the user
selects a setup using both, record which provider owns each artifact and execution
step and verify the runtimes that setup uses. For a mixed suite, preserve the
setup of the affected cases; clarify only when the target is ambiguous. The user
has the final say and can override the default or request a provider change.

Select the output provider independently of the evidence source: a Gym trace
does not require Gym output. Use the selected provider's native scaffolding,
validation, and execution path. Missing setup is a reported prerequisite, not
permission to silently switch providers.

## Establish Ethos before authoring

First-eval requires an applicable `ETHOS.md` before case design,
new scoring decisions, or task authoring. Read [Local Ethos](references/local-ethos.md)
for the shared locate, reuse, intent, creation, validation, and review procedure.
Agent documentation can inform Ethos before any evaluation inventory. Existing
evals establish what is currently tested; they do not replace intended behavior.

## Gather requirements and setup from evidence

During discovery and authoring source inspection, read and follow
[Requirements and setup evidence](references/authoring-context.md#gather-requirements-and-setup-from-evidence).
It defines the requirements to gather, their source and verification status,
and where to record unknowns. Reuse documentation and prior answers; ask only
about missing facts affecting the next work. Never request secret values.

## Route after the evaluation starting point

The entry routes above own discovery performed before authoring. Within an
authoring flow, the shared milestone procedure establishes Ethos and the selected runtime,
then reuses the settled starting point or calls `eval-author-discover` if source
selection is still needed. Its runtime probes can be used separately during the
runtime stage. Reuse completed checks and answers when their inputs are unchanged.

When a discovery scan found no Harbor evals, follow its **When no Harbor evals
were found** conversation before routing candidate material; an empty Harbor
scan does not establish that no evals exist.
An explicitly supplied source already answers source selection. Once the evaluation
starting point is settled, carry the established Ethos, runtime invocation, findings,
and answered check-ins into the selected path:

- **Existing Gym or Harbor suite or tasks:** continue discovery and resolve configuration
  or readiness failures. Broken evals are not a reason to start over.
- **Other evaluation material:** report what exists and its documented run
  instructions or missing execution details. Preserve the user's selected source
  and requested outcome. Conversion is currently unavailable; do not enter the
  disabled adaptation flow. If the desired outcome is unclear, ask one focused
  question before continuing with a supported flow.
- **Confirmed absence of evals:** read
  [`eval-author-first-eval`](../eval-author-first-eval/SKILL.md) and continue at
  **Define the evaluation scope** once the authoring prerequisites above are
  complete. A missing config does not block this starter flow; do not substitute
  audit-gap task creation.
- **Unknown or inaccessible evals:** ask for the location or an accessible
  representative case and its scoring. An access failure does not prove absence.

An inventory-only request ends with findings and an offered next step. Discovery
alone does not authorize creating or running evals.

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
  `eval-author-first-eval` writes plans, drafts, configs, and jobs there and uses
  the narrow local Ethos write exception above. `eval-author-environment` writes
  environment kits, generated data, and smoke tasks under
  `.eval-author/environments/`, copying any repository file it needs.
- **A missing tool is a finding, not a task.** When the provider is not installed,
  say so and stop short of proving anything. Report what you found regardless, and
  do not install the provider automatically. For Gym, report its separate runtime
  requirements and repository setup instructions; for Harbor, use discovery's
  setup guidance to explain installation and verification. An explicit request to install is
  authorization for that setup; a broad eval request alone is not.
- **Do not run without approval.** Discovery proves an existing suite can run and
  hands over the command. Task creation may run Gym verifier controls or Harbor Oracle locally, then starts
  real-agent jobs only when the user explicitly asked for or approved that spend.
  First-eval authoring may run local task sanity checks when task creation is requested;
  real-agent or paid-judge runs require authorization for that execution and spend.
- **Trusted repositories only.** Validating a config can execute repository code,
  because an agent named by import path gets imported. If the repository is not
  trusted, say so and stop.
- **Intake reads are narrow.** Only `eval-author-inspect-trace` reads Intake. It
  uses read-only `nemo intake` commands against the configured instance and
  workspace. No sub-flow discovers accounts, ingests data, uploads files, or
  changes a remote resource.

## Communicating with the user

Do not append a footer that names or links a skill, quotes its instructions, or
explains that a skill requires you to ask or wait, unless the host explicitly
requires that disclosure. Keep the questions, check-ins, waits, and authorization
requirements unchanged.

### Explain the eval pieces as they become relevant

Before asking the user to make an authoring or setup decision, read
[Explaining evaluation components](references/authoring-context.md#explain-the-eval-pieces-as-they-become-relevant).
Use its definitions for the pieces relevant now, grounded in inspected material
and the user's answers. Explain unfamiliar grading terms before asking about them;
keep task grading distinct from coverage auditing.

### Validation and result reports

For completed discovery, validation, or execution reports, lead with the verdict
or outcome, then the evidence. This format does not apply to onboarding or an
intent question while establishing Ethos.

For dataset proposals, follow `eval-author-task-create` Step 1: lead with concrete
recommendations and their evidence, then supporting coverage counts and limits. An empty
task-generation selection does not establish that the dataset needs no changes.

In validation and execution reports, state whether findings are proven, whether
the suite is ready, and the checks that failed. Never describe a suite as ready while a required check
fails, and never present an observation as proof. When a sub-flow could not reach
its provider, its validation report must state that readiness was not proven.
This does not block the empty-scan conversation or first-eval Ethos and case
planning within their prerequisite rules.

For a trace, use `success`, `failure`, or `unknown`. Tie key moments and findings
to span IDs, evaluator result IDs, or source symbols. A healthy trace doesn't
need an issue finding.

## Prerequisites

Routing needs the user's desired outcome and any supplied repository or trace
location, not Harbor or credentials. Authoring needs local Ethos; runtime
dependencies belong to the selected flow.

## Limitations

The router does not execute evals or establish readiness itself. Discovery is
not an audit, a coverage gap is not proof of an agent defect, and trace-derived
environments remain experimental. Existing-eval adaptation is disabled.

## Troubleshooting

- Unclear desired outcome: ask one focused question; reuse known starting facts.
- Missing Ethos: follow [Ethos recovery](references/local-ethos.md#recovery) when the
  selected flow requires it; inventory and trace inspection can proceed without it.
- Missing provider or inaccessible source: report the blocked claim and use the
  selected flow's recovery; an access failure never proves that evals are absent.

## Examples

```text
Request: "Audit the evals in ./evals; the agent Ethos is ./ETHOS.md."
Route: eval-author-audit, carrying both paths into its pre-flight.
Deliverable: coverage findings and their evidence, not a readiness-only report.

Request: "I don't know whether this repo has agent evals."
Route: eval-author-discover's inventory entry, without runtime probes.
Deliverable: candidate paths, documented run commands, and search limits.
```
