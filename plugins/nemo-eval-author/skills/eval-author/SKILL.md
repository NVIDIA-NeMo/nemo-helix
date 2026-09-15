---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: eval-author
description: >-
  Build first evals from a required Ethos, work on existing evaluation suites
  in a user's repository, adapt non-Harbor evals into Harbor, or derive an environment from
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
  - my evals do not use Harbor
  - what happened in this agent trace
  - create an evaluation environment from a trace
  - which eval author step do I need
not-for:
  - eval-author-first-eval (use to establish Ethos and build first evals without prior coverage or traces)
  - eval-author-discover (use to run the discovery pass and get a runnable verdict)
  - eval-author-adapt (use to adapt existing non-Harbor evaluations after discovery)
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
- For adapting existing evals, the original cases, fixtures, and scoring rules
  establish what to preserve. Harbor validation and controlled task runs prove
  the new wiring; they do not by themselves prove scoring equivalence.
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
| `eval-author-adapt` | Convert existing non-Harbor material into task files, implement supported checks, and guide validation and execution when available |
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

## Show the path ahead

At the start of repository onboarding, lead with the Harbor introduction in
Communicating with the user, including its documentation link. Then show this checklist so the
user knows what getting their evals working involves:

- [ ] **Find your starting point** — existing evals, test cases, reports, examples, and documentation.
- [ ] **Understand what success means** — requirements, expected behavior, and how results should be graded.
- [ ] **Understand the setup** — agent connection, software and services, dependencies, supported operating systems, licenses, credentials, and starting/reset conditions.
- [ ] **Prepare your Harbor evals** — turn the available material into tasks, grading checks, and environment configuration.
- [ ] **Run and verify** — check that the setup and grading work, then evaluate the agent.
- [ ] **Review the results** — explain outcomes, remaining gaps, and how to rerun.

Use the checklist to orient the conversation. At each stage, explain what you
learned, why it matters, and the next decision or action in ordinary language.
Show one current stage; later stages remain ahead even if some background
inspection overlaps. Merely listing statuses does not guide the user.

Source selection establishes which evals the user has. The adaptation introduction
then explains Harbor and asks whether they want to convert those evals. Wait for
each needed answer; do not treat the checklist as an instruction to complete every
stage without the user's input. Other milestones need no automatic approval.
Tailor the preparation step after
discovery: reuse or fix Harbor evals, adapt other material, or use the available
first-eval flow. For narrower inventory, audit, or trace requests, show only the
steps for that outcome; a checklist must not expand the requested work.

Show the checklist when a user returns, using saved findings and inspection of
the current artifacts to establish where they are. Reuse answers and completed
work when still applicable; do not restart onboarding or assume old readiness
evidence still applies after inputs change. Refresh it at meaningful milestones
and in the final handoff, rather than with every tool call. Sub-flows carry
forward the same checklist instead of starting their own.

Use `[x]` only for completed milestones. Mark the current item **We're here**
and explain what you are working out together. Leave future items unchecked;
name a specific missing prerequisite only when it affects the work underway.
An ordinary onboarding question does not need a **Needs input** status.
Keep partial work unchecked
and name its scope: task files may exist while grading or configuration remains
unfinished. Identifying a dependency does not prove it is available, and loading
task files does not prove an agent run works. A completed run may have low scores.
Follow the checklist with the next action and any specific input needed from the
user. Continue independent authorized work while other items need input.

## Gather requirements and setup from evidence

During discovery and source inspection, read relevant repository and supplied
documentation before asking the user to reconstruct it. Follow references to
requirements, rubrics, setup guides, dependency manifests and lockfiles, agent
configs, runner scripts, and license or access instructions. Gather what applies:

- **Purpose and success:** scenarios, expected behavior, grading rules, and relevant constraints.
- **Inputs and documentation:** source cases, fixtures, example outputs, and instructions for using them.
- **Execution:** the agent and how to invoke it, required software/services and versions, OS, hardware, and where each dependency runs.
- **Access and licenses:** documented installation and execution requirements, license provisioning, required accounts and credential variable names. Do not request secret values in chat or copy them into reports.
- **Repeatability:** starting data/state, session handling, reset procedure, and how results reach the grader.

Distinguish documented requirements, user-confirmed information, verified
availability, and unknowns. Cite the source and record what an unresolved item
blocks: task preparation, a particular grading check, or live execution. A repo
license does not establish the license or availability of its dependencies.
Ask focused questions only about missing facts that affect the next work, and
reuse earlier answers. Missing access or license provisioning can leave execution
pending while task files and independent checks proceed.

Keep the gathered requirements, progress, and next action in the selected
sub-flow's existing human-readable findings or task README under `.eval-author/`.
Do not overwrite generated evidence reports or require a new intake document
before conversion. For read-only requests, explain findings in the reply within
that sub-flow's reporting boundaries.

## Start with discovery

For repository eval onboarding, start with `eval-author-discover` for every user,
including users who say they have no evals or already name another framework.
Reuse a completed discovery pass from this session when its repository and inputs
are unchanged. Do not repeat questions the user has already answered. Explicit
audit, task-gap, and trace requests still use their own sub-flows.

Discovery's empty-scan conversation introduces Harbor as the framework Eval
Author uses to run eval cases and score results, then asks whether the user has
evals in any form and where to find them. An empty Harbor scan is not proof that
the user has no evals. Inspection can identify candidates, but only the user can
settle whether those are the evals they want to work from. When this has not
already been answered, explain what you found and ask before choosing a case,
extracting the full collection, or creating adapted tasks. For example, when
the inspected material supports it:

> I didn't find Harbor evals in the locations I checked. I did find reports with
> example conversations and written checks, which look like possible eval material.
> Are these the evals you'd like us to work from, or do you have evals somewhere
> else? If you don't have existing evals, we can start with your first one.

Keep **Find your starting point** current while awaiting that answer. Save the
discovery findings, but wait to begin conversion. A broad “Help me get my evals
working” does not identify discovered candidates as the user's intended source.
If the user already identified the material as their evals or explicitly asked
to convert it, reuse that source answer without asking where their evals are again. Saying that
evals exist without giving their location still requires locating them. Follow
the answer:

- **Harbor suite or standalone Harbor tasks:** continue discovery and resolve
  the missing configuration or failed checks. Broken Harbor evals are existing
  evals, not a reason to start over.
- **The user identifies existing evals in another form:** read
  [`eval-author-adapt`](../eval-author-adapt/SKILL.md). Use the supplied path or
  artifact, including tests, scripts, notebooks, datasets, and manual rubrics.
  If Harbor and other evals coexist, preserve both and follow the user's chosen
  suite; do not silently ignore their request to adapt the non-Harbor evals.
  A conversion request should produce Harbor task files from the available
  material. App access and runtime verification can remain pending; they must
  not redirect task creation into a specification-only interview. Written
  criteria from reviews are usable inputs even if the evals have never run.
  Once the source is identified, follow adaptation Step 1: explain Harbor, what
  conversion will create, and what execution will require, then ask whether the
  user wants to proceed. “Use these reports” settles the source; it does not
  answer this informed choice. Wait before detailed requirements gathering,
  choosing a case, bulk extraction, or task creation. Reuse an earlier acceptance
  of this explained conversion path rather than asking again.
  If the user points elsewhere, inspect that location and use those evals.
  Let the adaptation flow choose a representative
  case unless there is a material scope ambiguity; do not end at a scenario menu.
- **The user confirms there are no existing evals:** use `eval-author-first-eval`
  for the guided intent-to-starter-eval experience. Carry forward discovery and
  answers already given; a missing-config finding must not block authoring.
  This flow requires Ethos and follows [Local Ethos](references/local-ethos.md)
  to save and review it in the repo when needed. No platform service or upload
  is involved. Harbor is needed for scaffolding and execution; Ethos and case
  planning can proceed without it. Do not substitute audit-gap task creation.
- **Unknown or inaccessible evals:** ask for the location or a representative
  case and how it is scored. Do not treat an unanswered question, unreadable
  path, or discovery error as confirmation that no evals exist.

For an inventory-only request, report what was found and offer the appropriate
next step. Creating or running evals requires that intent; discovery alone does
not authorize either.

## Continue from adaptation to coverage auditing

When the user indicates they are done with their adapted evals and task files
exist, use `eval-author-adapt` Step 6 to offer a coverage audit. Explain that it
compares the agent's Ethos with the created evals to identify what is addressed
and what is missing. Wait for acceptance unless the user already requested it.
Task creation alone does not trigger an audit, and declining does not invalidate
the adaptation work. Carry forward the user's choice and existing artifacts.

On acceptance, route to `eval-author-audit` and its Ethos pre-flight, then its
adapted-eval coverage review. Keep the task scope and any run evidence explicit;
drafts can support a review of planned coverage even if live execution is pending.
An audit-only request ends with the report. Proposing or creating further tasks
remains a separate requested continuation.

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
  `eval-author-adapt` writes its mapping, task copies, configs, and job outputs
  under `.eval-author/`; the original evals remain the source of truth.
  `eval-author-first-eval` writes plans, drafts, configs, and jobs there. Its
  Ethos is a repository-owned document, saved and checked locally. Never upload
  it, create a Fileset, or require NeMo services or account configuration for
  local first-eval or audit authoring.
- **A missing tool is a finding, not a task.** When the provider is not installed,
  say so and stop short of proving anything. Report what you found regardless, and
  do not install the provider automatically. Use discovery's Harbor setup guidance
  to explain installation and verification. An explicit request to install is
  authorization for that setup; a broad eval request alone is not.
  Adapting evals can still proceed through source inspection and mapping without
  Harbor; scaffolding and execution remain unproven until it is available.
  First-eval Ethos and case planning can also proceed before Harbor is available.
- **Do not run without approval.** Discovery proves an existing suite can run and
  hands over the command. Task creation may run Oracle locally, then starts
  real-agent jobs only when the user explicitly asked for or approved that spend.
  Adapting evals may run local task sanity checks when task creation is requested;
  real-agent or paid-judge runs require authorization for that execution and spend.
- **Trusted repositories only.** Validating a config can execute repository code,
  because an agent named by import path gets imported. If the repository is not
  trusted, say so and stop.
- **Intake reads are narrow.** Only `eval-author-inspect-trace` reads Intake. It
  uses read-only `nemo intake` commands against the configured instance and
  workspace. No sub-flow discovers accounts, ingests data, uploads files, or
  changes a remote resource.

## Communicating with the user

### Discovery and adaptation onboarding

For validation and execution reports, lead with the verdict or outcome, then the
evidence. First-time onboarding has a different opening: explain that this skill
uses Harbor to run evals before presenting readiness or installation findings.
Include this explanation in the first substantive discovery reply, even if an
earlier progress message introduced Harbor. For example:

> Eval Author uses [Harbor](https://www.harborframework.com/docs) to run evals.
> It gives your agent a task, checks the result, and lets you repeat the same
> test after changes to see how your agent is doing.

Make clear that Harbor is needed for execution; users can bring existing work
in other formats. Then explain what you found and why it looks useful in plain
language. For example, describe “conversations and descriptions of what a good
answer should look like” before discussing assertions, scores, or file formats.
Keep the discovery scope accurate: say no Harbor evals were found in the locations
checked, rather than asserting the whole repository has none.

Keep that explanation together as its own opening paragraph. Successful tool
installation is usually incidental to the source conversation; keep it in the
saved findings. If a missing tool affects the next step, explain the effect
briefly. When no Harbor evals were found, the source question does not need an
additional “readiness remains unproven” verdict. Describe possible source material
without treating its historical scores as verified results.

Show the shared checklist, keeping **Find your starting point** current until
the source is settled. Explain the immediate purpose: “Let's first make sure
we're starting with the right material.” Follow with the source question.
Put artifact links beside what they contain or after the explanation; an
installation verdict, report link, or technical inventory should not replace
the welcome and guidance. Tool versions and scan mechanics belong in saved
findings unless needed to resolve a problem. Introduce Harbor filenames when
they become useful. The later adaptation introduction explains the proposed
conversion in more detail using the source the user has identified.
Once the user has identified the eval source, follow `eval-author-adapt` Step 1
to explain the conversion and ask whether to proceed. Begin conversion after that
answer. Before source selection, use the discovery conversation. Use the task-explanation guidance
when files have been created. Ask a question only when its answer is needed;
an inventory and a scenario menu are not a completed conversion.

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

In validation and execution reports, state whether findings are proven, whether
the suite is ready, and the checks that failed. Never describe a suite as ready while a required check
fails, and never present an observation as proof. When a sub-flow could not reach
its provider, its validation report must state that readiness was not proven.
This does not block the empty-scan conversation, source-grounded adaptation
planning, or first-eval Ethos and case planning within their prerequisite rules.

For a trace, use `success`, `failure`, or `unknown`. Tie key moments and findings
to span IDs, evaluator result IDs, or source symbols. A healthy trace doesn't
need an issue finding.
