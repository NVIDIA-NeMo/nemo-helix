---
name: eval-author-first-eval
description: >-
  Establish a required Ethos, plan evaluation cases, and build first evals or
  expand a starter Gym or Harbor suite while explaining
  its parts. No prior traces or coverage reports are required. Missing selected
  runtime blocks scaffolding and execution, not planning.
triggers:
  - help me build my first evals
  - my agent has no evals yet
  - expand the starter eval suite we created
  - create an evaluation suite from scratch
not-for:
  - eval-author (use for the shared standard and routing)
  - eval-author-task-create (use for measured audit coverage gaps)
  - eval-author-discover (use to check an existing suite)
compatibility: >-
  Ethos is required before evaluation design and is saved and checked locally
  in the user's repository. No NeMo service, account, platform CLI, or upload is needed.
  Planning needs no evaluation runtime. Native Gym authoring needs Gym v0.6.0+
  in its separate Python 3.13.14+ environment. Harbor authoring needs its CLI
  and Python environment. Execution may require Docker,
  an agent adapter, and provider credentials.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: Bash Read Write Grep Glob
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Eval Author: first eval

## Purpose

Read `eval-author` for the shared standard and boundaries. Follow
[Milestone check-ins](../eval-author/references/milestone-checkins.md) throughout
these ten internal stages; that procedure owns transitions, progress, and user
check-ins. Display progress using the core's five-step checklist.
For a direct invocation, use the core's first-eval entry route and authoring
opening before repository work. The user's statement that there are no evals is
enough to select this flow. When routed here from discovery inventory, carry its
findings and prior answers forward, then start at the earliest unfinished
milestone; inventory alone does not establish Ethos or selected-runtime readiness. Enter
stage 4 only when the first three milestones and applicable check-ins are settled.
Do not repeat completed work or restart an existing intent interview.

When the user asks to expand a starter suite created here, resume at scope
planning for the affected behaviors. Reuse the saved plan, corpus, Ethos, setup,
and accepted cases; recheck prerequisites only where the change affects them.
Do not restart onboarding or require an audit merely to add agreed examples.
Requested coverage measurement still belongs to `eval-author-audit`.

No evals is a normal starting state. Do not require a previous run or manufacture
a coverage report to enter `eval-author-task-create`. The initial deliverable is
a working suite at the agreed breadth that the user understands and can rerun.
A minimal pilot is acceptable when explicitly chosen; comprehensive coverage,
repeated agent success, and trace-driven optimization are not prerequisites.
Include difficult cases when relevant to the agreed behavior and purpose.
A working suite and strong evaluation quality are separate claims.

## 1. Establish the agent’s Ethos

Read and follow [Local Ethos](../eval-author/references/local-ethos.md) using the
agent's documentation and intended behavior; this does not depend on discovering
existing evals. Carry its result into the later case plan: the applicable checked
Ethos path, established intent, completed content review when needed, and any
unresolved prerequisite.

## 2. Get the evaluation runtime ready

Follow the core's [provider-selection rule](../eval-author/SKILL.md#select-the-evaluation-provider),
as audit-derived and trace-derived creation do. Both Gym and Harbor are supported.
Follow the shared milestone procedure to explain the selected setup and verify
the runtimes it uses.

## 3. Understand the evaluation starting point

Reuse the user's stated absence of evals or discovery's confirmed starting point,
along with available agent and documentation findings. Do not run discovery or
ask whether evals exist again solely to complete this milestone. Explain the
outcome concretely: a starter eval set of customer scenarios
with criteria for scoring responses, which the user can rerun after changes to
detect improvement or regression.

Read [Corpus and coverage planning](../eval-author/references/coverage-planning.md#select-trace-evidence-before-using-it)
before trace search or inspection. Present already-known source candidates and
confirm the intended source and corpus, or reuse the user's explicit selection.
Support proceeding without traces. Knowing there are no evals does not settle
the trace choice. Carry the selected paths, bounds, exclusions, and inspected
counts into the plan; repository examples must not become the full corpus by
default. Save this selection and current progress in `.eval-author/first-eval.md`
as soon as established, even before the case plan is complete.

Use “starter eval set” for the collection and “eval case” for each scenario.
Introduce a case as a request with criteria for scoring the response. Reserve
“sanity checks” for validation of the cases themselves; use Harbor's technical
term “task” when discussing its files or CLI.

## 4. Define the evaluation scope

Read the agent entry point, tool definitions, and usage docs to understand how
the actual agent runs. Follow [Corpus and coverage planning](../eval-author/references/coverage-planning.md)
to propose breadth per behavior from Ethos and the selected evidence. Explain
a minimal pilot or broader coverage in terms of distinct examples, meaningful
variations and difficulty, positive and negative scenarios, and remaining gaps.
Include the request, initial fixture, expected observable outcome, a meaningful
failure example, and the Ethos requirement each case tests. Ask only for intent
or invocation details not already established. Do not make the user supply
Harbor YAML or choose a framework.

When agent or setup evidence points to software or state outside the agent
process, use [Execution dependencies](../eval-author/references/execution-dependencies.md)
to establish the selected cases' runtime and result-collection requirements.
Record the agent's provider fit and dependency inventory using Steps 1 and 2 of
[the environment sub-flow](../eval-author-environment/SKILL.md); files and
embedded databases the agent works on count as dependencies. The kit is built
and proven at stage 5, before any test. Raise a poor fit for the provider selected
at stage 2 in the scope checkpoint; if the user changes provider, verify that
runtime as stage 2 describes before preparing cases.

Save `.eval-author/first-eval.md` with the Ethos path and requirement references,
selected corpus (or no-trace choice), per-behavior scope and example counts,
outcomes and grading, priorities, exclusions, task/attempt mapping, cost
implications, agent invocation, prerequisites, and unresolved questions.
Separate proposed choices from the user's agreement and record subsequent feedback.
This is an **evaluation plan**, not a coverage report
or runnable evals.

Use reproducible inputs and variations supported by Ethos and the agreed scope,
including semantically justified difficult negatives where relevant. Grade an
observable result rather than a specific tool call or exact prose. Simple
assertions are acceptable if their limits are clear.
Do not replace subjective quality with brittle string
matching or mock away the behavior under test. If the requested case cannot be
tested with available resources, retain the original requirement as blocked.
Explain the limitation; select a supported alternative only with the user,
and record it as a scope change rather than completion of the original case.

Without the selected runtime, the deliverable at this point is the evaluation
plan; native task creation remains blocked on that runtime's setup.

Present the plan using the shared [scope checkpoint](../eval-author/references/milestone-checkins.md#scope-checkpoint).
Settle unresolved scope choices and incorporate changes before preparing cases.
When expanding, record retained and added examples and the new unique totals;
the pilot's size does not determine the expansion's scope.

## 5. Prepare cases and grading

Apply [Task validation and execution evidence](../eval-author/references/task-validation.md)
for requirement-to-check review, declared verifier controls, and revision-bound
run receipts. Include a realistic incorrect result and applicable valid-alternative
and side-effect cases. These are local task controls, not extra model repeats.

Read the reference for each provider in the selected setup for the runtime-specific
work in stages 5–9:

- [Native Gym first evals](references/gym-first-eval.md): native scaffolding,
  component configuration, verifier fixtures, and Gym execution.
- [Harbor first evals](references/harbor-first-eval.md): native task files,
  environment and agent setup, NOP/Oracle controls, and Harbor execution.

Keep the shared scope, stage check-ins, explanations, suite README, and result
reconciliation here. A combined setup uses the relevant parts of both references;
reading a reference does not change the selected provider.

Build and prove the kit before the tests. Before writing any case's verifier,
complete Steps 3 to 7 of [`eval-author-environment`](../eval-author-environment/SKILL.md):
choose how each dependency is provided, generate the kit's seeded starting data,
make its end state readable, set its isolation, and run its smoke task, recording
each job in `environment-plan.md`. Write each case's own records on top of that
data, and derive its expected values with an independent reference query over
the generated data rather than by hand. Hand-typed fixtures of a few rows are
not starting data.

Create the agreed examples and map their stable IDs to native tasks or dataset
rows. Keep source provenance and expected-label rationale in the plan or
verifier-only data, outside the agent's starting environment. Start with one
representative case and check its available parts before reusing the pattern.
If its live environment or agent connection is pending, continue independent
case and grading work for the other selected cases.

Before running, document each case's reward format, metric names and ranges,
and expected control results. Derive these criteria from the intended outcome.
Add a task README with the Ethos requirement, fixtures, verifier, and run commands.
Also create the suite entry point `.eval-author/README.md` following
[Suite review and rerun instructions](../eval-author/references/suite-readme.md).
Present its case inventory and link the guide at this milestone, with input,
expected outcome, grading, and available source evidence; mark new cases as not
run until executed. Keep it current as the environment, agent connection, and run
configuration are completed and fresh control or agent results become available.

Apply the core's **Explain the eval pieces as they become relevant** guidance to
the files being created. For each verifier, explain its actual assertion and a
concrete incorrect example it must reject. Fix demonstrated false acceptance
before claiming that check is validated; retain untested distinctions as explicit
limitations. Introduce the provider's reference solution or known-correct fixture
when creating it. Keep this teaching part of building the starter cases, without
a separate tutorial or exhaustive methodology exercise.

Before the checkpoint question, introduce Harbor's case viewer in the reply and
walk the user to one created case's input and grading files, following
[Show the user how to review](../eval-author/references/suite-readme.md#show-the-user-how-to-review).
Include how to launch and open it; a link to the suite guide alone is insufficient.

## 6. Prepare the execution environment

Hand off the kit proven at stage 5 with Step 8 of
[`eval-author-environment`](../eval-author-environment/SKILL.md), and use the
selected provider reference to confirm each task's environment builds on that
kit; rerun the smoke task if the kit changed after it passed. Keep solutions and
verifier-only data outside the agent's initial environment. This milestone is complete only
when the kit's `environment-plan.md`, seeded data generator, and smoke results
exist under `.eval-author/environments/<agent-slug>/`; the tasks' own controls
never count as environment proof, and without the smoke task's own jobs the
plan's status stays `unproven`.
Verify the selected backend and required application access before declaring
this milestone complete; identifying their requirements is only partial progress.
Record unavailable access and which cases or checks it blocks, naming credential
variables without recording secret values.

## 7. Connect the agent

Use the selected provider reference to establish a supported integration for the
actual agent from its entry point, installed interfaces, registry, and CLI help.
Do not substitute another agent or rewrite the application. If integration is
missing, identify the specific connection required. Verifier-control validation
remains available, but performance of the user's agent remains unmeasured.

Record the selected cases, agent and model settings, agreed attempts, reset
procedure, and output paths using the provider's configuration format. Propose
one attempt for an initial baseline unless the purpose needs repeats; record
that choice and its cost during scope planning. Repeats do not add examples.
Do not include unrelated drafts merely because they share a parent directory.
Resolve paths from the repository root and reference credential environment
variables rather than embedding secrets. Configure request delivery, required
conversation state, and collection of outputs and actions through supported
interfaces. Record connection checks under **Validate the evals**; configuration
alone does not prove that the connection works.

## 8. Validate the evals

Follow [Execution recovery](../eval-author/references/execution-recovery.md)
for control and target-agent runs, compatibility repairs, and all result recaps.
Confirm the current environment kit's proof (Harbor's smoke task or Gym's
persistence and reset cases) passed first; a proof failure is an environment
defect to fix, not a task or agent result. That proof is recorded in the kit's
plan, not with the task evidence recorder below.

Use the selected provider's native validation and controls, retaining fresh
artifacts on reruns. Require checks to complete without exceptions and meet the
case's predeclared reward criteria, including reward shape. When inaction is
correct, also exercise an explicitly incorrect response or action: a successful
no-action control alone does not show that incorrect behavior is rejected.

Run the native commands through the shared evidence recorder, along with the
other declared controls. For each case, inspect trial results and rewards.
These are basic wiring and verifier sanity checks, not evidence of broad coverage
or a robust benchmark. Keep the declared controls proportional to the task; do
not require repeated proof or additional model runs as an onboarding gate.
Investigate obvious unconditional rewards or leaked answers. Fix broken cases
and rerun affected checks; do not weaken the intended assertion to force a pass.
Report blocked cases separately and deliver the working subset without silently
dropping planned cases or claiming the whole suite passed. Keep unfinished
configuration or connection validation visible separately from successful
verifier controls.

## 9. Evaluate the agent

Confirm that the resolved case set matches the selected starter cases. Explain
the task count, distinct examples, attempts, execution-cost assumptions, and
where rewards and errors will appear. Check that packaging still executes the
agreed examples. Once validated, show the cases, agent, model, and provider-native
command with the agreed attempt settings, then run with established execution
and spend authorization using the selected reference.

Inspect results for every selected case; retain outputs, actions, rewards, and
exceptions as fresh run evidence. A completed evaluation can have low scores.
Recording this baseline with the evidence recorder is optional.

## 10. Review results and explain reruns

Update `.eval-author/first-eval.md` with exact commands, artifact paths, control
results, per-task agent rewards and exceptions, and remaining blockers. Show how
to rerun the suite, inspect one result, and add or modify a task. Finish and link
`.eval-author/README.md` with the full-suite command and concrete commands for
individual coverage items, following the shared suite review and rerun instructions.
Refresh the case inventory with links to control and actual-agent results and
available execution traces, keeping source evidence and missing traces distinct.
Save any needed subset configs and verify their selected cases and retained
agent settings. Label rerun commands as task sanity checks or actual agent evaluation.

When an actual-agent attempt exists, explain in the results reply how to open
Harbor's run viewer and find it, using the shared
[review walkthrough](../eval-author/references/suite-readme.md#show-the-user-how-to-review).
Give its launch command and browser URL or a verified live link, and identify
the saved job and case. State what can be reviewed when the adapter emits no
ATIF; captured actions and replies do not imply a native trajectory exists.
If only controls ran or execution was blocked before any actual-agent run,
say the actual agent has not run and offer case review plus any clearly labeled
control evidence.

Reconcile agreed versus delivered examples per behavior using
[Expand and reconcile](../eval-author/references/coverage-planning.md#expand-and-reconcile).
Separate generated, validated, and executed counts and name omissions, changed
fixtures, and remaining gaps. Repeated attempts never increase example totals.
Render the final five-step checklist, checking every completed step and leaving
blocked or partial steps open with their next action.

The working-suite deliverable is reached when the selected starter tasks execute
and record rewards through a validated config, even if the agent scores poorly
or checks are basic. A functioning case that the agent fails provides a useful
baseline; do not weaken it to obtain success.

Name each created eval case and explain the customer behavior it measures.
Translate the recorded controls into their meaning. For Gym, explain the actual
positive and negative fixtures and observed rewards. For a Harbor binary action case,
doing nothing failed and the prepared reference solution passed, so the grader
distinguishes those examples. Do not call NOP an empty answer unless that is what
it tested. For graded metrics or cases where inaction is correct, explain the
actual acceptance criteria and observed controls. These checks do not show how
the user's agent performs or prove that all correct answers receive credit.

State whether the actual agent ran and explain the observed scores when it did.
If integration is missing, describe the concrete connection needed at the verified
entry point; use “Harbor adapter” only if that helps the user act. If execution is
blocked by a credential, name the required environment variable to configure in
the execution environment. Describe the next run as scoring actual agent responses,
distinguishing it from configuration or connection setup checks.

Explain practical limitations, such as narrow fixtures, a wording assertion
rejecting a correct paraphrase, or an assertion checking only part of an outcome.
Link the eval cases, results, and established Ethos requirements; identify relevant
intent left outside the starter set. Keep detailed scores, interpreter details,
and full rerun commands in the saved report unless useful in the explanation;
include the viewer instructions in the reply as described above.

Traces and improvement are optional follow-up work. Explain that traces reveal
steps and tool calls that can expose failure patterns, missing coverage, and weak
checks. Link actual trace artifacts when emitted; otherwise identify the adapter
or instrumentation needed to produce them. For requested coverage accounting,
hand the established Ethos and actual ATIF to `eval-author-audit`, then use
`eval-author-task-create` for measured actionable gaps.

## Prerequisites

Establish local Ethos before case design. Planning needs agent documentation and
intended behavior, not prior traces, coverage reports, or an evaluation runtime.
Scaffolding and validation require the selected Gym or Harbor installation. Execution
needs the chosen backend, agent connection, and any configured provider access.
Real-agent and paid-judge runs require authorization for execution and spend.

## Limitations

A starter suite establishes a baseline for the selected cases, not comprehensive
coverage. Gym verifier controls or Harbor NOP and Oracle validate task wiring
and selected verifier behavior; they do not measure the real agent. Missing runtime access can leave useful
plans and drafts complete while execution remains unproven.

## Troubleshooting

- Missing Ethos: resume the shared Local Ethos procedure before case design.
- Selected runtime unavailable: retain the case plan and identify the Gym setup
  requirement or follow discovery's Harbor setup guidance. Install only when
  authorized, following the user's selected setup.
- Oracle fails or NOP unexpectedly passes: inspect task state, reference solution,
  and verifier against the intended outcome; repair and rerun the controls.
- Agent connection or credentials missing: record the verified entry point or
  required variable name and the blocked run; never request secret values in chat.
