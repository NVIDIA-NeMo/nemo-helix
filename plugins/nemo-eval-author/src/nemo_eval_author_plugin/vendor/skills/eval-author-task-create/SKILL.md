---
name: eval-author-task-create
description: >-
  Propose dataset improvements from Eval Author audit findings, then optionally
  create one Gym or Harbor task from one actionable uncovered tool. Prove it with
  Gym controls and native execution or Harbor's Oracle, then run it repeatedly
  with the repository's real agent when authorized, and accept it only when
  measured ATIF closes the selected gap every time. Use when the user asks to
  suggest dataset changes, fill an eval gap, turn audit uncovered_items into a
  Gym or Harbor task, or add missing tool coverage. Writes proposals, drafts, and
  measurements only under `.eval-author/`.
triggers:
  - create a Gym task from an audit gap
  - create a Harbor task from an audit gap
  - fill an uncovered eval tool
  - generate missing eval tasks
  - close audit coverage gaps
  - turn uncovered_items into Harbor tasks
  - propose dataset improvements from audit findings
  - suggest new evals based on this audit
not-for:
  - eval-author-first-eval (use when the user has no evaluations yet)
  - eval-author (use for the shared standard and routing)
  - eval-author-audit (use to create the denominator and coverage report)
  - eval-author-discover (use to prove an existing suite is runnable)
  - nemo-evaluator (use to run an existing benchmark without authoring tasks)
compatibility: >-
  Proposals read local audit artifacts and task evidence without running Gym or Harbor.
  Gym task creation needs a separate Gym v0.6.0+ runtime (Python 3.13.14+);
  Harbor task creation needs Python 3.11+ and a CLI compatible with `harbor task init`.
  Docker is required for Oracle and Docker-backed real-agent runs. Real-agent
  runs may require provider credentials and explicit user approval.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: Bash Read Write Grep Glob
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Eval Author: create task

## Purpose

Read `eval-author` for the shared evidence standard and boundaries. Read
`eval-author-audit` for measurement and aggregation. Start with the proposal step
to turn audit evidence into prioritized dataset improvements across tools,
capabilities, and failure cases. For proposal-only requests, stop after Step 1.
Enter that step directly from the core's proposal route, carrying existing audit
findings and prior answers. Accepting the audit's offer to propose tasks also
enters Step 1 directly; reuse the reviewed Audit coverage report and agreed priorities.
When handing off from a guided audit, wait for an explicit answer accepting that
offer; report acceptance alone is not permission to begin proposals.
If required audit inputs are missing, obtain those
inputs through the audit flow and return to proposals; do not restart first-eval
onboarding or infer permission to create or run tasks.
For task-creation requests, continue with an eligible tool gap through the
existing execution path:

```text
actionable uncovered tool
  → Gym-native or Harbor-native draft
  → native positive and negative controls
  → repeated real-agent ATIF
  → target tool covered in every report
```

Create and prove one tool gap at a time. Keep every generated artifact under
`.eval-author/`; do not edit existing tasks or customer source.

For an existing suite without sufficient audit findings, preserve the requested outcome:
audit or proposal work obtains its missing coverage inputs through
`eval-author-audit`; readiness work belongs in `eval-author-discover`. A missing
report alone is not a first-eval request. A reviewed specification and Markdown
report can support proposal-only work without measurement JSON: label such ideas
as unmeasured candidates and carry the measurement limitation forward. Do not
repeat an already-deferred measurement just to propose candidates. Automatic
tool-gap task creation still requires the aggregate JSON report.

## Available Scripts

Run `uv run <skill_dir>/scripts/task_pipeline.py <command>`. Run `task_evidence.py`
with the selected runtime's Python; Harbor preparation imports Harbor.

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/task_evidence.py` | Record current-revision native controls and agent runs | `prepare --task-dir --contract --out`; `run --manifest --case --run-id --result --out [--trace] -- <command>` |
| `scripts/task_pipeline.py` | Select, scaffold, and verify one audit-gap task | `select --report`; `scaffold --report --target` plus task metadata; `verify --before --after --target --evidence` |

The three deterministic commands have these verdicts:

| Command | Verdict |
|---|---|
| `select` | Lists only tool items with `reason: not_covered_by_any_input_report` and emits a deterministic `task_slug` plus artifact paths |
| `scaffold` | Calls Gym's native scaffolder with `--provider gym` or Harbor's `harbor task init`, requires matching draft/proposal names for that slug, and installs the supplied instruction |
| `verify` | Exits 0 only with current-revision control and agent receipts, matching task/trace identities, and tool coverage in both distinct agent runs |

The script prints one JSON object. Exit code 0 is success; do not replace its
verdict with model judgment.

## Step 1: propose dataset improvements

Use the shared [Coverage planning](../eval-author/references/coverage-planning.md)
procedure for source scope, priorities, and proposed example breadth. Reuse the
audit's confirmed corpus and exclusions when inspecting evidence; a discovered
path is not permission to include another corpus. Ask about an unresolved source
before searching or reading it. Proposal-only work can use the available reviewed
findings without requiring new traces or a creation-scope checkpoint.

When available, read the **Audit coverage report** (`.eval-author/audit-coverage-report.md`)
for test mappings, findings, limitations, and agreed next actions. Check its applicability
against the underlying specification and reports; a mapped test does not establish
measured coverage. The Markdown report supplies context, while measurement JSON
remains the input for deterministic tool-gap selection and verification. Its
absence alone does not require repeating an otherwise usable audit.

Keep the audit's [artifact names](../eval-author-audit/references/coverage-report.md#artifact-names)
in proposal replies and links: Audit specification (`audit.md`), Audit coverage
report (`audit-coverage-report.md`), and Coverage measurements (JSON)
(`audit-coverage-report.json`). Review status does not change these names.

Read available aggregate measurements, relevant per-trace `details.json` and
capability judgments, and the selected source traces or verifier results needed
to explain the findings. When measurement is unavailable, ground candidate
proposals in the reviewed specification and inspected eval definitions instead.
Check existing task instructions, fixtures, and verifiers before
claiming a scenario is absent or proposing a duplicate. A capability covered in
one run can still fail in another; review observed failures even when the item
is absent from `uncovered_items`.

Distinguish the basis for each recommendation:

| Basis | What it supports |
|---|---|
| Observed failure | A trace or verifier shows incorrect behavior on an exercised scenario. Preserve the existing failing task as a regression; propose strengthening it only when a specific fixture or verifier change adds value. An agent fix may be the next action without any dataset change. |
| Coverage gap in inspected inputs | A measured item is not demonstrated and the inspected tasks or traces lack the intended scenario. Propose a concrete new task or extension; state the inspected scope rather than claiming the entire dataset lacks coverage. |
| Unmeasured or insufficient evidence | The method was not selected or is unsupported, judgments are missing or unclear, or the scenario's trigger is unverified. Recommend measurement or inspection first. Ethos-backed scenarios may still be proposed as candidates, with their unmeasured status explicit. |

`not_covered_by_any_input_report` alone does not distinguish an absent scenario,
an agent failure, or missing judgments. `not_measured_by_any_method` is a
measurement limitation, not proof of a dataset deficiency. Failure-case coverage requires the audit
skill's `failure_cases` measurement; manual observations alone do not change
measurement status. Failure cases remain ineligible for automatic tool-gap generation.

Write recommendations to `.eval-author/proposals/dataset-recommendations.md` as
skill-authored analysis; keep the generated coverage JSON unchanged. Rank by
expected value and strength of evidence, explaining why the first action comes
first. Apply established user priorities, including cost when relevant, and
distinguish them from suggested priorities still awaiting review. Prefer a short,
useful list over one suggestion per uncovered item.
For each recommendation, include:

- **Change and scenario:** add a task, strengthen a named existing task, retain
  an existing regression, or gather evidence; describe the concrete request and
  fixture or failure trigger that makes it useful.
- **Expected behavior and verification:** the outcome to check and how a verifier
  would distinguish correct from incorrect behavior. For failure cases, identify
  evidence that the trigger actually occurred as well as the expected response.
- **Basis and evidence:** the category above, stable audit item names, and task,
  run, trace-step, judgment, or verifier references supporting the recommendation.
  Mark proposed fixture details as proposals, not observed facts.
- **Proposed breadth:** the distinct input examples, behavior variations,
  difficulty, positive and negative scenarios, expected outcomes, exclusions,
  and cost assumptions that would make this recommendation useful. Label counts
  and tradeoffs as provisional until agreed; repeated runs are not new examples.
- **Next action:** say whether the suggestion is eligible for automatic tool-gap
  task creation, needs manual task design, or needs more measurement. A written
  recommendation is not a generated, validated, or accepted Gym or Harbor task.

For Gym recommendations, include a [version and rerun plan](references/gym-tasks.md#version-and-rerun-plan)
in the recommendation: exact pins supported by the available evidence, unresolved
versions, and the settings and reset procedure needed to repeat the evaluation.
Carry the selected plan into task creation; a proposed pin is not a verified lock.

Lead the proposal response with the highest-value recommendations and enough
scenario and expected-behavior detail to act on them. Follow with supporting
coverage counts, measurement limits, and links to the proposals and Audit coverage report.
Full tool coverage or an unsupported task-generation path must not suppress
useful capability or failure-case suggestions. Do not relabel those suggestions
as tool gaps to pass the selector. If evidence supports no dataset change, say why
and identify any useful measurement or agent-fix action instead of inventing
additions. Do not scaffold or run tasks for a proposal-only request.

## Step 2: select one actionable gap

```bash
uv run <skill_dir>/scripts/task_pipeline.py select \
  --report .eval-author/audit-coverage-report.json
```

Choose one item from `actionable_tools`. Stop task creation when the list is
empty, and surface the dataset recommendations from Step 1. An empty selector means no eligible tool
gaps, not that there are no useful dataset improvements. Capability and
failure-case items are not task-generation inputs in v1, even when capability
coverage was measured.

Each actionable tool includes a deterministic `task_slug` of the form
`cover-<tool-name>` and a `paths` object for the proposal, draft, and
measurement directories. Use those paths verbatim for the rest of this flow.
Do not invent alternate slugs or filenames.

## Step 3: design the smallest objective task

Before generating task instructions, fixtures, or scaffolds, use the shared
[coverage-planning checkpoint](../eval-author/references/coverage-planning.md).
Use the core's [provider-selection rule](../eval-author/SKILL.md#select-the-evaluation-provider)
to resolve Gym or Harbor when mapping scope to execution counts. Gym repeats
selected dataset rows, while Harbor repeats tasks. Keep estimates conditional
if that choice remains open.
Present the proposed priorities and breadth for review, including an option to
start with a minimal pilot and expand. Reuse already agreed scope rather than
asking again. Save the agreement in
`.eval-author/proposals/dataset-recommendations.md`: selected corpus and exclusions,
priorities and cost constraints, distinct input-example counts per behavior,
variations and difficulty, positive and negative scenarios, expected outcomes,
and pilot or expansion status. Keep this planning record out of the agent-facing
instruction file. Answer questions or complete scoped detours, then resume this
checkpoint with the existing decisions intact.

Explain how the agreement maps to the selected tool gap. The deterministic
pipeline creates one tool-gap draft at a time; repeated runs test closure on
the selected inputs and do not add distinct input examples. Keep any agreed
examples beyond the current draft visible as pending work. A capability or
failure-case proposal outside the generator's support remains an explicit gap
requiring manual design, not a completed example or a relabeled tool gap.
If the reviewed priorities change the target, return to Step 2 before generation.

Read the selected item's `description`, `focus`, `needed_tools`, and
`evidence_required`. Read one nearby task for domain conventions only. Do not
copy its directory: a sibling can carry an obsolete Harbor schema, placeholder
verifier, or unrelated solution.

Write the instruction only to `paths.proposal` from Step 2. State the observable
goal, paths, and constraints without naming the target tool or leaking verifier
logic. The task should naturally require the selected tool and no unrelated
capability.

Decide the verifier before scaffolding. Prefer deterministic shell or pytest.
The verifier must grade the task outcome, not the tool call; ATIF measurement
proves tool coverage separately.

## Step 4: scaffold with Gym or Harbor

Select the provider from the user's request and existing suite.
For **Gym**, follow [Author and prove a Gym evaluation](references/gym-tasks.md)
for native scaffolding, verifier controls, and real-agent runs. Keep Steps 1–3's
selected gap and instruction, then return to the shared coverage verification
steps. Do not apply Harbor file layouts or Oracle CLI flags to Gym. For either
provider, build the task's environment with
[`eval-author-environment`](../eval-author-environment/SKILL.md) and pass its
smoke task before writing the verifier, reusing the agent's existing proven
environment kit when one exists, and
derive expected values with an independent reference query over the kit's data.

For **Harbor**, use the following native scaffold:

```bash
uv run <skill_dir>/scripts/task_pipeline.py scaffold \
  --report .eval-author/audit-coverage-report.json \
  --target <tool-name> \
  --out .eval-author/task-drafts/<task-slug> \
  --task-name <org>/<task-slug> \
  --description "<one-line description>" \
  --author "<author>" \
  --instruction-file .eval-author/proposals/<task-slug>-instruction.md
```

Use the Step 2 `task_slug` for `<task-slug>` in every path above. `scaffold`
rejects mismatched draft, task-name, or proposal filenames.

Then complete Harbor's generated files:

- `environment/`: a copy of the agent's environment kit plus this task's own
  records, buildable on its own rather than `FROM` a local kit image, and never
  the solution.
- `tests/test.sh`: deterministic reward writer using absolute paths.
- `solution/solve.sh`: executable Oracle solution.
- `task.toml`: nonempty keywords, metadata, realistic timeouts and resources.
- `README.md`: purpose, environment, verifier, layout, and persistent suite and
  individual-item run instructions.

For either provider, create the generated suite's top-level review and rerun guide using
[Suite review and rerun instructions](../eval-author/references/suite-readme.md).
For a standalone draft, extend its task-level `README.md` without renaming it;
when adding it to a generated collection, also update that collection's guide
and coverage mapping. Present and link the case inventory when the cases are
authored, with execution pending, then refresh it after native controls and
actual-agent runs with their available results and execution traces.

Do not leave generated placeholders, `pass`, unconditional reward 1, or empty
keywords.

## Step 5: prove task correctness with native controls

Confirm the environment kit's proof passed first; a proof failure is an
environment defect to fix, not a task result, and that proof stays in the kit's
plan rather than among the task's receipts. The task's own controls never count
as environment proof.

Apply [Task validation and execution evidence](../eval-author/references/task-validation.md).
Prepare the result contract and revision manifest before execution. Record each
native control and each agent attempt as its own `scripts/task_evidence.py run`
invocation, with fresh native results and receipts outside the task tree. Harbor
requires a reference and a realistic incorrect control; include applicable
valid-alternative and side-effect cases. A failing command or missing evidence is
not a passing negative control. Use the same Harbor Python runtime for
preparation and runs.

Follow [Execution recovery](../eval-author/references/execution-recovery.md)
for native controls, real-agent trials, and reporting. Compatibility repairs
must preserve the selected provider and original grading semantics.

For Gym, use the positive and negative controls and native execution in the
[Gym guide](references/gym-tasks.md). For Harbor, record the Oracle reference
before spending model credentials. `harbor trial start` writes `result.json` and
`agent/trajectory.json` under its `--trials-dir` and `--trial-name`, giving the
recorder fixed paths; a Harbor job names its trials randomly:

```bash
TRIALS="$PWD/.eval-author/jobs/<task-slug>"
<harbor_python> <skill_dir>/scripts/task_evidence.py run \
  --manifest .eval-author/task-measurements/<task-slug>/revision-1.json \
  --case reference --run-id reference-1 \
  --result "$TRIALS/reference-1/result.json" \
  --out .eval-author/task-measurements/<task-slug>/reference-1.json \
  -- harbor trial start -p .eval-author/task-drafts/<task-slug> -a oracle \
  --trial-name reference-1 --trials-dir "$TRIALS"
```

Continue only when the receipt reports `passed`, with no exception and reward
1.0. Record the incorrect control the same way, choosing its agent as described
in [Review the verifier before running](../eval-author/references/task-validation.md#review-the-verifier-before-running).
Fix the task, solution, or verifier when Oracle fails; do not weaken the verifier
merely to make it pass.

## Step 6: run the real agent twice

Running a model spends credentials. Do it only when the user asked for the run
or approved it. Use the repository's proven agent configuration and point it at
the draft. Record each attempt as its own invocation with fresh native output:
one Gym repeat per recorded run, or one Harbor trial per run with a distinct
trial name. One job with two attempts cannot back two receipts. Keep all outputs
under `.eval-author/`:

```bash
<harbor_python> <skill_dir>/scripts/task_evidence.py run \
  --manifest .eval-author/task-measurements/<task-slug>/revision-1.json \
  --case agent --run-id repeat-1 \
  --result "$TRIALS/repeat-1/result.json" \
  --trace "$TRIALS/repeat-1/agent/trajectory.json" \
  --out .eval-author/task-measurements/<task-slug>/agent-1.json \
  -- harbor trial start -p .eval-author/task-drafts/<task-slug> \
  -a <agent> -m <model> --trial-name repeat-1 --trials-dir "$TRIALS"
```

Repeat with `repeat-2` and `agent-2.json`. Carry a proven Harbor job's agent
settings into the equivalent trial options or a trial `--config`.

Require both trials to:

1. finish without an exception,
2. receive the intended verifier reward, and
3. retain interaction evidence accepted as ATIF by the audit `measure.py`: Gym
   Responses converted with receipts, original retained ATIF, or Harbor's
   `agent/trajectory.json`. Trace Loader output alone is not ATIF.

`SUPPORTS_ATIF = true` is not evidence that the emitted JSON matches Harbor's
current schema. A `measure.py` parse failure is an agent-adapter defect, not a
coverage result.

## Step 7: measure and aggregate each trial

Run `eval-author-audit`'s `measure.py` and `report.py` separately for each
recorded attempt. Measure the recorded trace with the recorder's `--run-id` and
the selected `<task-slug>`; `verify` rejects any other trace. For Harbor, pass the
recorded trial directory; for Gym, pass the converted ATIF with `--trace`. Keep
repeat outputs separate so one successful run cannot hide another:

```bash
uv run --with-requirements <audit_skill_dir>/requirements.txt \
  <audit_skill_dir>/scripts/audit_spec/measure.py \
  --audit .eval-author/audit.md \
  --trial-dir "$TRIALS/repeat-1" \
  --task-id <task-slug> \
  --run-id repeat-1 \
  --out-dir .eval-author/task-measurements/<task-slug>/repeat-1

uv run --with-requirements <audit_skill_dir>/requirements.txt \
  <audit_skill_dir>/scripts/audit_spec/report.py \
  --audit .eval-author/audit.md \
  --coverage-dir .eval-author/task-measurements/<task-slug>/repeat-1 \
  --out .eval-author/task-measurements/<task-slug>/repeat-1-report.json
```

Repeat for `repeat-2`.

These newly generated trials are evidence for the selected draft. Retain their
relationship to the agreed examples separately from the original selected
corpus; do not silently add them to or replace that corpus in an audit report.

## Step 8: accept only deterministic closure

```bash
uv run <skill_dir>/scripts/task_pipeline.py verify \
  --before .eval-author/audit-coverage-report.json \
  --after .eval-author/task-measurements/<task-slug>/repeat-1-report.json \
  --after .eval-author/task-measurements/<task-slug>/repeat-2-report.json \
  --target <tool-name> \
  --evidence .eval-author/task-measurements/<task-slug>/reference-1.json \
  --evidence .eval-author/task-measurements/<task-slug>/incorrect-1.json \
  --evidence .eval-author/task-measurements/<task-slug>/agent-1.json \
  --evidence .eval-author/task-measurements/<task-slug>/agent-2.json
```

Add receipts for every other declared control. Use absolute trace paths in the
measurement subjects, matching the recorder's trace paths and run IDs. v2
verification rejects stale revisions, missing controls, unrelated tasks,
reused native runs, and traces unrelated to the execution receipts. Task
validation, successful agent execution, and coverage closure are separate fields.

Accept the draft only when `accepted` is `true`. Report native control rewards, both
real-agent rewards, both run/trace paths, and the verify JSON. If either repeat
misses the tool, revise the task and rerun both attempts.
Finish and link the suite's review and rerun guide at handoff, including when
execution is blocked. Preserve the actual agent settings in the full-suite and selected-item
commands; link measured coverage reports separately from run instructions.

At the handoff, reconcile agreed scope with delivered examples in
`.eval-author/proposals/dataset-recommendations.md`, including partial or blocked
work. Show planned and delivered distinct input counts per behavior, links to
their tasks, and generated, validated, and actually executed examples separately.
Include completed checks and run evidence, remaining variations or outcomes,
and any changed exclusions or cost constraints. Separate deterministic tool-gap
closure from the wider behavior coverage agreement. Reuse the shared planning
checkpoint before an expansion changes scope; keep pending expansion visible
without claiming the pilot completes it.

## Prerequisites

Proposals need audit findings and the relevant task or trace evidence. Automated
task creation additionally needs an actionable measured tool gap, Python 3.11+,
and the selected native runtime: Gym in its separate environment or a Harbor
CLI supporting `harbor task init`. Docker is needed for Docker-backed execution;
measurement uses the audit skill's dependencies.
Use the repository's proven agent configuration and authorize real-agent spend
before starting those jobs.

## Limitations

The generator handles one eligible tool gap at a time; capabilities and failure
cases can inform proposals but are not automatic task-generation inputs.
Two passing measured repeats prove closure only for those runs; distinct example
counts come from the task's inputs, not its repeats. An uncovered item
alone cannot distinguish missing scenarios, missing evidence, and agent failure.
Proposal-only requests end before scaffolding or execution.

## Troubleshooting

- Empty `actionable_tools`: report the evidence-backed proposals; do not relabel
  unmeasured or non-tool gaps to force selection.
- Scaffold path mismatch: use the selector's `task_slug` and `paths` verbatim.
- Native control failure: repair the task, solution, or verifier against the intended
  outcome, then rerun; preserve the assertion being tested.
- Invalid ATIF or repeated run identities: fix the adapter or obtain two distinct
  recorded trials before measurement; a reward cannot replace trajectory evidence.
- `verify` reports a trace or run-ID mismatch: measure the recorder's exact trace
  with its `--run-id` and `--task-id <task-slug>`.
- `accepted: false`: inspect both reports, revise the draft if needed, and rerun
  both attempts within the authorized scope before claiming closure.
