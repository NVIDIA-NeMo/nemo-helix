---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: eval-author-task-create
description: >-
  Propose dataset improvements from Eval Author audit findings, then optionally
  create one Harbor task from one actionable uncovered tool. Prove the task with
  Harbor's Oracle, run it repeatedly
  with the repository's real agent when authorized, and accept it only when
  measured ATIF closes the selected gap every time. Use when the user asks to
  suggest dataset changes, fill an eval gap, turn audit uncovered_items into a
  Harbor task, or add missing tool coverage. Writes proposals, drafts, and
  measurements only under `.eval-author/`.
triggers:
  - create a Harbor task from an audit gap
  - fill an uncovered eval tool
  - generate missing eval tasks
  - close audit coverage gaps
  - turn uncovered_items into Harbor tasks
  - propose dataset improvements from audit findings
not-for:
  - eval-author (use for the shared standard and routing)
  - eval-author-audit (use to create the denominator and coverage report)
  - eval-author-discover (use to prove an existing suite is runnable)
  - nemo-evaluator (use to run an existing benchmark without authoring tasks)
compatibility: >-
  Proposals read local audit artifacts and task evidence without running Harbor.
  Task creation needs Python 3.11 or later and a Harbor CLI compatible with `harbor task init`.
  Docker is required for Oracle and Docker-backed real-agent runs. Real-agent
  runs may require provider credentials and explicit user approval.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Write, Grep, Glob]
---

# Eval Author: create task

Read `eval-author` for the shared evidence standard and boundaries. Read
`eval-author-audit` for measurement and aggregation. Start with the proposal step
to turn audit evidence into prioritized dataset improvements across tools,
capabilities, and failure cases. For proposal-only requests, stop after Step 1.
For task-creation requests, continue with an eligible tool gap through the
existing execution path:

```text
actionable uncovered tool
  → Harbor-native draft
  → Oracle reward 1
  → repeated real-agent ATIF
  → target tool covered in every report
```

Create and prove one tool gap at a time. Keep every generated artifact under
`.eval-author/`; do not edit existing tasks or customer source.

## Script

`scripts/task_pipeline.py` has three deterministic commands:

| Command | Verdict |
|---|---|
| `select` | Lists only tool items with `reason: not_covered_by_any_input_report` and emits a deterministic `task_slug` plus artifact paths |
| `scaffold` | Calls Harbor's own `harbor task init`, requires matching draft/proposal names for that slug, and installs the supplied instruction |
| `verify` | Exits 0 only when the selected tool was uncovered before and covered in two distinct repeated after-reports with distinct ATIF `subject.run_id` values |

The script prints one JSON object. Exit code 0 is success; do not replace its
verdict with model judgment.

## Step 1: propose dataset improvements

Read the aggregate audit report, relevant per-trace `details.json` and capability
judgments, and the source traces or verifier results needed to explain the
findings. Check existing task instructions, fixtures, and verifiers before
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
measurement limitation, not proof of a dataset deficiency. Failure-case items
remain unmeasured in v1; manual trace observations do not change that status.

Write recommendations to `.eval-author/proposals/dataset-recommendations.md` as
skill-authored analysis; keep the generated coverage JSON unchanged. Rank by
expected value and strength of evidence, explaining why the first action comes
first. Prefer a short, useful list over one suggestion per uncovered item.
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
- **Next action:** say whether the suggestion is eligible for automatic tool-gap
  task creation, needs manual task design, or needs more measurement. A written
  recommendation is not a generated, validated, or accepted Harbor task.

Lead the proposal response with the highest-value recommendations and enough
scenario and expected-behavior detail to act on them. Follow with supporting
coverage counts, measurement limits, and links to the proposals and audit report.
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

## Step 4: scaffold with Harbor

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

- `environment/Dockerfile`: task prerequisites, never the solution.
- `tests/test.sh`: deterministic reward writer using absolute paths.
- `solution/solve.sh`: executable Oracle solution.
- `task.toml`: nonempty keywords, metadata, realistic timeouts and resources.
- `README.md`: purpose, environment, verifier, layout, and run commands.

Do not leave generated placeholders, `pass`, unconditional reward 1, or empty
keywords.

## Step 5: prove task correctness with Oracle

Run Harbor's Oracle before spending model credentials:

```bash
harbor run -p .eval-author/task-drafts/<task-slug> -a oracle
```

Continue only when Harbor reports no exception and reward 1.0. Fix the task,
solution, or verifier when Oracle fails; do not weaken the verifier merely to
make it pass.

## Step 6: run the real agent twice

Running a model spends credentials. Do it only when the user asked for the run
or approved it. Use the repository's proven agent configuration, point it at the
draft, and set `n_attempts: 2`. Keep the resulting job under `.eval-author/`.

Require both trials to:

1. finish without an exception,
2. receive the intended verifier reward, and
3. contain `agent/trajectory.json` accepted as ATIF by the audit `measure.py`.

`SUPPORTS_ATIF = true` is not evidence that the emitted JSON matches Harbor's
current schema. A `measure.py` parse failure is an agent-adapter defect, not a
coverage result.

## Step 7: measure and aggregate each trial

Run `eval-author-audit`'s `measure.py` and `report.py` separately for each
trial. Keep repeat outputs separate so one successful run cannot hide another:

```bash
uv run --with-requirements <audit_skill_dir>/requirements.txt \
  <audit_skill_dir>/scripts/audit_spec/measure.py \
  --audit .eval-author/audit.md \
  --trial-dir <job-dir>/<trial-1> \
  --task-id <task-slug> \
  --run-id repeat-1 \
  --out-dir .eval-author/task-measurements/<task-slug>/repeat-1

uv run --with-requirements <audit_skill_dir>/requirements.txt \
  <audit_skill_dir>/scripts/audit_spec/report.py \
  --audit .eval-author/audit.md \
  --coverage-dir .eval-author/task-measurements/<task-slug>/repeat-1 \
  --out .eval-author/task-measurements/<task-slug>/repeat-1-report.json
```

Repeat for trial 2.

## Step 8: accept only deterministic closure

```bash
uv run <skill_dir>/scripts/task_pipeline.py verify \
  --before .eval-author/audit-coverage-report.json \
  --after .eval-author/task-measurements/<task-slug>/repeat-1-report.json \
  --after .eval-author/task-measurements/<task-slug>/repeat-2-report.json \
  --target <tool-name>
```

Accept the draft only when `accepted` is `true`. Report Oracle reward, both
real-agent rewards, both trial paths, and the verify JSON. If either repeat
misses the tool, revise the task and rerun both attempts.
