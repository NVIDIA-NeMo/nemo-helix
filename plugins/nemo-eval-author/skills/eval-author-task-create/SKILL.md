---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: eval-author-task-create
description: >-
  Create one Harbor task from one actionable uncovered tool in an Eval Author
  audit coverage report, prove the task with Harbor's Oracle, run it repeatedly
  with the repository's real agent when authorized, and accept it only when
  measured ATIF closes the selected gap every time. Use when the user asks to
  fill an eval gap, turn audit uncovered_items into a Harbor task, or add missing
  tool coverage. Writes drafts and measurements only under `.eval-author/`.
triggers:
  - create a Harbor task from an audit gap
  - fill an uncovered eval tool
  - generate missing eval tasks
  - close audit coverage gaps
  - turn uncovered_items into Harbor tasks
not-for:
  - eval-author (use for the shared standard and routing)
  - eval-author-audit (use to create the denominator and coverage report)
  - eval-author-discover (use to prove an existing suite is runnable)
  - nemo-evaluator (use to run an existing benchmark without authoring tasks)
compatibility: >-
  Python 3.11 or later and a Harbor CLI compatible with `harbor task init`.
  Docker is required for Oracle and Docker-backed real-agent runs. Real-agent
  runs may require provider credentials and explicit user approval.
maturity: alpha
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Write, Grep, Glob]
---

# Eval Author: create task

Read `eval-author` for the shared evidence standard and boundaries. Read
`eval-author-audit` for measurement and aggregation. This sub-flow owns only:

```text
actionable uncovered tool
  → Harbor-native draft
  → Oracle reward 1
  → repeated real-agent ATIF
  → target tool covered in every report
```

Work on one tool gap at a time. Keep every generated artifact under
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

## Step 1: select one actionable gap

```bash
uv run <skill_dir>/scripts/task_pipeline.py select \
  --report .eval-author/audit-coverage-report.json
```

Choose one item from `actionable_tools`. Stop task creation when the list is
empty, and surface the dataset recommendations from `eval-author-audit` Step 6
(or complete that step if needed). An empty selector means no eligible tool
gaps, not that there are no useful dataset improvements. Capability and
failure-case items are not task-generation inputs in v1, even when capability
coverage was measured.

Each actionable tool includes a deterministic `task_slug` of the form
`cover-<tool-name>` and a `paths` object for the proposal, draft, and
measurement directories. Use those paths verbatim for the rest of this flow.
Do not invent alternate slugs or filenames.

## Step 2: design the smallest objective task

Read the selected item's `description`, `focus`, `needed_tools`, and
`evidence_required`. Read one nearby task for domain conventions only. Do not
copy its directory: a sibling can carry an obsolete Harbor schema, placeholder
verifier, or unrelated solution.

Write the instruction only to `paths.proposal` from Step 1. State the observable
goal, paths, and constraints without naming the target tool or leaking verifier
logic. The task should naturally require the selected tool and no unrelated
capability.

Decide the verifier before scaffolding. Prefer deterministic shell or pytest.
The verifier must grade the task outcome, not the tool call; ATIF measurement
proves tool coverage separately.

## Step 3: scaffold with Harbor

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

Use the Step 1 `task_slug` for `<task-slug>` in every path above. `scaffold`
rejects mismatched draft, task-name, or proposal filenames.

Then complete Harbor's generated files:

- `environment/Dockerfile`: task prerequisites, never the solution.
- `tests/test.sh`: deterministic reward writer using absolute paths.
- `solution/solve.sh`: executable Oracle solution.
- `task.toml`: nonempty keywords, metadata, realistic timeouts and resources.
- `README.md`: purpose, environment, verifier, layout, and run commands.

Do not leave generated placeholders, `pass`, unconditional reward 1, or empty
keywords.

## Step 4: prove task correctness with Oracle

Run Harbor's Oracle before spending model credentials:

```bash
harbor run -p .eval-author/task-drafts/<task-slug> -a oracle
```

Continue only when Harbor reports no exception and reward 1.0. Fix the task,
solution, or verifier when Oracle fails; do not weaken the verifier merely to
make it pass.

## Step 5: run the real agent twice

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

## Step 6: measure and aggregate each trial

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

## Step 7: accept only deterministic closure

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
