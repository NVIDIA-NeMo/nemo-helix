<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Task validation and execution evidence

Apply during first-eval and task-create authoring, after scope is agreed. This
adds no execution permission. Proposal-only work still ends with recommendations.
Keep runtime readiness, task validation, agent performance, and measured coverage
separate. A failing agent can supply a useful first-eval baseline; automatic
tool-gap acceptance additionally requires successful repeated agent executions.

## Review the verifier before running

Map each scored check to a visible requirement, fixture, and observable outcome.
Verify the public instruction and grader agree, including boundary values,
defaults, preserved state, and permitted alternative solutions. Ground expected
results in the specification, reviewed fixtures, or independently validated
reference execution, never the submission being graded. A source agent's answer
is evidence to investigate, not ground truth by itself.

For each outcome, identify a plausible incomplete implementation that the grader
must reject. Exercise the supported interface and observe the actual result or
side effect; a tool name, success message, or agent-written log is insufficient.
Keep graded metrics and correct inaction when the task calls for them. Do not
require every no-action check to fail or force all rewards into binary form.

Declare a small set of controls appropriate to the task:

- **Reference:** a known-correct response or interaction satisfies the checks.
- **Incorrect:** a realistic wrong result reaches grading and receives the
  expected rejection. A broken environment or unreadable result is not rejection.
- **Alternative:** a materially different valid solution passes where the public
  contract permits one. Otherwise record a concrete non-applicability reason.
- **Side effect:** check the requested state change and relevant preservation
  constraints through trusted observations, when applicable. Otherwise explain
  why the task has no side-effect requirement.

NOP can serve as the incorrect control only when doing nothing is incorrect and
its failure exercises the intended assertion. Harbor's built-in control agents
are `nop` and `oracle`; run any other control as a custom agent
(`-a <module:Class>`) whose source lives in the task tree. The recorder adds no
shell, so set its import path with `env PYTHONPATH=<dir> harbor ...`. Do not swap
`solution/` to run a control: that changes the task revision. Include a copy control when a
concrete visible-answer shortcut is plausible. Inspect actual image layers,
mounts, histories, and runtime access for reference/expected-answer leakage.
Keep grading and reward files verifier-owned and fresh. Use the provider's
supported isolation and declare the complete output handoff; verify the handoff
in native execution. Preserve necessary agent networking and native software.
If isolation cannot verify a required live side effect, retain that limitation.
Do not replace the original task with an easier one to obtain passing controls.

These controls test particular counterexamples; neither their existence nor a
passing result establishes comprehensive soundness, fairness, or difficulty.

## Prepare immutable inputs

Use `eval-author-task-create/scripts/task_evidence.py` with Python 3.11+; Harbor
preparation also needs the same Harbor Python runtime used for execution. Finish
the task, verifier, control implementations, and any result adapter before
preparing evidence. Keep those files inside the task tree. Keep receipts, jobs,
converted traces, and measurement outputs outside that tree under `.eval-author/`.

Write a private contract identifying the task and native result checks. Each
case has `kind`, a nonempty `requirement`, `health` assertions, and behavior
`checks`. Use absolute JSON pointers into retained native JSON. Assertions use
`equals` (type-sensitive) or inclusive numeric `min`/`max` bounds. Missing fields
are infrastructure errors, never successful negative controls. For numeric
rewards use bounds, so native `1` and `1.0` both work.

For example, a Harbor contract begins:

```json
{
  "task_id": "cover-read",
  "provider": "harbor",
  "native_run_id": "/id",
  "alternative_not_applicable": "The requested output is a single exact integer.",
  "side_effect_not_applicable": "Only a returned answer is requested.",
  "cases": {
    "reference": {
      "kind": "reference",
      "requirement": "Return the independently computed total.",
      "health": [{"pointer": "/exception_info", "equals": null}],
      "checks": [{"pointer": "/verifier_result/rewards/reward", "min": 1, "max": 1}]
    },
    "incorrect": {
      "kind": "incorrect",
      "requirement": "Reject a plausible total that omits one record.",
      "health": [{"pointer": "/exception_info", "equals": null}],
      "checks": [{"pointer": "/verifier_result/rewards/reward", "min": 0, "max": 0}]
    },
    "agent": {
      "kind": "agent",
      "requirement": "The actual agent returns the correct total.",
      "health": [{"pointer": "/exception_info", "equals": null}],
      "checks": [{"pointer": "/verifier_result/rewards/reward", "min": 1, "max": 1}]
    }
  }
}
```

Inspect the installed runtime's actual result shape before choosing pointers.
For Harbor, retain the trial's native `result.json`: the recorder additionally
checks its task checksum and absence of exceptions against the prepared task.
For Gym, use the maintained [native Gym evidence adapter](../../eval-author-task-create/references/gym-evidence.md).
It reads native verifier-case reports separately from rollout JSONL and health
artifacts; authors do not implement a result adapter. Prepare the selected dataset
row, resolved configuration, runtime record, and reset plan before recording.
The adapter compares executed inputs, preserves native health status, and checks
conversion receipts against the selected rollout and ATIF bytes. It does not
attest a remote service's deployed source.

```bash
python <task_create_dir>/scripts/task_evidence.py prepare \
  --task-dir .eval-author/task-drafts/cover-read \
  --contract .eval-author/task-measurements/cover-read/contract.json \
  --out .eval-author/task-measurements/cover-read/revision-1.json
```

The manifest snapshots file bytes, paths, directories and permissions, excluding
only `.git`, `.venv`, `__pycache__`, `.pytest_cache`, and `.ruff_cache`. Symlinks
are rejected. Do not put grading inputs in excluded caches; dependencies and
external services still need the provider's version and rerun plan.

## Record controls and agent runs

Record each control and each agent attempt as its own invocation around the
trusted native command. Every receipt needs fresh native output, so one job with
several attempts can back only one receipt:

```bash
TRIALS="$PWD/.eval-author/jobs/cover-read"
python <task_create_dir>/scripts/task_evidence.py run \
  --manifest .eval-author/task-measurements/cover-read/revision-1.json \
  --case reference --run-id reference-1 \
  --result "$TRIALS/reference-1/result.json" \
  --out .eval-author/task-measurements/cover-read/reference-1.json \
  -- harbor trial start -p .eval-author/task-drafts/cover-read -a oracle \
  --trial-name reference-1 --trials-dir "$TRIALS"
```

The command must produce the declared result. `harbor trial start` writes the
trial's `result.json` and `agent/trajectory.json` under its `--trials-dir` and
`--trial-name`, so they can be the declared `--result` and `--trace` without a
copying wrapper; Harbor jobs name their trials randomly. Gym uses the maintained
adapter described above. The recorder runs argv directly without adding a shell
and uses the current directory. Put wrappers and control sources in the prepared
task tree, and pass the actual selected task/configuration. Never copy old job
results to simulate a new execution. This is evidence bookkeeping for trusted
authoring, not an attestation against a malicious command or edited receipts.
Receipts store the command line verbatim. Pass credentials through the
environment, which the recorder forwards to the command; never put secret values
in arguments such as Harbor's `--ae KEY=VALUE`.

Run every declared control. For first-eval, recording the agent baseline is
optional: each Gym manifest binds one dataset row and every receipt needs fresh
native output, so per-case baseline receipts multiply model runs. Retain the
suite-level baseline run as native evidence instead. A recorded first-eval
baseline uses `--purpose baseline`; it allows one attempt without ATIF and cannot
count as gap-closure evidence.
For `--purpose closure` (the default), agent runs require `--trace <fresh-absolute-ATIF-path>`;
produce or convert that trace inside the recorded command, retaining conversion
receipts where required. Use that exact absolute trace path and the recorder's
`--run-id` when measuring coverage. Distinct measurements of the same trace may
share a report, but each after-report must describe exactly one agent run.

The recorder checks the task before and after execution, retains the command,
manifest/result/trace digests, native run identity, and an explicit outcome:
`passed`, `behavior_failed`, or `infrastructure_error`. Command failure, timeout,
missing or corrupt results, health failure, and changed inputs cannot pass.
Nonzero command exits are infrastructure errors even if a reward file exists;
use native negative controls whose execution succeeds while grading rejects them.

`diagnostic` cases are retained but cannot count as acceptance evidence. For
first-eval, report all control outcomes and the actual agent baseline, including
legitimate agent failures. Do not add mandatory repeated proof as onboarding.
For task-create, supply every control receipt and both agent receipts to
`task_pipeline.py verify` with repeated `--evidence` arguments. Missing evidence
fails closed; historical coverage-only reports cannot establish acceptance.

## Repair and retain

Keep failed, interrupted, and superseded runs. Classify the reason: verifier
false acceptance/rejection, instruction ambiguity, setup failure, missing
information, or evidence failure. Separate a legitimate agent capability failure
from a broken task. Fix the cause without weakening justified requirements.
A changed contract needs an explicit explanation; never silently substitute a
different task or reinterpret old rewards to make them pass.

After editing, prepare a new manifest with `--previous <old-manifest> --reason
"<classification and concrete change>"`, then rerun all required controls and
agent evidence for acceptance. The recorder retains digests of prior manifests
and limits a repair chain to three revisions after the initial attempt. Do not
reset the chain to evade that budget. Retain unresolved work when exhausted.
A process killed before it writes a receipt remains missing evidence; preserve
its partial job and use fresh paths for a later attempt.
