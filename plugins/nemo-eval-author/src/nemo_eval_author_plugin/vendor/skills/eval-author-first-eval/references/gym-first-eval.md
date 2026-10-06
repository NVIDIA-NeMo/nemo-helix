<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native Gym first evals

Use this reference for the Gym-specific work in first-eval stages 5–9. Keep the
parent skill's stage boundaries and shared deliverables. Gym is compatible with
Harbor; follow the user's selected setup. No audit report, actionable tool gap,
or measured gap closure is required.

## 5. Prepare cases and grading

Use the verified Gym v0.6.0+ installation in its separate Python 3.13.14+
environment. Inspect its CLI help and native scaffolder before use. Create each
planned draft in a fresh `.eval-author/task-drafts/<slug>/` directory using
Gym's `nemo_gym.environment.scaffold.scaffold_environment(kind="environment",
name=<module_name>, root=<draft_path>)`. Use a valid Python module name for the
Gym environment. Do not call the audit-driven `task_pipeline.py scaffold` route:
first-eval has no coverage report or uncovered-tool selector.

Follow [Complete the task without changing its intended capability](../../eval-author-task-create/references/gym-tasks.md#complete-the-task-without-changing-its-intended-capability)
for the generated manifest, component configuration, resources server, dataset
rows, and verifier fixtures. Replace template tools, rewards, and fixtures with
the agreed cases. Seed the resources server from the environment kit's generated
data and pass the kit's persistence and reset cases before writing verifier
fixtures, then derive expected rewards and end states with an independent
reference over that data. Map stable case IDs to selected
dataset rows; keep expected answers and provenance outside model inputs. Start
with one representative case before extending the pattern to the agreed breadth.

Document each case's reward contract and expected positive and negative control
results before testing. Include a known-correct response or interaction and a
no-action case; when inaction is correct, also include an explicitly incorrect
response or action. Explain the grader's assertion and a limitation to the user.
Maintain each task README and the shared `.eval-author/README.md`.

## 6. Prepare the execution environment

Prepare the generated resources, agent, and model components, dependency inputs,
fixtures, and state reset. Build the backends the resources server wraps, their
seeded session state, and its reset from the environment kit that
[`eval-author-environment`](../../eval-author-environment/SKILL.md) built and
proved at stage 5, following its [Gym environments](../../eval-author-environment/references/gym.md)
reference, and add each case's own records to its dataset row or session seed.
Apply the [Gym version and rerun plan](../../eval-author-task-create/references/gym-tasks.md#version-and-rerun-plan)
to the selected suite, recording it in `.eval-author/first-eval.md` and the drafts'
`reproducibility.md`; no dataset recommendation or audit proposal is required.
Use the component packaging checks linked there for the selected runtime.
Verify actual application access and component setup. Record missing credential
variable names and blocked cases without recording secret values.

## 7. Connect the agent

Use the actual repository agent through a supported Gym integration. Inspect
its entry point and the installed Gym interfaces; do not substitute Gym's example
agent to claim the user's agent ran. Record the resolved composition, configured
agent instance, model settings, selected dataset rows, reset procedure, and
output paths. If integration is missing, retain the drafts and verifier work and
name the specific connection required.

Propose one attempt per selected example for the initial baseline unless the
agreed purpose requires repeats. Record the count and cost in the plan. A Harbor
JobConfig or `.eval-author/first-eval.yaml` is not required for Gym.

## 8. Validate the evals

Confirm the environment kit's proof passed first; a proof failure is an
environment defect to fix, not a task or agent result. Its persistence and reset
verifier cases run with the commands below and are recorded in the kit's plan,
not declared as task controls.

From each draft root, use the verified Gym CLI:

```bash
/path/to/Gym/.venv/bin/gym env validate --manifest environments/<name>/manifest.yaml --json
/path/to/Gym/.venv/bin/gym env test <name> --json
```

Apply [Task validation and execution evidence](../../eval-author/references/task-validation.md)
to retain the task revision and native control results. Include the declared
incorrect, valid-alternative, and side-effect cases, with explicit reasons for
non-applicability. Record the verifier controls with the shared recorder and the
maintained [native Gym adapter](../../eval-author-task-create/references/gym-evidence.md);
this does not require an audit report or agent baselines.
Retain validation and verifier-control output under fresh run paths. Require the
controls to complete without exceptions and meet their predeclared reward
criteria, exercising real tools and reset behavior for stateful cases. Repair
broken cases without weakening assertions. Report blocked cases separately.
Gym has no Harbor `-a nop` or `-a oracle` requirement. Passing verifier fixtures
does not establish actual agent performance or broad coverage.

## 9. Evaluate the agent

Follow the native service startup and execution instructions in
[Validate, run, and retain evidence](../../eval-author-task-create/references/gym-tasks.md#validate-run-and-retain-evidence),
using the actual agent, selected rows, and agreed attempts. Its two recorded
attempts and measured-gap-closure requirements belong to audit task creation;
first-eval uses the agreed baseline count and does not require coverage measurement.
Check installed help before using options. Preserve the distinction between
explicit-file execution against running services (`--no-serve --input`) and
managed execution using prepared splits.

Run only with the established execution and spend authorization. Retain fresh
outputs, rewards, errors, resolved settings, and exact commands for every selected
case under `.eval-author/runs/`. Link these from the plan and suite README using
paths resolved from the documented working directory. A working suite requires
native execution of the actual agent and recorded rewards, even when scores are
low. Return to stage 10 for reconciliation and rerun instructions. Report Gym
control results in their own terms, without labeling them Harbor NOP/Oracle jobs.
