<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor first evals

Use this reference for the Harbor-specific work in first-eval stages 5–9. Keep
the parent skill's stage boundaries, scope, check-ins, and shared deliverables.
Harbor is compatible with Gym; follow the user's selected setup. No audit report
or measured gap closure is required.

## 5. Prepare cases and grading

Use the verified Harbor CLI and Python environment. Read `harbor task init --help`
and `harbor trial start --help` before using options. Use an unused descriptive slug for
each planned task under `.eval-author/task-drafts/`. One task may contain several
examples; preserve the agreed mapping of stable example IDs to tasks.

```bash
harbor task init <org>/<slug> --tasks-dir .eval-author/task-drafts \
  --description "<behavior being tested>" --author "<actual author>"
```

Verify that the expected directory was created. Build its `environment/` first,
from the environment kit proven at stage 5: copy the kit's build files and its
generated data, which passed the kit's smoke task, into the task and add the
case's own records, so the task builds on its own rather than `FROM` a local
kit image. Then
complete the generated `instruction.md`, `task.toml`, `tests/test.sh`, and
`solution/solve.sh` using the installed schema, deriving expected values with an
independent reference query over that data. Set executable permissions,
realistic timeouts, and deterministic rewards; leave no scaffold placeholders.
Introduce the Oracle as the prepared reference solution.

Before testing, document each case's expected NOP and Oracle acceptance criteria
in its README. For a binary completion metric, NOP should score 0 and Oracle 1.
For named or graded metrics, specify the expected values or thresholds for each
relevant metric instead of imposing a universal 0/1 pair. Apply the parent skill's
explicit incorrect-behavior control when inaction itself is correct.

## 6. Prepare the execution environment

Confirm each task's `environment/` builds on the kit that
[`eval-author-environment`](../../eval-author-environment/SKILL.md) built and
proved at stage 5, following its [Harbor environments](../../eval-author-environment/references/harbor.md)
reference, and rerun its smoke task if the kit changed after it passed. Check documented
access requirements, credential variable names, and reset behavior. Keep
solutions and verifier-only data outside the agent's initial environment. For
Docker-backed execution, check `docker info` first.
Verify the selected backend and required application access; record unavailable
access and the cases it blocks.

## 7. Connect the agent

Establish a supported Harbor integration for the actual repository agent from
its entry point, installed adapter code, registry, and CLI help. If the adapter
is missing, identify the specific integration requirement. NOP/Oracle validation
remains available, but performance of the user's agent remains unmeasured.

When supported, write `.eval-author/first-eval.yaml` using the installed Harbor
JobConfig schema. Explicitly select the working starter tasks, actual agent and
model settings, agreed attempts per task, and jobs under `.eval-author/jobs/`.
Use the parent skill's agreed attempt count and cost; do not include unrelated
drafts merely because they share a parent directory. Resolve paths from the
repository root and reference credential environment variables rather than
embedding secrets. Configure request delivery, conversation state, and output
collection through supported interfaces. Configuration is not proof of a working
agent connection.

## 8. Validate the evals

Use Harbor's installed validators to check task validity. Confirm the current
environment kit's smoke proof passed first; a smoke failure is an environment
defect to fix, not a task or agent result, and NOP and Oracle never stand in for
that proof. Before the first control, remove any locally built kit image
(`docker image rm <kit-tag>`), so each task proves it builds from its own
`environment/`. Exercise the environment and grading with NOP and Oracle. Run each
control as a single trial so the parent skill's evidence recorder gets fixed
result paths, retaining all trials under `.eval-author/jobs/` with fresh names
on reruns:

```bash
harbor trial start -p .eval-author/task-drafts/<slug> -a nop \
  --trials-dir "$PWD/.eval-author/jobs/<slug>-controls" --trial-name nop-1
harbor trial start -p .eval-author/task-drafts/<slug> -a oracle \
  --trials-dir "$PWD/.eval-author/jobs/<slug>-controls" --trial-name oracle-1
```

Record each as its own run, with `--result` set to that trial's `result.json`.

For each task, inspect trial results and recorded rewards. Both runs must
complete without exceptions and meet the predeclared criteria, including reward
shape. The reference solution should satisfy the intended outcome; doing nothing
should not earn completion credit when an answer or action is required. If
inaction is correct, exercise an explicit incorrect response or action as well.
Follow the parent skill's limits on validation claims, repairs, and blocked cases.

When the agent config is available, follow `eval-author-discover` to validate it
and report its per-config verdict if other configs exist. Keep unfinished config
validation visible separately from successful control runs. Do not claim a
config is runnable from YAML presence.

## 9. Evaluate the agent

Confirm the resolved task set, actual agent, model, and agreed attempts using the
parent skill's execution checks. With established execution and spend
authorization, run from the repository root:

```bash
harbor job start -c .eval-author/first-eval.yaml
```

Retain outputs, actions, rewards, exceptions, and exact commands for every
selected task under fresh job paths. Low agent scores do not invalidate a working
suite. Return to stage 10 to reconcile delivered cases and explain full-suite and
single-case reruns. Label NOP/Oracle commands as task validation and the actual
agent command as evaluation; use new trial and job names so reruns preserve prior evidence.
