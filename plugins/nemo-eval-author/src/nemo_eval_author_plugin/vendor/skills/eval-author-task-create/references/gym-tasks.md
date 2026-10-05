<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Author and prove a Gym evaluation

Use this provider path to create a task from an existing actionable audit gap
when Gym is selected. It replaces Harbor-specific scaffolding, Oracle, and job
commands; the shared proposal-only boundary, evidence rules, and repeated
measured coverage requirements still apply. Preserve the user's provider choice
and existing suite.

## Version and rerun plan

During proposal work, include this plan in the Gym recommendation in
`.eval-author/proposals/dataset-recommendations.md`. Use the repository's existing
versions and lockfiles when available. Record the intended consumer: native
Gym, Helix Evaluator, or another explicitly selected runner. For each item, give the exact pin or
artifact identity, its evidence source, and whether it is observed, proposed, or
unresolved. Do not fill missing versions with `latest` or imply that a version
range, release label, or proposed configuration is a verified lock.

| Item | What the plan fixes |
| --- | --- |
| Gym and runtime | Gym package version and full source commit or immutable distribution digest; Python patch version and relevant OS/architecture. A dirty checkout needs the retained changes as well as its commit. |
| Dependencies and services | The native dependency manifest and lockfile, their SHA-256 digests, selected extras, and the command that uses the lock without resolving upgrades. Pin container images by digest when used; identify external service versions or record that they cannot be pinned. |
| Task and grading | Resource-server and verifier code, agent code, prompts, configuration overrides, and verifier fixtures, identified by immutable revision or retained files and SHA-256 digests. Include judge model/settings when grading uses a model. |
| Data and initial state | Dataset bytes and SHA-256 digest, split, selected rows and ordering, fixture version, and how to restore the same initial state and isolate each attempt. |
| Model | Provider and exact model snapshot/revision where available, plus sampling settings, token limits, and supported seeds. A mutable model alias remains an explicit reproducibility limit. |
| Run protocol | Working directory, runtime/CLI paths, exact commands, repeat count, concurrency, timeouts, retry policy, supported seeds, and output locations. Record required credential variable names, never secret values. |

Keep this proportional to the proposed task; mark unused components as not
applicable. Missing pins do not prevent a useful proposal: identify the specific
evidence or decision needed to resolve each one. Proposal-only work records the
plan without installing runtimes, changing dependency files, or running evals.

When task creation is requested, carry the selected plan into the draft's
`reproducibility.md`. Reuse the existing package manager's lock format. If the
task has no dependency lock, generate one for its selected dependencies in
a dedicated `reproducibility/locks/` subdirectory under the draft. Keep lock
projects separate from Gym component roots: a root `pyproject.toml` can change
Gym's install mode. Apply [component dependency and packaging checks](gym-packaging.md)
so each server actually consumes the pins. Retain the manifest and lockfile together with a verified command that
consumes them without updates. Preserve source provenance and
local changes needed to reconstruct the task. Keep configuration free of secret
values. Resolve proposed pins against the actual runtime before calling the
environment locked; the scaffolder's `draft.json` records a Gym version but is
not a dependency lock or proof of repeatability.

## Scaffold the native draft

Require an installed Gym v0.6.0+ runtime in its separate Python 3.13.14+
environment. Obtain its Python and CLI paths from the existing installation;
use the installed CLI's `--help` before invoking it. If the runtime is missing,
report the prerequisite and retain the proposal without scaffolding.
Keep each draft in a fresh `.eval-author/task-drafts/<task-slug>/` directory.
For an actionable uncovered tool, the existing selector chooses the same gap:

```bash
python scripts/task_pipeline.py scaffold --provider gym \
  --gym-python /path/to/Gym/.venv/bin/python \
  --report <aggregate-coverage.json> --target <tool-name> \
  --out .eval-author/task-drafts/<task-slug> \
  --task-name <org>/<task-slug> --description '<scenario>' --author '<actual-author>' \
  --instruction-file .eval-author/proposals/<task-slug>-instruction.md
```

The wrapper uses Gym's native `scaffold_environment`, puts the proposed
instruction into its example dataset, and records `draft.json`. Hyphenated task
slugs map to underscored Gym module names. **This is a draft, not a verified task.**
The generated example grader, rewards, and verifier cases are placeholders for
the proposed scenario. Replace them; passing template tests does not prove the
proposal works or exercises its selected tool.

## Complete the task without changing its intended capability

- `environments/<name>/manifest.yaml` identifies the environment, reward contract,
  components, dataset, and provenance. Preserve unknown license or authorship
  until the source or user establishes it; a scaffold is not permission to publish.
- `environments/<name>/config.yaml` composes resources, agent, and model roles.
  Configuration is authoritative; keep manifest mirrors consistent.
- `resources_servers/<name>/app.py` implements the scenario tools, per-session
  state/reset, and verifier. Grade the requested outcome; do not award full reward
  merely for a tool call when the requirement includes arguments or side effects.
  Choose the backends these tools wrap, build their starting data, and prove
  state and reset with the
  [environment sub-flow's Gym guide](../../eval-author-environment/references/gym.md).
- Dataset JSONL rows supply `responses_create_params` and task-specific verifier
  fields. Keep expected answers out of the model input unless the task requires
  them. Distinguish training, validation, and benchmark splits. Document fixture
  origins and preserve native application and access requirements.
- `tests/verifier_cases.jsonl` exercises the actual scenario's positive, negative,
  malformed, and partial-credit cases. Include a no-action/empty-output case and
  a known-correct reference response or interaction. These are verifier controls,
  not measurements of a model's ability.

Inspect the installed Gym base request models and a comparable resources server
when authoring. Do not assume Harbor's task.toml, solve.sh, or reward.txt protocol
applies. Stateful tasks need controls against the same stateful tools and reset
behavior, not just fabricated final responses fed to a stateless verifier.

Maintain the draft root's task-level `README.md` using
[Suite review and rerun instructions](../../eval-author/references/suite-readme.md).
Preserve its existing filename and content when adding review and rerun sections.
Present and link its case inventory when the cases are authored, before execution.
Link `reproducibility.md` for pinned inputs and setup; include concrete full-data
and individual-item commands with the actual dataset selection, composition,
service startup, reset procedure, and fresh output paths. Keep it current through
validation and execution, and link it when handing off the draft.

Before calling the task portable, follow [Component dependencies and packaging](gym-packaging.md)
to check dependency installation and implementation selection in fresh child
environments. Its Helix export section applies only when that consumer is selected.

## Validate, run, and retain evidence

Apply [Task validation and execution evidence](../../eval-author/references/task-validation.md)
for the shared control contract and recorder. Prepare the revision manifest after
completing native files. Use the maintained [native evidence adapter](gym-evidence.md) to retain native
verifier cases, rollout and health records, and verified ATIF conversion receipts. Keep job outputs outside the hashed draft. Report unverified
service deployment identity explicitly; the local digest does not attest a
running remote service. Verifier fixtures alone do not establish task-gap closure.

Before validation or execution, compare the actual runtime, installed dependencies and lock,
task/configuration, and dataset identities with `reproducibility.md`. Resolve
unexpected drift before claiming a comparable rerun. Intentional changes create
a new recorded plan revision; retain earlier run evidence instead of overwriting
it. A fixed environment does not guarantee identical stochastic model outputs.

From the draft root, using the verified Gym CLI:

```bash
/path/to/Gym/.venv/bin/gym env validate --manifest environments/<name>/manifest.yaml --json
/path/to/Gym/.venv/bin/gym env test <name> --json
```

Gym's verifier fixtures provide the known-correct and no-action/negative controls;
Gym does not use Harbor's `-a oracle` or `-a nop` flags. Inspect observed rewards
and errors, including resets and repeatability. A manifest may validate while a
model is still unconfigured; neither validation nor verifier fixtures establish
service readiness or actual agent performance.

Select the repository's actual agent and its required model configuration. Start
and run it through Gym, retaining the complete composition and input identities.
For the native simple-agent composition and an explicit example JSONL file,
start services in one terminal, then collect in another from the same draft root.
Write outputs outside the draft, in a fresh directory per recorded run; `REPO` is
the absolute repository root:

```bash
/path/to/Gym/.venv/bin/gym env start \
  --environment <name> --config /path/to/verified-model-config.yaml \
  +observability_enabled=true

/path/to/Gym/.venv/bin/gym eval run --no-serve --agent <configured-agent-instance> \
  --input environments/<name>/data/example.jsonl \
  --output "$REPO/.eval-author/runs/<task-slug>/<run-id>/rollouts.jsonl" \
  --num-repeats 1 --concurrency 1
```

Use the head-server address from the running composition when it differs from
Gym's default. In v0.6.0, end-to-end `gym eval run` without `--no-serve` uses
prepared `train`, `validation`, or `benchmark` splits; it rejects `--input` and
`--split example`. For that mode, declare repeats in the dataset configuration.
The example above uses an explicit file and `--num-repeats` against running servers.
Each recorded attempt is its own invocation with a new `<run-id>`; task creation
records two.

For each authorized attempt, restore the planned fixture state and use the same
locked inputs and run settings. Retain a copy of the resolved version/rerun plan
with the run's artifacts, including actual identities, overrides, and any
unresolved limits. Check that positive and negative controls behave as planned
after reset, and retain results from every repeat, including errors and retries.
Report the observed variation in rewards and outcomes; repeated execution or a
fixed seed alone does not establish deterministic results. Hand off the exact
rerun command and state-reset steps alongside the results.

Adapt agent selection and dependencies to the inspected repository. Do not
replace the actual agent with a reference implementation to claim gap closure.
Do not disable Gym health checks, discard failed attempts, or mark a missing
model credential as successful execution. Use fresh output paths and retain run
IDs, input digests, Gym revision/version, configured agent/model, rewards,
exceptions, and control results. Only claim a working task after native execution
succeeds; only claim measured gap closure after every required real-agent repeat
closes the selected gap under the existing `verify` command.
