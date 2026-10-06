<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gym environments

Read this when Gym is the provider. The decisions in Steps 1 through 4 of
[the environment sub-flow](../SKILL.md) are unchanged; construction differs in
one structural way. In Gym, the environment owns the tool surface: the
resources server implements the tools, keeps per-session state, and runs the
verifier.

For scaffolding, the version and rerun plan, native validation commands, and
real-agent runs, follow
[Native Gym first evals](../../eval-author-first-eval/references/gym-first-eval.md)
from first-eval or
[Author and prove a Gym evaluation](../../eval-author-task-create/references/gym-tasks.md)
from task-create, with
[Component dependencies and packaging](../../eval-author-task-create/references/gym-packaging.md).
For trace-derived output, follow
[the Gym extension](../../eval-author-trace-environment/references/gym-output.md).
Do not apply Harbor layouts, `task.toml` fields, or NOP and Oracle flags to Gym.

## Tools wrap the realized backend

- Implement each tool endpoint in `resources_servers/<name>/app.py` as a thin
  adapter over the backend chosen in Step 3: the repository's real service, an
  authorized vendor sandbox, or a stateful fake. Keep domain rules in the
  backend instead of re-implementing them inside endpoints.
- Copy the agent's real tool names, argument schemas, and error shapes into the
  endpoints and into each row's `responses_create_params.tools`. A renamed or
  simplified tool changes what the evaluation measures.
- Start or embed the backend as part of the composition, so runs and grading do
  not depend on live systems. Record how it starts in `reproducibility.md`.

## State and verification

- Seed each session's state from the kit's base data plus the row's own
  records when the session starts. Never share mutable state across sessions.
- The verifier runs in the resources server and can read session state
  directly. Grade the end state the case expects; a tool call alone does not
  establish a side effect. Compare state the case must leave unchanged with the
  kit's per-table digests from [Starting data](starting-data.md#digests-for-preservation-checks).
- Keep expected state and decoy labels in verifier-only row fields or
  server-side data. Check what the selected agent actually receives, including
  tool outputs; a separate JSONL key alone does not prove isolation.
- Gym's HTTP services are not Harbor's separate, offline verifier. Describe the
  actual trust boundary in the plan instead of claiming Harbor-style isolation.

## Prove the environment

- Add verifier cases to `tests/verifier_cases.jsonl` beyond the usual
  known-correct and no-action controls: a write followed by a read in the same
  session, showing state persists, and a fresh session showing it starts from
  the seed with no leakage from the previous one.
- Run Gym's native validation and tests from the draft root. From task-create,
  also repeat the positive and negative controls at least twice from restored
  state, as its Gym authoring guide describes. From first-eval, the fresh-session
  case is the reset proof; repeated proof is not an onboarding requirement.
- These controls establish environment and verifier behavior, not model
  performance. Record their commands, results, and evidence paths in the plan.
  The persistence and reset cases prove the environment, so do not declare them
  as a task's controls in its
  [native Gym evidence](../../eval-author-task-create/references/gym-evidence.md)
  contract.
