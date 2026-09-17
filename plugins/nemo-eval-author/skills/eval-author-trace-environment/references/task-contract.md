<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Task contract: layout, isolation, and README

The Harbor task lives under `<task-dir>/task/` and contains:

- `task.toml` with realistic timeouts and resources;
- `instruction.md` matching `candidate.json` without leaking tool names or test
  logic;
- `environment/` containing prerequisites but never the solution;
- `tests/test.sh` grading only the observable outcome, following
  `references/check-grammar.md`;
- `solution/solve.sh` with the reference solution; and
- `README.md` with reviewer-facing development context, not a copy of the
  agent instruction.

## Verifier isolation contract

`task.toml` must explicitly set `[verifier].environment_mode = "separate"`,
`[verifier].network_mode = "no-network"`, a `[verifier.environment]` table
whose `network_mode` is also `"no-network"`, and `[environment].network_mode
= "no-network"`. The helper rejects shared verification because it lets the
grader observe or alter the agent container and can expose hidden tests. When
the grader needs a different image, provide a verifier-owned `tests/Dockerfile`
or a pinned verifier image through `[verifier.environment]` with its complete
grading dependency closure and `/tests` tree.

For a multi-step task, every step inherits the top-level separate verifier.
An explicit `[steps.verifier]` override must not select `shared`, and a
`[steps.verifier.environment]` must also set `network_mode = "no-network"`.
Transfer only the declared agent-produced artifacts into the verifier
environment; never mount or copy the agent workspace wholesale. Add a
step-local environment only when that step needs a different verifier image or
resource configuration.

## Reviewer-facing README

The task README is not passed to the agent. Give it a level-one task title
(reconstructed tasks name the reconstruction in `Ground-truth provenance`) and
these substantive level-two sections:

- `Difficulty explanation`: why the task is difficult for agents and humans;
- `Environment and software requirements`: runtimes, services, hardware,
  versions, licensing, and availability constraints;
- `Ground-truth provenance`: what establishes correctness and where that
  evidence came from, without exposing private values;
- `Solution explanation`: the high-level reference approach without duplicating
  `solution/solve.sh`;
- `Verification explanation`: the observable outcomes and how the verifier
  distinguishes success from failure; and
- `Relevant experience`: human-supplied experience relevant to authoring or
  reviewing the task.

Keep each section concise and evidence-backed. Do not repeat `instruction.md`,
reveal verifier internals to the agent, or invent author experience. If a human
cannot supply and review `Relevant experience`, keep the environment `unproven`
rather than claiming it is ready. Do not copy private trace payloads into the
task: include only the minimal files needed to reproduce the starting state,
pinning external source to an exact public commit when truly required and
otherwise preferring a small local fixture.
