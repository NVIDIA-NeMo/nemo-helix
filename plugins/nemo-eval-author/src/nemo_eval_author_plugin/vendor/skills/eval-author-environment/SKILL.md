---
name: eval-author-environment
description: >-
  Required by Eval Author's first-eval and task-create flows before they write
  a Harbor or Gym task's tests, and available to trace-environment: builds the
  environment the task runs in, meaning the agent's databases, files, services,
  and tools, seeded realistic starting data, and how the end state is read back
  for grading. Checks provider fit, inventories the agent repository read-only,
  chooses real, sandboxed, faked, or replayed dependencies, generates starting
  data, and proves the environment with a smoke task before tasks rely on it.
  Reached through eval-author and its sub-flows rather than invoked directly.
triggers:
  - eval-author routed to environment preparation
  - prepare the environment for the selected eval tasks
  - continue eval-author environment setup
not-for:
  - eval-author (use for the standard, the boundaries, and to pick a sub-flow)
  - eval-author-trace-environment (experimental; use to derive a whole task from one recorded trace)
  - eval-author-task-create (use to propose dataset improvements and create audit-gap tasks)
  - eval-author-first-eval (use to plan and build a starter suite)
  - eval-author-discover (use to check whether an existing suite runs)
compatibility: >-
  Reads the agent repository locally. Harbor environments need the Harbor CLI
  (0.20.0 or later) and Docker with Compose for sidecar services; Gym
  environments need an existing Gym v0.6.0+ runtime in Python 3.13.14+. Vendor
  sandboxes use the user's credentials, referenced by variable name only.
  Writes only under `.eval-author/`.
maturity: alpha
license: Apache-2.0
user-invocable: false
allowed-tools: Bash Read Write Grep Glob
---
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Eval Author: environment

## Required outputs

First-eval and task-create are not finished with an environment until the kit
under `.eval-author/environments/<agent-slug>/` holds:

- `environment-plan.md`, started from [the template](templates/environment-plan.md),
  with fit, inventory, dependency choices, fidelity card, and proof status;
- a seeded generator in `data/` that produces the starting data, with its
  integrity check and per-table content digests that verifiers use to check
  preserved tables. Size it like the data the agent meets in practice, large
  enough that the agent must search and choose, with the variety real data has:
  mixed statuses, similar records, and edge values;
- the smoke task in `smoke/` with its NOP run and two reference-solution runs.
  The plan may say `proven` only when it names those three jobs.

Start the plan from its template at Step 1 and keep it current; it is a working
record, not a summary written at the end.

Build and prove the kit before the tests: generate the starting data and pass the
smoke task first, then derive every expected value with an independent reference
query over that data. A few
hand-typed fixture rows with hand-computed answers are not starting data, and a
task's own NOP, Oracle, or verifier controls never prove the environment.

## Purpose

Read `eval-author` for the shared evidence standard and boundaries. Start this
sub-flow as soon as an authoring flow has agreed its cases and before it writes
their tests: first-eval at **Prepare cases and grading**, or task-create before
it writes a draft's verifier. Steps 1 and 2 happen at scope; Steps 3 to 7 build
and prove the kit before the calling flow writes any test; Step 8 hands it off.
It builds the world those tasks run in and proves that world works before any
task depends on it.

A task is only as informative as its environment. When the starting data holds
just the records a case needs, the agent succeeds without searching, choosing,
or checking anything. When dependencies are faked to always succeed, the agent
never meets the failures it sees in production. When the end state cannot be
read back, nothing confirms the agent did the work. Each step below exists to
prevent one of those outcomes.

This sub-flow owns provider fit, how each dependency is provided, the starting
data and its reset, how the end state reaches the verifier, isolation, the
environment's smoke proof, and the reusable kit. The calling flow keeps case
design, instructions, grading criteria, the agent connection, and per-task
controls such as Harbor's NOP and Oracle runs. Which difficult conditions a case
plants is a case-design decision; this sub-flow makes the records those
conditions need consistent, reproducible, and invisible to the agent.

## Inputs and outputs

Carry these from the calling flow instead of asking again:

- the applicable Ethos path and the requirements the cases test;
- the agreed cases, their expected outcomes, and any per-task records they name;
- the provider (Harbor or Gym) and the existing suite layout;
- requirements recorded at scope with
  [Execution dependencies](../eval-author/references/execution-dependencies.md);
- installation, execution, and spend authorizations already given.

Build one environment kit per agent and reuse it for every task of that agent:

```text
.eval-author/environments/<agent-slug>/
  environment-plan.md   fit, inventory, choices, data, export, isolation, fidelity card, proof
  build/                base Dockerfile and docker-compose.yaml, or Gym backend code
  data/                 seed generator, its seed, generated base data, integrity checks
  smoke/                the smoke task and its retained results
```

Start `environment-plan.md` from [the template](templates/environment-plan.md)
and keep it current; it is how the calling flow, task READMEs, and later runs
learn what the environment does and does not reproduce. When a kit already
exists, reuse it. Re-prove it only when the repository revision, the data
generator, the provider version, or a dependency choice changes.

## Step 1: Check fit

Classify how the agent runs from its entry point, deployment files, and the
Ethos Harness section:

| Agent shape | Harbor fit | Approach |
| --- | --- | --- |
| Runs inside a sandbox: coding, CLI, or computer-use agents | Direct | Files, services, and tools the agent uses live in the task container and its sidecars |
| An application whose tools call backends, such as a framework agent | Good when the application runs in a container | Run the application as the agent in Harbor's `main` container; replace its backends with sidecars and point its clients at them |
| A hosted agent acting on hosted product APIs, with no runnable code | Poor | Harbor's own guidance says it may not be the right fit; consider Gym, a runnable build, a vendor sandbox, or report the limit |

For the last shape, do not build a Harbor task that only forwards a request to
a hosted endpoint and inspects a hosted result afterwards. It controls neither
the starting state nor trial isolation, and it can write to production. Gym
helps only if the agent's tool calls can be pointed at tools the environment
provides. Record the decision and its evidence in the plan, and raise a fit
problem in the calling flow's current check-in. A poor fit is a proposal under
the core's [provider-selection rule](../eval-author/SKILL.md#select-the-evaluation-provider):
the user decides whether to change provider, and the calling flow verifies the
new runtime before continuing. Never switch providers silently.

## Step 2: Inventory the repository

Read [Dependencies](references/dependencies.md#inventory) and record, for each
tool and service the agreed cases exercise:

- what it touches: files, a database, an in-repository service, or an external
  API;
- whether it reads or writes, judged from the code: HTTP methods, SQL or ORM
  calls, MCP `readOnlyHint` annotations, documented side effects;
- where it runs, how it is reached, and what reset requires;
- assets the repository already has: Dockerfiles, compose files, migrations,
  API specifications, fixtures, seed scripts;
- the failures its client code handles, which reveal how the real service
  behaves;
- unknowns, with what would resolve them.

This is read-only work. Read configuration variable names from examples,
settings modules, and code; never open `.env` files or print secret values.
Copy any file the kit needs into `.eval-author/environments/`; never edit, move,
or reformat the user's files.

## Step 3: Choose how to provide each dependency

Prefer the most faithful option that runs reproducibly:

1. **Real:** the repository's own service, database image, or CLI, run in the
   task container or as a sidecar.
2. **Vendor sandbox:** the provider's test mode, used only with the user's
   authorization, behind a network allowlist, with credentials supplied by
   variable name.
3. **Stateful fake:** a small service built from the API contract that keeps
   state across calls and returns the real error shapes.
4. **Replay:** recorded responses from traces, for read-only calls whose inputs
   the cases repeat exactly.
5. **None:** the cases never need the dependency.

Each step down loses fidelity in a predictable way, and
[Dependencies](references/dependencies.md#choose-a-realization) describes what
each one loses. Record every choice and its gap in the plan's fidelity card;
task reports link it so nobody reads a pass against a fake as a pass against
production.

Provide backends, not tools. Harbor treats tools as part of the agent, so the
agent's own tool code should reach the backend you supply, through configuration
its client already supports;
[Dependencies](references/dependencies.md#point-the-agents-clients-at-the-realization)
covers clients that hard-code an endpoint. Gym's resources server owns the tool
surface instead; wrap the same backend in endpoints that copy the agent's real
tool names, arguments, and errors.

Derive the environment's behavior from the systems the agent acts on and from
Ethos, never from the agent's prompt or tool-selection logic; otherwise the
task grades the agent against itself. When no faithful option exists, for
example a production-only system nobody has authorized for testing, explain the
gap and settle it in the calling flow's check-in rather than substituting a mock
that hides it.

## Step 4: Build the starting data

Read [Starting data](references/starting-data.md) before generating records.

- Generate a base population in code from the repository's schema with a fixed
  seed. Size and shape it like the data the agent meets in practice, using the
  selected traces when available, and materialize it in `data/`.
- Layer each case's own records over the base, keeping them with the task.
- When a case names a condition that needs confusable records, such as a
  similar customer or an ineligible order, build them as additive decoys that a
  careful agent can tell apart through permitted reads, and keep their labels
  verifier-only.
- Check integrity before any task relies on the data: references resolve,
  statuses permit the expected actions, totals recompute, and identifiers the
  system assigns at runtime are not pre-created.
- Record verifier-only, per-table content digests of the starting data, so each
  task checks what it must preserve by content rather than by row counts.
- Derive each case's expected end state from its Ethos-backed outcome applied to
  this data, with an independent reference query over the generated records
  rather than arithmetic over a few rows, before the verifier is written. When
  Ethos does not settle what is correct, mark it unresolved and raise it in the
  calling flow's check-in instead of choosing.

## Step 5: Make the end state readable

Graders need the resulting state, not the agent's account of it. Declare exactly
what reaches the verifier and which volatile fields, such as timestamps and
generated identifiers, it should ignore.

- **Harbor:** export state from the agent container and sidecars, then grade it
  in a separate, offline verifier. Follow
  [Harbor environments](references/harbor.md#export-the-end-state).
- **Gym:** the verifier runs in the resources server and reads session state.
  Follow [Gym environments](references/gym.md#state-and-verification).

A missing or unreadable export is an infrastructure failure, never evidence that
the agent did or did not act. Each task's side-effect control in
[Task validation and execution evidence](../eval-author/references/task-validation.md)
reads this export as its trusted observation.

## Step 6: Isolate the run and protect hidden answers

- Restrict the network by phase. The agent reaches only what it needs during its
  run, normally its model provider and any authorized vendor sandbox. The
  verifier has no network. When the runtime cannot enforce a policy, record
  isolation as unproven instead of quietly running with public access.
- Reference credentials by variable name. Never write secret values into the
  kit, task files, plans, or reports.
- Keep expected state, decoy labels, reference solutions, and verifier helpers
  out of everything the agent can see: images, mounted files, tool outputs, and
  dataset fields it receives. This covers the shared kit; each task's own
  leakage inspection stays with the calling flow's task validation.
- Start every trial from the seeded state. Resetting containers does not reset
  external services; document their reset and keep such trials serial until
  isolation is proven.

## Step 7: Prove the environment

Prove the environment once, right after building it and before any task's tests
are written:

- **Harbor:** run the smoke task in `smoke/`. Its reference solution performs
  one read and one write per dependency through the agent's own client code,
  and its verifier checks that the exported state shows exactly those changes.
  Run NOP once and the reference solution twice; the second run passing shows
  each trial starts clean. See [Harbor environments](references/harbor.md#smoke-task).
- **Gym:** add verifier cases for state that persists within a session and
  resets between sessions, then run the native validation in
  [Gym environments](references/gym.md#prove-the-environment).
- **Agent-provided discovery:** Harbor passes MCP server configuration to the
  agent but does not prove the agent connected. Confirm discovery and calls in
  one run of the actual agent, which requires the existing authorization for
  agent execution and spend; until then, record discovery as unproven.

Record each proof run's command, job path, and result in `environment-plan.md`.
These runs prove the environment, not a task: do not declare them as a task's
controls or count them among its evidence receipts. Likewise, a task's own NOP,
Oracle, or verifier controls never substitute for this proof.

Treat setup failures as infrastructure, not agent results. Fix and rerun them,
recording each repair and its reason in the plan as
[Execution recovery](../eval-author/references/execution-recovery.md) describes,
with at most three repairs before reporting the environment as failed. That
budget is separate from any task's revision chain, and diagnostic probes never
complete the proof. Do not weaken a smoke check to make it pass. Record the
outcome as `proven`, `unproven` (prerequisites missing or checks not run),
`blocked` (a required dependency cannot be provided), or `failed`. `proven`
requires the smoke task's own three jobs (for Gym, the persistence and reset
cases) in the plan's proof table; without them the outcome is `unproven`.

## Step 8: Hand off the kit

- Finish `environment-plan.md`: choices, fidelity card, proof results, and
  evidence paths.
- Add setup, service startup, credential variable names, and reset steps to the
  suite's review and rerun guide using
  [Suite review and rerun instructions](../eval-author/references/suite-readme.md),
  and list the fidelity-card gaps that affect each case among its known
  limitations in the case inventory.
- Tell the calling flow how its tasks use the kit: a Harbor task copies the
  kit's build files and data into its own `environment/` (or uses a pushed image
  pinned by digest) and adds its own records, so it builds without the kit image
  this session built; a Gym task reuses the backend and seeds its records per
  session.
- Return to the calling step. A proven environment shows the world works; it
  does not validate a task's grader or measure the agent.

## Prerequisites

Planning and inventory need only repository access. Building and proving a
Harbor environment needs the Harbor CLI and Docker with Compose; Gym needs its
separate runtime. Vendor sandboxes need the user's authorization and
credentials, referenced by variable name. Agent runs used to prove tool
discovery require authorization for execution and spend.

## Limitations

The environment reproduces what the repository, traces, and authorized
sandboxes reveal. Stateful fakes model only the operations and errors they
implement, and replay covers only recorded inputs; the fidelity card names those
gaps. A proven environment does not establish task correctness, grader
soundness, or agent performance.

## Troubleshooting

- Smoke reference solution fails: inspect service health, seeding, and the
  export before touching the smoke checks; fix the environment, then rerun.
- Sidecars or collect hooks unsupported: the selected sandbox may lack Compose
  support; record the limit or choose a supported backend.
- `no-network` or `allowlist` rejected before anything runs: the runtime cannot
  enforce that policy, which is common on Docker Desktop. Follow the fallback in
  [Harbor environments](references/harbor.md#network-policy) and record
  isolation as unproven; never relax a policy silently.
- A dependency only exists in production: settle it in the calling flow's
  check-in; never point a task at production without explicit authorization.
