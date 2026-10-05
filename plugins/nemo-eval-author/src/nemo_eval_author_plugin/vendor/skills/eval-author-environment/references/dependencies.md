<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dependencies: inventory and realization

Read this during Steps 2 and 3 of [the environment sub-flow](../SKILL.md). It
explains where each fact comes from, how to choose a realization for each
dependency, and how to record what that choice costs in fidelity.

## Inventory

Treat the agent's code as the primary source. Documentation and Ethos describe
intent and can lag the implementation; when they disagree, record both and
resolve the difference in the calling flow's check-in.

| Fact | Where to look | Notes |
| --- | --- | --- |
| Tools the cases exercise | Tool registrations, function schemas, MCP server definitions, agent configuration | Use runtime tool names; audit `tool` items already use them |
| Reads or writes | HTTP methods, SQL statements or ORM calls, MCP `readOnlyHint` and `destructiveHint`, documented side effects | When unknown, plan as a write and say so |
| Backend and location | Client constructors, base URL settings, connection settings, compose files, deployment manifests | Record setting names, never their values |
| Schema and data model | Migrations, ORM models, OpenAPI or JSON Schema documents, protobufs, fixtures | The base population is generated from this |
| Existing environment assets | `Dockerfile`, `docker-compose*.yaml`, devcontainer files, Makefile targets, seed scripts, test factories | Reuse these before writing anything new |
| Real failure behavior | Client error handling, retry and backoff code, recorded traces | Shows which errors the real service returns |
| Credentials and configuration | `.env.example`, settings modules, deployment manifests | Names only; never open `.env` or print values |
| Reset | Migrations plus seed, truncation scripts, vendor sandbox reset tools | Containers reset with each trial; external services do not |

[Execution dependencies](../../eval-author/references/execution-dependencies.md#learn-the-execution-requirements-at-scope)
already records software and location, access path, runtime needs, starting
state, result collection, and trial isolation at scope. Reuse that record and
add only what it lacks: reads or writes, existing assets, failure behavior, and
the realization decision.

## Choose a realization

| Realization | Use when | What it loses | Record |
| --- | --- | --- | --- |
| Real | The repository builds or pulls the service, database, or CLI | Little; versions can drift from production | Image or package version, digest when reproducibility depends on it |
| Vendor sandbox | The vendor offers a test mode and the user authorizes its use | Sandbox limits, data, and behavior can differ from production | Allowed hosts, credential variable names, reset procedure |
| Stateful fake | Nothing real can run reproducibly, but the contract is known | Every operation and error it does not model | Contract source, modeled operations, modeled errors |
| Replay | Read-only calls whose exact inputs appear in recorded traces | Any unrecorded input; all side effects | Trace provenance and the reviewed fixtures |
| None | The cases never need the dependency | Behavior when the dependency is missing | Why the cases do not need it |

### Point the agent's clients at the realization

The agent's own client must reach the backend you provide through
configuration it already supports: a base-URL or host setting, an SDK option
read from an environment variable, or standard proxy variables the client
honors. A vendor sandbox often needs no redirection at all; Stripe's test mode,
for example, uses the production API host with test-mode keys.

When the client hard-codes a production endpoint and supports no override, do
not patch the agent's source or intercept its TLS traffic to force the
connection. Prefer a vendor sandbox that needs no redirection; otherwise record
the dependency as not reproducible with a fake and settle it in the calling
flow's check-in. Adding a supported setting is the user's change to make.

### Real services

Copy the service definitions the cases need from the repository's compose files
into the kit's `build/` directory, keeping their image versions. Run migrations
and seeding when the image is built or the service starts, so every trial begins
from the same state. Avoid named volumes that would carry state between trials.

### Vendor sandboxes

Use a vendor's test mode only after the user authorizes that account and any
cost. Allow only the sandbox's hosts during the agent phase, supply credentials
through variable references, and document how the sandbox is reset between
trials. If reset is impossible, keep trials serial and record the limitation.

### Stateful fakes

Build a fake from the contract: an OpenAPI document, the client library's
request and response types, or recorded traces.

- Keep state across calls within a trial, so a write is visible to a later read.
- Return the real error shapes and codes for the failures the agent must handle:
  not found, validation errors, conflicts, declines, rate limits, partial
  failures. The client's error handling lists which ones matter.
- Model only the operations the cases use, and fail loudly for anything else.
  A clear error for an unmodeled call is better than a silent success that lets
  a wrong action pass.
- Never vary behavior on task identifiers, expected answers, or instruction
  text. The fake implements the service, not the test.
- Make the fake's state exportable so the verifier can read the end state.

### Replay

Trace-environment's reviewed tool-call fixtures and its replay adapter serve
read-only calls whose inputs the cases repeat exactly; that workflow calls this
access state `mock`. See
[Trace-derived tool-call access](../../eval-author-trace-environment/references/trace-derived-fixtures.md).
Replay cannot reproduce side effects, so never use it for a write the verifier
checks.

### Production systems

Avoid them. If the user explicitly authorizes a production endpoint, use
read-only credentials where possible, allow only that endpoint, perform no
writes the user has not approved, and record the risk in the fidelity card.

## Keep the agent out of its own grading

Derive environment behavior and expected outcomes from the systems the agent
acts on and from Ethos. The agent's prompt, planner, and tool-selection code are
what the evaluation measures; copying their assumptions into a fake or an
expected state makes the task agree with the agent by construction.

## Fidelity card

For each dependency, record in the plan:

- its realization and version or contract source;
- evidence of fidelity, such as smoke results or comparisons with recorded
  trace observations;
- known gaps, for example missing webhooks, rate-limit timing, or unmodeled
  endpoints;
- the cases each gap affects.

Task READMEs and run reports link the card, so readers can tell what a pass or
failure does and does not say about production behavior.
