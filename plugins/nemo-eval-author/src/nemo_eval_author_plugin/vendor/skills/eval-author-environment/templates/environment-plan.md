<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Environment plan: <agent name>

- **Status:** <draft | proven | unproven | blocked | failed> (`proven` only when the Proof table names the smoke task's own jobs)
- **Provider:** <Harbor or Gym> <installed version>
- **Kit:** `.eval-author/environments/<agent-slug>/`
- **Calling flow and cases:** <first-eval or task-create>, <case IDs>
- **Repository revision:** <commit, or uncommitted with a note>

## Fit

- **Agent shape:** <runs in a sandbox | application whose tools call backends | hosted agent on hosted APIs>
- **Evidence:** <entry point, deployment files, Ethos Harness section>
- **Decision:** <fits | fits with changes | does not fit, with the route or limit>

## Inventory

| Dependency | Touches | Reads or writes | Location and access | Existing assets | Reset | Unknowns |
| --- | --- | --- | --- | --- | --- | --- |
| <name> | <files, database, service, external API> | <read, write, both> | <where it runs, how it is reached> | <Dockerfile, compose, migrations, spec> | <how state resets> | <open questions> |

## Fidelity card

| Dependency | Realization | Source or version | Fidelity evidence | Known gaps | Cases affected |
| --- | --- | --- | --- | --- | --- |
| <name> | <real, vendor sandbox, stateful fake, replay, none> | <image, contract, trace provenance> | <smoke results, trace comparisons> | <what is not reproduced> | <case IDs> |

## Starting data

- **Generator, seed, and digests:** <path, seed, `data/digests.json` (verifier-only, per table)>
- **Base population:** <entities and counts, and where the sizing came from>
- **Per-task records:** <where each task's records live>
- **Decoys:** <the condition each serves; where the verifier-only labels live>
- **Integrity checks:** <script path and latest result>
- **Unresolved end-state questions:** <items raised in the check-in, or none>

## End-state export

- **What reaches the verifier:** <artifacts, collect hooks, or session state>
- **Ignored volatile fields:** <timestamps, generated identifiers>

## Isolation

- **Network:** environment <mode>; agent <mode and allowed hosts>; verifier <mode>
- **Credentials, by variable name:** <names only>
- **Hidden answers:** <where expected state, decoy labels, and solutions live>
- **Reset of external services:** <command, or not applicable>

## Proof

Harbor rows name jobs of the smoke task in `smoke/`; for Gym, list the
persistence and reset verifier cases instead. A task's own NOP, Oracle, or
verifier controls never fill these rows. Leave a row's result as `not run` and
the status `unproven` until its job exists.

| Check | Command or job | Result | Evidence path |
| --- | --- | --- | --- |
| Smoke NOP | <command> | <result> | <path> |
| Smoke reference solution, first run | <command> | <result> | <path> |
| Smoke reference solution, second run | <command> | <result> | <path> |

- **Agent tool discovery:** <proven in run X | unproven, with the reason>
- **Repairs:** <count and reasons, at most three>

## Reuse

- **How tasks use the kit:** <base image reference or copied build files; where per-task records go>
- **Re-prove when:** the repository revision, data generator, provider version, or a dependency choice changes.
