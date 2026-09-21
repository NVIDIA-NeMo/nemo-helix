<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ASTD-526 — live platform verification

Manual verification of global-workspace sharing against a running local NeMo Platform,
in addition to the automated unit and integration suites.

## Environment

| Item | Value |
| --- | --- |
| Services | `entities`, `models`, `inference-gateway`, `secrets`, `files` (+ `models` controller) |
| Health | `/health/ready` → `{"status":"ready"}`; all 5 services ready, controller healthy |
| Auth | **disabled** (local default) |
| Pre-existing data | 184 models and 108 virtual models in `default`; providers `nvidia-build` (82 served models), `spark1` (25), `spark1-glint` (1); workspaces `default`, `system`, `mschwab` |
| Test isolation | A per-run workspace `astd526-team-$(date +%s)`; all fixtures prefixed `astd526x` and removed afterwards |

Tests were run against real seeded data rather than an empty database, so the
regression rows (`G*`) exercise genuine pre-existing entities.

## What this does and does not cover

Auth is disabled on a local instance. With auth off, `get_accessible_workspaces`
returns `None` (unrestricted), so `expand_readable_workspaces` passes through and the
OPA policy never evaluates. This run therefore exercises **resolution, listing, the
cascade guard, and inference-gateway routing**, but **not** the authorization layer:

| Layer | Covered here | Covered elsewhere |
| --- | --- | --- |
| Entity resolution + listing | yes | entity-store unit + integration tests |
| Cascade guard on delete | yes | 6 integration tests |
| Inference gateway routing | yes | 14 unit tests |
| OPA `global_read` rule | **no — auth disabled** | 10 `opa test` cases |
| Read/write split (`get_readable_workspaces`) | **no — auth disabled** | 9 unit tests |

Verifying the authz layer end to end needs an instance with auth enabled and role
bindings seeded; that is the main gap in this run.

## Results

32 scenarios, 32 passed, 0 failed.

| # | Scenario | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| R1 | GET a global model from another workspace | `200` | `200` | PASS |
| R2 | Global model reports its owning workspace, not the caller's | `default` | `default` | PASS |
| R3 | Non-shareable type in default does NOT leak cross-workspace | `404` | `404` | PASS |
| R4 | Workspace-private model does NOT leak to another workspace | `404` | `404` | PASS |
| R5 | Unknown name still 404s (fallback is not a wildcard) | `404` | `404` | PASS |
| R6 | Local entity shadows the global one (local-wins precedence) | `astd526-team-1790033684` | `astd526-team-1790033684` | PASS |
| R7 | Shadowing does not disturb the global entity itself | `default` | `default` | PASS |
| L1 | Listing a workspace includes global entities | `yes` | `yes` | PASS |
| L2 | Listing does NOT include unrelated workspaces | `no` | `no` | PASS |
| L3 | Listing a non-shareable type stays workspace-scoped | `no` | `no` | PASS |
| L4 | Listing the global workspace itself is not duplicated | `1` | `1` | PASS |
| G1 | Pre-existing models service still lists default (184 seeded + test rows) | `ok` | `ok` | PASS |
| G2 | Models service GET on a real pre-existing model still works | `200` | `200` | PASS |
| G3 | Models service GET a real global model from another workspace | `200` | `200` | PASS |
| G4 | Entity update in own workspace unaffected | `200` | `200` | PASS |
| S1 | Filesets are NOT shared (scoped out to ASTD-640) | `404` | `404` | PASS |
| S2 | Fileset listing stays workspace-scoped | `no` | `no` | PASS |
| C1 | Cross-workspace adapter on a global base model can be created (AC #2) | `200` | `200` | PASS |
| C2 | Deleting a global parent with foreign children is refused | `409` | `409` | PASS |
| C3 | The 409 names the affected workspace | `yes` | `yes` | PASS |
| C4 | Refused delete leaves the parent intact | `200` | `200` | PASS |
| C5 | Refused delete leaves the foreign child intact | `200` | `200` | PASS |
| C6 | force=true performs the delete | `200` | `200` | PASS |
| C7 | force=true actually cascaded the foreign child away | `404` | `404` | PASS |
| C8 | Same-workspace children do NOT block delete (no new false refusal) | `200` | `200` | PASS |
| C9 | Childless delete is unaffected | `200` | `200` | PASS |
| I1 | IGW lists models from a workspace that owns none (global fallback) | `ok` | `ok` | PASS |
| I2 | IGW listing reports real provenance in owned_by | `yes` | `yes` | PASS |
| I3 | IGW ids are addressed in the request workspace (round-trippable) | `astd526-team-1790033684` | `astd526-team-1790033684` | PASS |
| I4 | IGW GET on a global model resolves from the request workspace | `200` | `200` | PASS |
| I5 | IGW listing of the global workspace is not duplicated | `108` | `108` | PASS |
| A1 | force parameter is published in the OpenAPI spec | `yes` | `yes` | PASS |

Legend: `R` resolution · `L` listing · `G` regression against pre-existing data ·
`S` scope (filesets excluded) · `C` cascade guard · `I` inference gateway · `A` API surface.

## Post-run state

Cleanup verified: no `astd526*` entities remain in any workspace, the workspace list is
back to `default`, `system`, `mschwab`, and `default` again reports exactly 184 models —
the pre-test count.

## Harness notes

Two earlier runs produced misleading results and were corrected rather than reported:

1. The first run pointed the gateway checks at `/apis/inference/v2/...`; the real prefix
   is `/apis/inference-gateway/v2/...`. The wrong path returned no `data` key, which the
   parser turned into a count of 0 and four apparent gateway failures. Those were harness
   bugs, not product behaviour.
2. The second run reused a fixed workspace name. Workspace deletion is staged rather than
   immediate, so re-creating it silently failed and every entity assertion in that
   workspace 404'd. The harness now uses a per-run name and **asserts its own setup**,
   aborting rather than reporting vacuous passes.

The second failure is the interesting one: without the setup assertion, a suite whose
fixtures never got created would report passes for every "does not leak" row — the
negative assertions all succeed trivially when nothing exists.
