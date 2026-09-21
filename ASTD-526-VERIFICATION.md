<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ASTD-526 — live platform verification

Two runs: one against the developer's real local instance with auth disabled
(resolution and routing), and one against a purpose-built auth-enabled instance
(authorization).

## Run 1 — resolution and routing (auth disabled)

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
| OPA `global_read` rule | yes — see run 2 below | 10 `opa test` cases |
| Read/write split (`get_readable_workspaces`) | yes — see run 2 below | 9 unit tests |

The authorization layer was covered by a second run against an auth-enabled instance.

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

---

## Run 2 — authorization (auth enabled)

A second instance was started on a throwaway data directory with
`NMP_AUTH_ENABLED=true` and `NMP_AUTH_ALLOW_UNSIGNED_JWT=true`, so distinct principals
could be impersonated with `alg=none` JWTs. Role bindings were seeded with auth off, then
the platform was restarted with auth on against the same data directory.

Principals:

| Principal | Bindings |
| --- | --- |
| `admin@test.local` | `PlatformAdmin` in `system` |
| `alice@test.local` | `Editor` in `team-a` **only** — deliberately no binding in `default` |
| `carol@test.local` | `Editor` in `default` |
| `bob@test.local` | none at all |

`alice` is the case the feature exists for: a user with a read right *somewhere* but no
membership in the global workspace. `bob` and the secrets rows are the controls that
distinguish the allowlist-scoped rule from a blanket "everyone can read `default`".

22 scenarios, 22 passed, 0 failed.

| # | Principal | Scenario | Expected | Actual | Result |
| --- | --- | --- | --- | --- | --- |
| P1 | `alice` | GET a model in her own workspace (baseline) | `200` | `200` | PASS |
| P2 | `alice` | GET a GLOBAL model with no binding in default | `200` | `200` | PASS |
| P3 | `alice` | LIST models in the global workspace | `200` | `200` | PASS |
| P4 | `alice` | Her own workspace listing includes global models | `yes` | `yes` | PASS |
| W1 | `alice` | POST a model into the global workspace | `403` | `403` | PASS |
| W2 | `alice` | DELETE a global model | `403` | `403` | PASS |
| W3 | `alice` | PUT (update) a global model | `403` | `403` | PASS |
| W4 | `alice` | POST a model into an unrelated workspace | `403` | `403` | PASS |
| N1 | `alice` | GET secrets in default (permission NOT on the allowlist) | `403` | `403` | PASS |
| N2 | `alice` | GET a non-shareable entity type in default | `403` | `403` | PASS |
| N3 | `alice` | GET a model in an unrelated workspace she has no binding for | `403` | `403` | PASS |
| N4 | `bob` | GET a global model holding NO permission anywhere | `403` | `403` | PASS |
| N5 | `bob` | LIST models in the global workspace with no bindings | `403` | `403` | PASS |
| N6 | `anon` | GET a global model with no token at all | `401` | `401` | PASS |
| X1 | `carol` | GET a model in default (she is Editor there) | `200` | `200` | PASS |
| X2 | `carol` | POST a model into default (Editor may write) | `201` | `201` | PASS |
| X3 | `carol` | GET a model in team-a (no binding there, not global) | `403` | `403` | PASS |
| X4 | `admin` | PlatformAdmin reads the global workspace | `200` | `200` | PASS |
| X5 | `admin` | PlatformAdmin writes to the global workspace | `201` | `201` | PASS |
| X6 | `admin` | PlatformAdmin reads any workspace | `200` | `200` | PASS |
| D1 | `admin` | Deleting a global parent with a foreign child is refused | `409` | `409` | PASS |
| D2 | `admin` | force=true deletes it | `200` | `200` | PASS |

Legend: `P` the feature (reads widen) · `W` writes are not widened · `N` negative controls ·
`X` existing roles unaffected · `D` cascade guard under auth.

`N1` is the load-bearing control: `secrets.read` is not marked `global_read`, so the same
principal that can read a model in `default` cannot read a secret there. `N4`/`N5` are the
other one: holding the permission *somewhere* is required, so a principal with no bindings
gets nothing. Together they show `P2` is the allowlist rule firing rather than a blanket allow.

## Finding: the default wildcard binding defeats acceptance criterion 3

Criterion 3 says *"Only entitled users with write access to default/Global can CRUD models
in Global."* Platform seeding (`seed_default_workspace_editor` in
`services/core/auth/.../app/seeding.py`) creates a role binding for the **wildcard
principal `*` with role `Editor` on the `default` workspace**, documented in that function
as *"gives all authenticated users Editor access to the default workspace."*

The run above deliberately omitted that binding. Adding it back and repeating the write
checks:

| Principal | Bindings | Action | Without wildcard | With wildcard (OOTB) |
| --- | --- | --- | --- | --- |
| `alice` | `Editor` in `team-a` | `POST` model into `default` | `403` | **`201`** |
| `bob` | none at all | `POST` model into `default` | `403` | **`201`** |

So in a stock installation every authenticated user — including one with no role bindings
whatsoever — can create and modify entities in the global workspace. Criterion 3 is not
met, and no change in this PR can meet it while that binding is seeded.

This is pre-existing platform behaviour, not something this PR introduces. What the PR
changes is the blast radius: `default` used to be one workspace among peers, so a
permissive binding on it was contained. Making it the workspace every other workspace
resolves through means that same binding now governs shared infrastructure.

Options, none of which are taken here because the call is not the author's to make:

1. Stop seeding the wildcard `Editor` binding on `default` and require explicit grants.
   Cleanest, but changes onboarding for every existing install.
2. Keep the binding but narrow the role, so the wildcard grants read-only on `default`
   and writes need an explicit binding.
3. Separate the global workspace from `default` entirely, so the permissive default
   workspace and the shared global one are different things.

## Observation: one malformed role binding disables all authorization

While setting up, a hand-written `role_binding` entity missing the required `granted_by`
and `granted_at` fields caused `build_authorization_data` to raise during every policy
refresh. The bundle never loaded, and every request — including `PlatformAdmin` — was
denied with 403 until the row was corrected.

Failing closed is the right direction. Worth noting anyway: a single malformed row, which
the entity store accepts on write because it stores opaque `data`, locks every principal
out of the platform, and the only signal is a Pydantic traceback in the auth service log.
Unrelated to this PR.

## Environment notes

- The sandbox blocks **all** socket binds (`bind()` fails with `EPERM` on any port), so the
  platform has to be started outside it. The CLI reports this as
  `Port 8080 is in use by NeMo Platform instance '<id>'`, attributing an `EPERM` from
  `is_port_bindable` to a stale instance descriptor. The message sends you after a lock
  that is not the problem.
- Run 2 used a throwaway `NMP_DATA_DIR` so that seeded role bindings never touched the
  developer's real local database.
