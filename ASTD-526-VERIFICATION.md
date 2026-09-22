<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ASTD-526 — live platform verification

Verification of global-workspace sharing against a running local NeMo Platform, in
addition to the automated unit and integration suites.

## What is being verified

The point of the feature is that a workspace can **use a model that already exists in the
global workspace instead of deploying its own copy**. That is a name-resolution problem,
not an authorization one, and the design reflects that:

- A shared entity **resolves by name** from another workspace, so a reference to it works
  without the entity being duplicated.
- Resolution is **gated on being entitled to the global workspace**. A caller scoped only
  to their own workspace sees nothing of the global one, by any route.
- **Listings stay workspace-pure.** Folding shared entities into every listing would
  redefine what `GET /workspaces/{workspace}/models` returns for every existing caller.
  Surfacing them is worth doing behind an explicit opt-in, in its own change.
- Nothing in the permission model changed. No permission was added, no endpoint's
  requirements altered, no role redefined. `opa test` sits at main's 232/248 baseline.

An earlier revision of this branch widened *where a permission could be satisfied from*,
via an OPA rule and a per-permission allowlist. That was removed: it was not needed to
avoid duplicating models, and it meant a user entitled only to their own workspace could
still read the global one.

## Environment

| Item | Value |
| --- | --- |
| Services | `entities`, `models`, `inference-gateway`, `secrets` (+ `auth` for run 2, `models` controller throughout) |
| Backends | SQLite (local default) and PostgreSQL 16 (what Helm ships) |
| Isolation | Throwaway `NMP_DATA_DIR` per run; the developer's real database was never written to |

## Run 1 — resolution, listings, guard, gateway (SQLite)

21 scenarios, 21 passed.

| # | Scenario | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| R1 | Shared model resolves by name from another workspace | `200` | `200` | PASS |
| R2 | Resolved entity reports the global workspace as owner | `default` | `default` | PASS |
| R3 | Non-shareable type does not resolve globally | `404` | `404` | PASS |
| R4 | Unknown name still 404s | `404` | `404` | PASS |
| R5 | Local entity shadows the global one | `team-1790060918` | `team-1790060918` | PASS |
| R6 | Shadowing leaves the global entity intact | `default` | `default` | PASS |
| R7 | Models service resolves a shared model via the caller's workspace | `200` | `200` | PASS |
| L1 | Listing a workspace does NOT fold in global entities | `no` | `no` | PASS |
| L2 | Listing returns only the requested workspace | `team-1790060918` | `team-1790060918` | PASS |
| L3 | Listing the global workspace still returns its own rows | `yes` | `yes` | PASS |
| L4 | Models service listing is also workspace-pure | `no` | `no` | PASS |
| C1 | Adapter on a shared base model can be created cross-workspace | `200` | `200` | PASS |
| C2 | Deleting the shared parent is refused | `409` | `409` | PASS |
| C3 | The 409 names the dependent workspace | `yes` | `yes` | PASS |
| C4 | Refused delete leaves the parent intact | `200` | `200` | PASS |
| C5 | Delete stays refused while the foreign child exists | `409` | `409` | PASS |
| C6 | Once the child is removed, the parent deletes | `200` | `200` | PASS |
| G1 | Gateway catalogue lists only the caller's workspace | `no` | `no` | PASS |
| G2 | Gateway catalogue still lists the workspace's own models | `yes` | `yes` | PASS |
| G3 | A shared VirtualModel is still routable by name (no duplicate deploy) | `200` | `200` | PASS |
| G4 | An unknown model is still 404 | `404` | `404` | PASS |

Legend: `R` resolution · `L` listings stay pure · `C` cascade guard · `G` gateway.

`R1`/`R7` are the feature: a model deployed once in the global workspace is reachable
by name from another workspace, through both the entity store and the models service.
`G3` is the same thing for inference — a shared VirtualModel is routable, so no second
deployment is needed. `L1`–`L4` and `G1` confirm listings were not redefined.

## Run 2 — authorization (auth enabled)

Auth enabled with unsigned JWTs so distinct principals could be impersonated. Role
bindings were seeded with auth off, then the platform restarted against the same data
directory.

| Principal | Bindings |
| --- | --- |
| `admin@test.local` | `PlatformAdmin` in `system` |
| `carol@test.local` | `Editor` in `team-a` **and** `default` — entitled to the global workspace |
| `alice@test.local` | `Editor` in `team-a` only — **not** entitled to the global workspace |
| `bob@test.local` | none |

19 scenarios, 19 passed.

| # | Principal | Scenario | Expected | Actual | Result |
| --- | --- | --- | --- | --- | --- |
| P1 | `carol` | GET a model in her own workspace | `200` | `200` | PASS |
| P2 | `carol` | Resolve the SHARED model via her own workspace (the feature) | `200` | `200` | PASS |
| P3 | `carol` | GET the global workspace directly (she is entitled) | `200` | `200` | PASS |
| N1 | `alice` | GET the global workspace directly | `403` | `403` | PASS |
| N2 | `alice` | Resolve the shared model via her OWN workspace | `404` | `404` | PASS |
| N3 | `alice` | Her own workspace still works normally | `200` | `200` | PASS |
| N4 | `alice` | LIST the global workspace | `403` | `403` | PASS |
| N5 | `bob` | GET the global workspace with no bindings at all | `403` | `403` | PASS |
| N6 | `bob` | GET any workspace with no bindings at all | `403` | `403` | PASS |
| N7 | `anon` | GET with no token | `401` | `401` | PASS |
| W1 | `carol` | POST into the global workspace (Editor there) | `201` | `201` | PASS |
| W2 | `alice` | POST into the global workspace | `403` | `403` | PASS |
| W3 | `alice` | DELETE a global model | `403` | `403` | PASS |
| W4 | `alice` | POST into an unrelated workspace | `403` | `403` | PASS |
| X1 | `admin` | Read the global workspace | `200` | `200` | PASS |
| X2 | `admin` | Write the global workspace | `201` | `201` | PASS |
| X3 | `admin` | Read any workspace | `200` | `200` | PASS |
| D1 | `admin` | Delete a global parent with a foreign child is refused | `409` | `409` | PASS |
| D2 | `admin` | Still refused while the child exists (no override) | `409` | `409` | PASS |

`P2` and `N2` are the pair that define the design. `carol`, entitled to the global
workspace, resolves the shared model through her own workspace and uses it without a
duplicate deployment. `alice`, entitled only to `team-a`, gets `404` for the same
request and `403` addressing the global workspace directly — sharing resolves names for
callers who already have access, it does not grant access.

## Run 3 — PostgreSQL

There is no PostgreSQL job anywhere in CI: no workflow references it, and the entity
store's integration conftest hardcodes SQLite. Production ships PostgreSQL, so this
backend is checked by hand.

PostgreSQL 16 in Docker with migrations applied on startup. The full run 1 suite:
**21 of 21, identical to SQLite.**

## Run 4 — concurrency

`entities.parent` is `ON DELETE CASCADE`, so deleting a parent removes its children.
The guard refuses when children live in another workspace. An earlier revision checked
for those children in a separate transaction from the delete, and a child created in
that window was destroyed unseen in 24 of 25 trials. The check now runs inside the
delete transaction with the row locked.

| Backend | Scenario | Result |
| --- | --- | --- |
| PostgreSQL | 40 concurrent child-create vs delete | **0 silently destroyed** — all 40 serialized, delete won, foreign key rejected the child |
| SQLite | 40 concurrent child-create vs delete | **0 silently destroyed** |

`FOR UPDATE` is a no-op on SQLite; the repository's existing write lock and the
single-transaction delete produce the same outcome there.

## Totals

| Run | Focus | Scenarios | Result |
| --- | --- | --- | --- |
| 1 | Resolution, listings, guard, gateway (SQLite) | 21 | all pass |
| 2 | Authorization, four principals | 19 | all pass |
| 3 | Same suite on PostgreSQL | 21 | all pass |
| 4 | Concurrency, both backends | 80 trials | 0 silent destructions |

## Harness notes

Three separate runs produced misleading results before being corrected. All three were
setup failures hidden by `curl -o /dev/null`, and all three would have reported passes
for negative assertions that succeed trivially when the fixtures do not exist:

1. Gateway checks pointed at `/apis/inference/v2/...`; the real prefix is
   `/apis/inference-gateway/v2/...`.
2. A fixed test workspace name was reused across runs. Workspace deletion is staged, so
   re-creating it silently failed and every assertion in that workspace 404'd.
3. Test VirtualModels were created with `autoprovisioned: true`, and the provider
   reconciler deletes autoprovisioned VirtualModels that no provider backs
   (`Deleted orphaned autoprovisioned VirtualModel` in the models controller log).
   They vanished within seconds of creation.

The suites now assert their own setup and fail loudly rather than reporting vacuous
passes.

## Known limitations

- **No automated PostgreSQL coverage.** Run 3 checked it by hand. Nothing in CI will
  catch a future PostgreSQL-only regression in this path.
- **Shared entities are not discoverable.** By design for this change: they resolve by
  name but do not appear in listings, so a caller has to know the name. Surfacing them
  behind an opt-in query parameter is the natural follow-up.
- **A shared entity with children elsewhere cannot be deleted until they are removed.**
  There is no override. Refusing is the whole point, and an escape hatch that cascades
  into other workspaces is not something this change needs to ship with; if operators find
  the cleanup painful, an explicit override can be added deliberately later.
- **The global workspace is world-writable in a stock install.** Platform seeding grants
  the wildcard principal `Editor` on `default`, so every authenticated user can write
  there. That predates this work and is unchanged by it, but it is worth knowing when
  `default` becomes the workspace others depend on. Acceptance criterion 3 asks for the
  opposite and cannot be met while that binding is seeded.
