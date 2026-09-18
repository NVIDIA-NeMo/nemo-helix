<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-scaled-evals (Phase 1 ephemeral plugin)

Vendors the scaled-evals control plane into NeMo Platform as an ephemeral plugin so Harbor/Gym scaled evaluation keeps working end-to-end while substrate plugins (builder/registry/sandbox) and the nemo-evaluator API merge land later.

Portable behavior is reconciled through standalone scaled-evals `1.13.0`
(`c64f23e71dc829414ab9279483973a84a17eea8d`). This remains a platform-adapted
fork: hosted deployment, identity-provider, and private corpus surfaces are not
vendored. Benchmark-wide Harbor archive exports are additionally ported from
standalone `1.22.0` (`cffd0af2d7aa3e9cc323cc4c32204faf785688fe`); this does not
imply reconciliation of the other changes after `1.13.0`. Selected portable
operational fixes are ported through standalone `1.23.1`: threadpool-independent
liveness (`a5ab1eb3`), short-lived database checkouts for streaming downloads
(`03efad39`), and JSON-aware artifact redaction (`880ba9d5`).

## Install (ephemeral — not in `enabled-plugins` yet)

```bash
# From nemo-platform repo root
uv sync
uv pip install -e plugins/_temporary-scaled-evals/
```

Restart `nemo services run` after install.

## Required configuration

| Variable | Notes |
|---|---|
| `SCALED_EVALS_DATABASE_URL` | scaled-evals' own Postgres. **Not** `DATABASE_URL` — the platform injects that into the same process, and reusing it would run these migrations and claim queues inside the platform database. |
| `SCALED_EVALS_DATABASE_SCHEMA` | Default `scaled_evals`. Every table lives here. Redundant in a dedicated database, but it keeps the vendored SQL safe to point at a shared one. |
| `CREDENTIALS_ENCRYPTION_KEY` | Fernet key for BYOK credentials. |
| `SCALED_EVALS_RUN_MIGRATIONS` | Default `true`. Set `false` where an external Job owns schema rollout. |
| `HARBOR_DATASET_UPSTREAM_ALLOWED_REGISTRIES` | Comma-separated registries allowed as public source images for dataset-only Harbor imports. Defaults to `docker.io,ghcr.io`; this is separate from the managed destination-image allowlist. |

Settings resolve lazily, so the plugin still loads when these are unset; the failure
surfaces on the first request that needs them rather than making the plugin vanish
from service discovery.

## Database and migrations

**scaled-evals runs on its own Postgres.** It shares the platform's *process* but
not its database: the plugin reads `SCALED_EVALS_DATABASE_URL` and never consults the
platform's `DatabaseConfig` or `DATABASE_*` variables. Point it at an empty Postgres
and it does the rest.

The DSN resolves in this order:

1. `SCALED_EVALS_DATABASE_URL`, when set. Every deployment sets it.
2. A `localhost:5432` development default, matching the standalone repo, so a local
   `docker run postgres:16` is the only setup step.

**Postgres is required.** The claim queues need `FOR UPDATE SKIP LOCKED`, plus JSONB,
advisory locks, and enum types, so the platform's own SQLite default is not a usable
target — which is a second reason not to inherit it. When the database is unreachable
the plugin degrades rather than failing the boot: the platform starts normally, other
services are unaffected, and `/v1/readyz` returns `503 degraded`.

Tables live in a `scaled_evals` schema, reached through a `search_path` option that
`resolved_database_url()` appends to the DSN — the one choke point the API pool, both
workers, and the migration applier share. In a dedicated database this is redundant,
and `public` would work; it is kept because the vendored SQL is unqualified, so the
schema is the whole reason this code is *also* safe against a shared database. Startup
creates the schema and then **verifies** the connection resolves there, since Postgres
silently drops an unknown schema from `search_path` instead of erroring.

The plugin migrates on startup, so the database it points at needs no manual
`psql` step. `db/schema` loads only when the schema is fresh
(sentinel: `evaluations`), then every `db/migrations` file is applied in filename
order. The migration set has no version ledger and each file is written to be
re-appliable, so re-applying all of it on each boot is the upstream contract rather
than a shortcut.

**Vendored-SQL divergence:** migrations `001`, `005`, `008`, and `013` had 25 hardcoded
`public` qualifiers (`table_schema = 'public'`, `schemaname = 'public'`,
`to_regclass('public.x')`). These are now `current_schema()` / unqualified so they work
in any schema. Enum literals such as `visibility = 'public'` were left alone.

Concurrent replicas are serialized with a Postgres advisory lock. Migration failure
is logged and **not** raised: an ephemeral plugin should not take the platform API
down because its own database is unreachable, and `/v1/readyz` already reports
`schema` as a required check.

Same logic runs one-shot for a compose service, k8s Job, or manual rollout:

```bash
uv run scaled-evals-migrate            # same resolution order as the plugin
uv run scaled-evals-migrate --dsn ...  # or an explicit target
```

## Routes

| Path | Notes |
|---|---|
| `GET /apis/scaled-evals/healthz` | Plugin liveness stub — static, does not check Postgres |
| `/apis/scaled-evals/v1/{healthz,readyz,metrics}` | Vendored ops probes; `readyz` checks dependencies |
| `/apis/scaled-evals/v1/*` | scaled-evals `/v1` semantics (tasks, evaluations, …) |

Benchmark-run creation accepts `member_framework_profile_ids`, a task ID to
framework profile ID map whose entries override the run-level
`framework_profile_id` for matching members. The CLI exposes the map as
repeatable `--member-framework-profile TASK_ID=PROFILE_ID` options. Overrides
are rejected unless both references are valid and the task belongs to the
selected benchmark revision. Dataset-only checks cover both the run-level and
member-level profiles.

Dataset-only image imports perform unauthenticated registry reads and reject
registries outside `HARBOR_DATASET_UPSTREAM_ALLOWED_REGISTRIES` before network
I/O. Bearer-token challenges must remain on the registry host; Docker Hub may
also use `auth.docker.io`.

**Not mounted:** Switchyard lease/publish (`/v1/switchyard/*`). Switchyard modules remain in-tree for dispatch import compatibility but are out of Phase 1 product surface.

**Removed:** OAuth client discovery (`/v1/auth/config`), which existed only to
hand the CLI login flow the coordinates of a fixed internal identity provider.

There is no second way to serve these routes. The package used to also ship a
standalone FastAPI app (`scaled-evals-api`) that mounted the Switchyard router and
registered the external auth middleware, so installing the package handed you a
command that bypassed the curation above. That app is gone: this service class is
the only HTTP surface, which makes "not mounted" a property of the code rather
than a convention.

## Download a benchmark as one Harbor experiment

The plugin can combine a Harbor benchmark run's member evaluation archives into
one experiment directory. This is separate from the existing per-evaluation
archive and Harbor Viewer download routes.

Point the bundled CLI at the plugin mount (the CLI appends `/v1`):

```bash
export SCALED_EVALS_BASE_URL=http://localhost:8080/apis/scaled-evals
# Supply SCALED_EVALS_TOKEN when platform authentication requires it.
uv run scaled-evals benchmark-run archive <run-id> --build
uv run scaled-evals benchmark-run download <run-id> --wait -o ./harbor-results
# Alternatively, keep the gzip tarball without extracting it:
uv run scaled-evals benchmark-run download <run-id> --archive-only -o ./results.tar.gz
# Re-snapshot member executions after a retry or to refresh a ready export:
uv run scaled-evals benchmark-run archive <run-id> --force
```

The API exposes `GET` and `POST /v1/benchmark-runs/{run_id}/archive`, plus
`GET /v1/benchmark-runs/{run_id}/archive/download`, under `/apis/scaled-evals`.
POST accepts `{"force": false}` and returns 202; GET reports `missing`, `queued`,
`building`, `ready`, or `failed`. A non-Harbor run, an empty run, or members that
have not finished execution, evidence, teardown, and individual archive generation
produce a 409 rather than an incomplete snapshot. Repeated requests reuse queued,
building, or ready work; `force` replaces a ready/failed export, not an active build.

The separate **dispatch worker must be running**. It processes benchmark archives
after higher-priority lifecycle, evaluation, evidence, and per-evaluation archive
work. Leases survive worker restarts, and changed member executions/archives
prevent publication of the in-progress export. Upload failures and lease loss
trigger cleanup of unpublished claim-token objects. Cleanup rechecks the archive
row under a lock before deleting: a lost commit acknowledgement must not delete
a successfully published archive. If the database or object store is unavailable,
cleanup defers rather than guessing.

Idle dispatch workers also reconcile archive prefixes periodically, retaining the
published object and the current building claim's object. This recovers objects
left by process crashes, failed deletions, superseded generations, or an upload
that completed after a revoked worker exited. The per-run throttle is persisted
in Postgres (`BENCHMARK_ARCHIVE_CLEANUP_INTERVAL_SECONDS`, default 300 seconds);
sweeps repeat, rather than treating an initially empty prefix as permanently clean.
The dispatcher handles evaluation/archive work before these maintenance sweeps.

The API records the completed
tarball's SHA-256, and the CLI verifies its size and digest **before** extracting
or keeping it. Old exports without a recorded checksum still download with an
explicit warning; rebuild them with `benchmark-run archive <run-id> --force` to
get checksum-verified downloads. Without `--wait`, download queues
a missing archive and returns; `--wait --timeout 600` waits up to ten minutes.
A timeout leaves the server build running. Existing local destinations are refused.

The export contains combined `config.json`/`result.json`, distinct trial directories,
original member metadata under `_scaled_evals/evaluations/`, shared benchmark
artifacts under `_scaled_evals/benchmark/artifacts/`, and
`scaled-evals-benchmark-archive.json`, a validated Pydantic manifest with source
identities, the complete already-redacted per-member provenance (including agent
bundle, framework/adapter versions, runner images, source revisions and config
hashes), and checksums for every final exported file except the manifest itself.
Original producer metadata remains alongside the exported data; no live/private
execution snapshot or credential payload is added to the export.

Before rewriting Harbor metadata, the exporter cross-checks each downloaded
member's files against its producer `scaled-evals-manifest.json`. Size, checksum,
missing/extra file, duplicate-path and unsafe-path mismatches fail the export.
Legacy members without that manifest explicitly report
`artifact_integrity: manifest_missing`, rather than claiming verification.
Producer checksums detect content inconsistency; they are not signatures or proof
against replacement of both the data and its manifest. The API's outer tarball
checksum independently verifies the subsequent download. Inspect it with
`scaled-evals --json benchmark-run archive <run-id>` and compare a saved tarball
using `sha256sum` when downloading outside the CLI.

Run/evaluation IDs are validated as safe directory components, not special test
IDs. The exported Harbor job UUID is derived from the benchmark run and archive
generation: it is stable across worker retries, and changes on a forced rebuild.
Missing trial data or explicitly unavailable campaign evidence is marked `partial` and
reported by the CLI. Custom metrics and pass@k are **not** aggregated; recompute
them downstream. This is an analysis export, not an exact rerun configuration.

Source extraction is bounded across members and shared artifacts by
`BENCHMARK_ARCHIVE_MAX_FILES` (default 100,000) and
`BENCHMARK_ARCHIVE_MAX_SOURCE_BYTES` (default 10,000,000,000). These are not total
scratch-space or tenant quotas; workers need space for extracted inputs and the
compressed output. New migration `042_benchmark_run_archives.sql` is schema-scoped
and safe to replay at startup. No new platform database or Studio UI is introduced.

### Provider boundary

Archive assembly, input integrity, and the typed manifest are provider-independent.
The dispatcher injects evidence checks; Switchyard-specific campaign reference
validation lives in `dispatch/switchyard_archive.py`, not in the generic exporter.
Future workload implementations can supply their own evidence checks without
modifying archive assembly. Callers of the HTTP API cannot disable these checks.

The existing Switchyard provisioning/cleanup lifecycle still belongs to the
standalone-compatible dispatcher. When moving it into a separate workload service,
that boundary should own provisioning, readiness, cleanup, lease persistence and
public evidence publication together. The dispatcher would sequence the workload
contract, and the archive would consume its published evidence through the same
validator seam. This port does not introduce a speculative workload framework or
rewrite existing execution/credential lifecycle behavior.

## Who a caller is

The plugin does not authenticate anyone. It used to: the standalone service did
OIDC discovery against a fixed internal issuer, fetched JWKS and validated JWT
signatures, ran an OAuth client-credentials exchange, introspected API keys, and
trusted identity headers from a specific private proxy — roughly 700 lines in
`api/auth.py`, plus a CLI that ran authorization-code + PKCE login and cached the
session in the OS keyring.

All of it is gone. The platform authenticates the request before a plugin route
runs, so a second identity provider inside the plugin would be a competing source
of truth. What survives is the ownership model the repositories actually filter
on — `CurrentPrincipal.owner_id` — and one function that produces it.

Consequences worth knowing before you rely on this:

- **`/v1` is single-tenant.** Every caller resolves to owner `dev`, so all
  callers share one view of tasks, evaluations, and credentials. Bridging
  platform identity means setting `request.state.principal` upstream; no route
  changes.
- **`CONTROL_PLANE_AUTH_ENABLED=true` now fails closed with 401** rather than
  gating on a provider. There is nothing left to consult, and silently falling
  back to the shared owner would hand one caller another caller's data.
- **The CLI takes a token, it does not obtain one.** `--token` /
  `SCALED_EVALS_TOKEN` is the only credential source; `scaled-evals auth
  login|status|logout` are gone and `auth whoami` is now top-level
  `scaled-evals whoami`.

## Verify locally (end to end)

**1. Plugin loads.** The `nemo.services` loader swallows import errors, so a broken
plugin disappears with one log line instead of failing loudly — check discovery
explicitly rather than assuming.

```bash
uv run python -c "
from nmp.platform_runner.registry import get_available_services, get_service_groups
print('discovered:', 'scaled-evals' in get_available_services())
print('in groups:', [g for g, v in get_service_groups().items() if 'scaled-evals' in v])
"
uv run pytest plugins/_temporary-scaled-evals/tests/ -q
```

Expect `discovered: True`, `in groups: ['api', 'all']`, passing tests. To include the
real-Postgres migration check, point the suite at a throwaway server — it creates and
drops its own scratch database per run:

```bash
SCALED_EVALS_TEST_DATABASE_URL=postgresql://scaled_evals:scaled_evals@127.0.0.1:5434/scaled_evals \
  uv run pytest plugins/_temporary-scaled-evals/tests/ -q
```

**2. Postgres.** scaled-evals' own, so an empty server is all it needs.

```bash
docker run -d --name scaled-evals-pg \
  -e POSTGRES_USER=scaled_evals -e POSTGRES_PASSWORD=scaled_evals -e POSTGRES_DB=scaled_evals \
  -p 5434:5432 postgres:16
```

**3. Run the platform.** The platform keeps its own default store (SQLite locally);
only scaled-evals is pointed at Postgres. Generating the key via command substitution
keeps it out of shell history.

```bash
export SCALED_EVALS_DATABASE_URL="postgresql://scaled_evals:scaled_evals@127.0.0.1:5434/scaled_evals"
export CREDENTIALS_ENCRYPTION_KEY="$(uv run python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"

uv run nemo services run --services scaled-evals --port 8080
```

Banner should read `Services (1): scaled-evals`. If the port is held by a previous run,
`nemo services stop` first. Startup logs the migration result — `8 schema files` on a
fresh schema, `0 schema files` on every boot after that:

```text
scaled-evals: database ready in schema 'scaled_evals' (8 schema files, 41 migrations)
```

Confirm the tables landed, and that nothing leaked into `public`:

```bash
docker exec scaled-evals-pg psql -U scaled_evals -d scaled_evals -tAc \
  "select table_schema, count(*) from information_schema.tables \
   where table_schema in ('public','scaled_evals') group by 1 order by 1;"
# scaled_evals|19
```

**4. Fire the API.**

```bash
B=http://127.0.0.1:8080/apis/scaled-evals
curl -s $B/healthz
curl -s $B/v1/readyz
curl -s $B/v1/tasks
curl -s -X POST $B/v1/tasks -H 'content-type: application/json' \
  -d '{"name":"smoke-task","description":"phase1 plugin smoke"}'
```

`readyz` returns 503 `degraded` with `postgres: ok` and `schema: ok`. The failing
`object_store`, `buildkit`, `registry`, and `build_worker` checks are expected until
T1.3/T1.5 stand those up; the control plane itself is healthy. `POST /v1/tasks` returns
201 with a presigned upload URL, which needs an object store before the upload
completes.

**5. Teardown.** `nemo services run` spawns a child uvicorn that outlives its parent, so
kill the listener rather than the wrapper.

```bash
lsof -ti tcp:8080 | xargs kill
docker rm -f scaled-evals-pg
```

## Background execution

With `SCALED_EVALS_PLATFORM_BUILD_JOBS_ENABLED=true` and
`SCALED_EVALS_PLATFORM_EVALUATION_JOBS_ENABLED=true`, the
`scaled-evals-jobs` controller submits task builds and evaluation executions to
Platform Jobs. Postgres remains the admission queue and metadata source of
truth during this migration.

`scaled-evals-dispatch-worker` remains deployed for cleanup, provenance, and
SBOM queues. The legacy `scaled-evals-build-worker` command remains available
only for deployments that leave Platform Jobs disabled.

## Licensing

Every `.py`, `.sql`, `.sh`, `.yaml`, `.yml`, and `.toml` file carries the
two-line SPDX header the rest of the platform uses. `tests/test_spdx_headers.py`
enforces it over `git ls-files`, and also asserts a shebang still owns line 1 —
inserting a header above `#!` silently stops the kernel honouring it, and nothing
else in the suite would notice.

## Out of this plugin

- Upstream-only evaluation apps and corpora
- Hosted image-admission canary (stubbed no-op)
- Hosted auth sidecars (`auth-router` / `preview-router`)
- Platform image-builder / registry / sandbox plugins (Phase 2)
- nemo-evaluator API fold (Phase 3)
