<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# nemo-scaled-evals (Phase 1 ephemeral plugin)

Vendors the scaled-evals control plane into NeMo Helix as an ephemeral plugin so Harbor/Gym scaled evaluation keeps working end-to-end while substrate plugins (builder/registry/sandbox) and the nemo-evals API merge land later.

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
# From nemo-helix repo root
uv sync
uv pip install -e plugins/_temporary-scaled-evals/
```

Restart `nemo services run` after install.

## Required configuration

| Variable | Notes |
|---|---|
| `HARBOR_DATASET_UPSTREAM_ALLOWED_REGISTRIES` | Comma-separated registries allowed as public source images for dataset-only Harbor imports. Defaults to `docker.io,ghcr.io`; this is separate from the managed destination-image allowlist. |

Settings resolve lazily, so the plugin still loads when these are unset; the failure
surfaces on the first request that needs them rather than making the plugin vanish
from service discovery.

## Storage

scaled-evals has no database of its own. Its state is platform entities (see
[Background execution](#background-execution)), credentials are platform Secrets, and
artifacts live in the object store. There are no migrations to run. Controller health
is reported by the platform controller manager, so `/v1/readyz` checks only the object
store, plus BuildKit and the registry when they are enabled.

## Routes

| Path | Notes |
|---|---|
| `GET /apis/scaled-evals/healthz` | Plugin liveness stub — static, checks no dependency |
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

Evaluation and benchmark-run `credentials` map a provider (`openai`,
`anthropic`, `nvidia`, `nhx`, `openshift`) to a platform Secret reference
`workspace/name`. Create the Secret with the platform Secrets API; preflight
and create reject a reference the caller cannot read. The value never touches
scaled-evals storage: the evaluation Job resolves it from the platform at
launch.

**Removed:** the `/v1/credentials` store and its encryption key. Credentials
are platform Secrets referenced by name.

**Removed:** OAuth client discovery (`/v1/auth/config`), which existed only to
hand the CLI login flow the coordinates of a fixed internal identity provider.

There is no second way to serve these routes. The package used to also ship a
standalone FastAPI app (`scaled-evals-api`) that mounted extra routers and
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

The scaled-evals Jobs controller submits
one `scaled-evals.benchmark-archive-build` Job per claim attempt, and the Job takes
the lease when it starts. Each evaluation Job builds its own evidence and archive;
the controller rebuilds any a Job left behind. Leases survive restarts, and changed member executions/archives
prevent publication of the in-progress export. Upload failures and lease loss
trigger cleanup of unpublished claim-token objects. Cleanup rereads the archive's
references before deleting: a lost write acknowledgement must not delete a
successfully published archive. If the entity store or object store is unavailable,
cleanup defers rather than guessing.

The controller also reconciles archive prefixes periodically, retaining the
published object and the current building claim's object. This recovers objects
left by process crashes, failed deletions, superseded generations, or an upload
that completed after a revoked worker exited. The per-run throttle is stored
on the run's archive (`BENCHMARK_ARCHIVE_CLEANUP_INTERVAL_SECONDS`, default 300 seconds),
and sweeps also reach deleted runs. They repeat, rather than treating an initially
empty prefix as permanently clean.
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
compressed output. No Studio UI is introduced.

### Provider boundary

Archive assembly, input integrity, and the typed manifest are provider-independent.
Workload implementations can supply their own evidence checks without modifying
archive assembly. Callers of the HTTP API cannot disable these checks.

Switchyard is not managed by this plugin. Evaluations and benchmark runs no longer
accept a Switchyard profile, and any queued evaluation that still references one
fails with `switchyard_unsupported` instead of running without its
model-traffic fence.

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

- **Ownership is the platform principal.** With platform auth enabled, the owner
  of every row is the authenticated principal, or the user a service acts on
  behalf of. With platform auth disabled every caller is owner `dev`.
- **Admin is a platform permission.** `scaled-evals.admin` gates the
  `/v1/admin/*` routes and image qualification and promotion. The platform
  grants it, and `HelixAdmin` in the `system` workspace holds it implicitly.
  Admins also see every owner's agent bundles and can edit any config profile.
  Non-admins get 403 from the platform. There is no plugin allowlist.
- **Admin analytics come from Intake.** Each terminal evaluation records one
  span (source `scaled-evals-executions`) in the platform jobs workspace.
  `/v1/admin/usage`, `/compute`, and `/failures` aggregate those spans over a
  window of at most 90 days. Queued and active counts still come from the
  evaluation store. A span that fails to record is logged and skipped, so
  analytics can undercount during an Intake outage.
- **The CLI takes a token, it does not obtain one.** `--token` /
  `SCALED_EVALS_TOKEN` is the only credential source; `scaled-evals auth
  login|status|logout` are gone and `auth whoami` is now top-level
  `scaled-evals whoami`.

Evaluation and benchmark-run APIs enforce owner/admin access in their Entity
Store reads and compare-and-swap writes. Private resources are visible only to
their effective owner or an authorized platform admin. Existing `public`,
`team` and `org` visibility permits authenticated reads; it does not authorize
another owner to cancel, retry, delete or build archives. Team/org membership
is not implemented by this sharing convention. Lists apply access checks before
search and pagination; missing, deleted and inaccessible resources return 404.
Streams retain request scope and recheck resource access on each poll. Internal
controllers and workers keep deliberately unrestricted storage access.

Deleting a benchmark run soft-deletes its parent entity; member evaluations and
stored artifacts remain. Cascading member deletion is a
separate follow-up, not part of the ownership-isolation port. The isolation unit
regressions use in-memory entities and do not require Postgres or other services.

## Verify locally (end to end)

**1. Plugin loads.** The `nemo.services` loader swallows import errors, so a broken
plugin disappears with one log line instead of failing loudly — check discovery
explicitly rather than assuming.

```bash
uv run python -c "
from nhx.platform_runner.registry import get_available_services, get_service_groups
print('discovered:', 'scaled-evals' in get_available_services())
print('in groups:', [g for g, v in get_service_groups().items() if 'scaled-evals' in v])
"
uv run pytest plugins/_temporary-scaled-evals/tests/ -q
```

Expect `discovered: True`, `in groups: ['api', 'all']`, passing tests.

**2. Run the platform.** scaled-evals stores everything in the platform's own default
store (SQLite locally), so there is nothing else to start.

```bash
uv run nemo services run --services scaled-evals,entities,secrets,jobs --port 8080
```

Banner should list `scaled-evals`. If the port is held by a previous run,
`nemo services stop` first.

**3. Fire the API.**

```bash
B=http://127.0.0.1:8080/apis/scaled-evals
curl -s $B/healthz
curl -s $B/v1/readyz
curl -s $B/v1/tasks
curl -s -X POST $B/v1/tasks -H 'content-type: application/json' \
  -d '{"name":"smoke-task","description":"phase1 plugin smoke"}'
```

`readyz` returns 503 `degraded` until an object store is configured; the control plane
itself is healthy. `POST /v1/tasks` returns
201 with a presigned upload URL, which needs an object store before the upload
completes.

**4. Teardown.** `nemo services run` spawns a child uvicorn that outlives its parent, so
kill the listener rather than the wrapper.

```bash
lsof -ti tcp:8080 | xargs kill
```

## Background execution

The `scaled-evals-jobs` controller submits task builds and evaluation executions to
Platform Jobs. Creating or retrying a single evaluation also submits its Job
from the API right after the evaluation is stored, if the admission caps allow it; the
controller submits anything that path misses, including benchmark-run members.

There is no separate worker process. The controller also runs execution cleanup,
the evidence and archive backstops, and benchmark archive cleanup.

Before creating an evaluation Job, the submitter writes that execution's
immutable launch inputs to a `scaled_evals_execution` entity, and the Job
launches from it. The Job reads the changing run state from the evaluation
entity and writes status and results there. Because the inputs include prompt content and
the private execution snapshot, they are stored in
`SCALED_EVALS_ENTITY_STORE_PRIVATE_WORKSPACE` (default `scaled-evals-internal`).
The service creates that workspace itself, so only the service has a role in
it, and it refuses to use a workspace of that name that someone else created.

Config profiles (`scaled_evals_config_profile`) and agent bundles
(`scaled_evals_agent_bundle`) are also stored as entities in that workspace,
so the plugin's owner and admin checks cannot be bypassed through the Entities
API. Deleting either one removes the entity. Editing a profile or bundle that
changed since it was read returns `409 profile_changed` or
`409 agent_bundle_changed`. A profile referenced by a non-terminal evaluation
can be renamed, but its config cannot change and it cannot be deleted
(`409 profile_in_use`).

Tasks are `scaled_evals_task` entities in the same workspace, with their
revisions and build state embedded. The build queue claims a revision by
rewriting its task entity, so build heartbeats also move the task's
`updated_at`. A write that loses a concurrent update retries, then returns
`409 task_changed`. Slug uniqueness is checked before the write, and of two
simultaneous creates with the same slug the later one is withdrawn with
`409 slug_conflict`; a simultaneous rename onto a taken slug can still land. A
task referenced by
a non-terminal evaluation cannot be deleted (`409 task_in_use`). Benchmark and
benchmark-run member lists skip tasks or revisions that no longer exist.

Benchmarks are `scaled_evals_benchmark` entities with their revisions and
member pins embedded; a write that loses a concurrent update returns
`409 benchmark_changed`, and slugs are kept unique as for tasks. A benchmark
run's preflight blocks a member whose task was deleted. Benchmark imports are
`scaled_evals_benchmark_import` entities whose id is derived from the owner,
manifest digest, and visibility, so a repeated create resumes the existing
import. A create racing an identical one returns
`409 benchmark_import_in_progress` until the first has finished populating it.
A create that fails partway deletes its import; task revisions it already
allocated remain as unused uploads.

Evaluations are `scaled_evals_evaluation` entities with their status events,
per-execution telemetry, resource usage, and cleanups embedded; benchmark runs
are `scaled_evals_benchmark_run` entities. Every queue claims work by rewriting
the entity under a compare-and-swap, and a write that keeps losing returns
`409 evaluation_changed` or `409 benchmark_run_changed`. Differences from the
Postgres version:

- Admission caps are counted without a lock, so claimants running at the same
  time can each admit one evaluation past a cap.
- List endpoints, admin counts, and queue scans read every live evaluation or
  run, and run status is derived from all members on each read.
- A benchmark run and its members are separate writes. A failed create deletes
  what it wrote; a crash in between can leave queued members of a missing run.
  Cancelling a run likewise cancels members one at a time.
- A superseded execution is detected with a check, not a lock, so it can
  publish once more if a retry lands in between.
- Cleanup ids are strings (`<evaluation>/<execution>`), not integers.
- A run's combined archive and its build lease are embedded in the run entity.
  Claims, heartbeats and publication are fenced by the claim token. Object
  cleanup reads the archive's references without a lock: objects are listed
  first, so a later claim's upload is never a candidate.
- `SCALED_EVALS_ENTITY_STORE_PROJECTION_ENABLED`, `_READS_ENABLED`,
  `_WORKSPACE`, and `_PROJECTION_BATCH_SIZE` are gone with the projection.

Each evaluation Job may run for every sequential trial wave
(`ceil(n_attempts / parallelism)`) times the profile's sandbox
`lifecycle_timeout` plus a grace, and never less than
`DISPATCH_RUN_POLL_INTERVAL_SECONDS` x `DISPATCH_RUN_MAX_POLLS`.

**Known limitation: hosted Gym.** `gym_sandbox_opensandbox` evaluations have no
working path on Kubernetes yet (AALGO-906). The evaluation Job always runs in
the plugin's app image with a 1 CPU / 1 Gi limit; it does not switch to the
snapshot-pinned Gym runner image, set `GYM_RUNNER_MODE=process`, or mount
`/dev/shm`. With the default `GYM_RUNNER_MODE=docker` the run fails, because
the Job pod has no Docker daemon. Gym evaluations are still accepted, so leave
`GYM_SANDBOX_OPENSANDBOX_ENABLED` off on hosted deployments until AALGO-906 lands.

## Harbor runners for `harbor_opensandbox`

The `harbor_opensandbox` runtime runs Harbor **0.13.2** and **0.20.0**. An
evaluation that omits `framework_version` gets 0.20.0, not the catalog default:
Harbor 0.13.2 accepts task files written for 0.20 but silently ignores the
features 0.20 added, such as `[[verifier.collect]]` hooks, so pick 0.13.2 only for
tasks that don't use them. Other selectable runners are rejected at submission
with `422 unsupported_framework_version`.

Harbor loads `NemoOpenSandboxEnvironment` through `environment.import_path`. It
is a plain Harbor `BaseEnvironment` written on the OpenSandbox SDK
(`nhx_sandbox.harbor_opensandbox_driver`), not a subclass of Harbor's own
OpenSandbox environment, so it uses only the environment API both releases share.
Harbor's task model sets one difference: a task's `allowlist` may name hostnames
and `*.` wildcards on 0.13.2, and also IP addresses and CIDRs on 0.20.0. The
operator allowlist accepts all of them on both.

To support another runner, pin `opensandbox==1.2.0` in its
`packages/nhx_sandbox/harbor-runners/<version>.txt` and list `opensandbox` under that release's
`packages` in `data/harbor_runner_qualifications.json` (the image then installs the
environment into that venv, and CI runs the environment tests against it), then
add the version to `HARBOR_OPENSANDBOX_HARBOR_VERSIONS`.

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
- nemo-evals API fold (Phase 3)
