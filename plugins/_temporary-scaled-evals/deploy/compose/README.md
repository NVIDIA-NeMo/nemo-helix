<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Compose stack (Phase 1)

The NeMo Helix API serving the scaled-evals plugin, both scaled-evals
workers, and the substrate they need — Postgres, RustFS, BuildKit, and a
registry. Enough to take a task from `create` to a built, pushed image.

```bash
cd plugins/_temporary-scaled-evals/deploy/compose
cp .env.example .env     # then set CREDENTIALS_ENCRYPTION_KEY and HARBOR_EXTRA_INDEX_URL (see Configuration)
docker compose up -d     # first run builds the app image (~1 min)
./smoke.sh               # create -> upload -> finalize -> ready, then verify the push
docker compose down -v
```

`smoke.sh` fails loudly on the first broken step, so use it as the stack's
health verdict rather than reading `docker compose ps`.

## What runs where

| Service | Image | Purpose |
|---|---|---|
| `api` | built here | `nemo services run --services scaled-evals` on `:8080` |
| `build-worker` | same image | claims finalize jobs, drives BuildKit, pushes |
| `dispatch-worker` | same image | claims evaluations (no runtime enabled in Phase 1) |
| `postgres` | `postgres:16` | platform database; scaled-evals lives in its `scaled_evals` schema |
| `rustfs` | `rustfs/rustfs` | S3-compatible object store on `:9000` |
| `buildkit` | `moby/buildkit` | builds task Dockerfiles; no published port |
| `registry` | `registry:2` | local stand-in for NGC on `:5000` |

Everything is published on `127.0.0.1` only.

## The app image is not the platform's image

`docker/Dockerfile.nhx-api` is the platform's real image. It builds through
bake, from base images in internal registries, and installs the
`functional-services` dependency group — which does not include this ephemeral
plugin. Building it locally is neither quick nor plugin-aware.

The `Dockerfile` here installs the workspace scoped to the plugin instead
(`uv sync --package`), which drops ~370 packages (guardrails, transformers, NAT)
that a scaled-evals-only API never imports. That is what makes a from-scratch
build about a minute instead of tens of minutes.

Three packages are installed that the plugin does not depend on:

- `nhx-platform-runner` serves the plugin. It is not in the plugin's dependency
  closure because the runner loads plugins, not the reverse.
- `nemo-helix-ext` implements the `nemo` CLI imported by the SDK wrapper.
  Release wheels vendor it, but a source-checkout image must install it.
- `nhx-entities` is needed only because the runner's startup banner imports
  `nhx.core.entities.config` unconditionally without declaring the dependency.
  A full `uv sync` hides that; a scoped install does not.

## Differences from the standalone scaled-evals compose stack

Four services are gone because the plugin now does the work itself:

| Dropped | Why |
|---|---|
| `schema-migrate` | the plugin migrates on startup |
| `rustfs-init` (`mc mb`) | the plugin creates its bucket on startup |
| Postgres TLS + `compose-certs` | Compose explicitly sets `DATABASE_SSL_MODE=disable` for its loopback-only plaintext Postgres |
| `harbor-runner`, `gym-runner` | evaluation runtimes are out of Phase 1 scope |

One deliberate substitution: **BuildKit runs privileged here, not rootless.**
Rootless buildkitd has to create an unprivileged user namespace, and hosts that
set `kernel.apparmor_restrict_unprivileged_userns=1` — Ubuntu 24.04, and the
Colima VM that runs it — refuse, so rootlesskit dies with
`fork/exec /proc/self/exe: permission denied`. The documented workaround
installs an AppArmor profile on the host, which a dev stack should not require.

The hosted-cluster, identity-provider, Switchyard, and Gym environment blocks are
not carried over; none are part of Phase 1.

## scaled-evals' own database

The Postgres here belongs to scaled-evals. All three processes read one explicit
`SCALED_EVALS_DATABASE_URL`; none of them consults the platform's `DATABASE_*`
config, so the platform keeps its own default store even though the API serves
the plugin in-process.

```bash
docker compose exec postgres psql -U scaled_evals -d scaled_evals -tAc \
  "select table_schema, count(*) from information_schema.tables
   where table_schema in ('public','scaled_evals') group by 1 order by 1;"
# scaled_evals|19
```

Tables sit in a `scaled_evals` schema rather than `public`. That is redundant in a
database of its own; it is kept because the vendored SQL is unqualified, so the
schema is what makes the same code safe against a shared database later.

**Upgrading an existing stack:** the database was previously named
`nemo_helix`, so a volume from before this change has no `scaled_evals`
database and the API will report that it does not exist. `docker compose down -v`
for a clean start, or `createdb` it by hand.

## Configuration

Two variables are required.

`CREDENTIALS_ENCRYPTION_KEY`, because a committed default would encrypt BYOK
credentials under a key published in this repository's history:

```bash
cp .env.example .env
python3 -c 'import base64,os; print("CREDENTIALS_ENCRYPTION_KEY=" + base64.urlsafe_b64encode(os.urandom(32)).decode())' >> .env
```

That is a valid Fernet key (32 random bytes, urlsafe base64) and needs only the
standard library, so it works before any project dependency is installed.

And `HARBOR_EXTRA_INDEX_URL`, for the image build only. The runner venv installs
`sandbox-k8s[harbor]`, which is not published on PyPI, so the build stage fails
without an index that serves it:

```bash
echo 'HARBOR_EXTRA_INDEX_URL=https://<index>/simple' >> .env
```

Everything else has a working default. Override by exporting it or adding it to
that same `.env`; the notable knobs:

| Variable | Default | Notes |
|---|---|---|
| `API_PORT` | `8080` | |
| `DATABASE_SSL_MODE` | `disable` | Compose-only default for the local plaintext Postgres service |
| `CREDENTIALS_ENCRYPTION_KEY` | *(required)* | see above; a real deployment sources this from a secret |
| `HARBOR_EXTRA_INDEX_URL` | *(required to build)* | index serving `sandbox-k8s`; unused once the image exists |
| `IMAGE_BUILD_PLATFORM` | *(empty)* | empty builds for the host arch. The setting's own default is `linux/amd64`, which sends every build through QEMU on an arm64 laptop; set it when the image must run on amd64 nodes |
| `IMAGE_REGISTRY` | `registry:5000` | point elsewhere with `REGISTRY_INSECURE=false` plus credentials |
| `S3_PUBLIC_ENDPOINT` | `http://localhost:9000` | baked into presigned URLs; SigV4 signs the Host header, so this must be the address the *client* calls |

## Known gaps

- `dispatch_worker`, `gym_dispatch`, and `sandbox_k8s_dispatch` report
  `skipped: disabled` in `readyz`. The dispatch worker process runs, but no
  evaluation runtime is enabled in Phase 1, so evaluations do not execute here.
- Both workers are silent: upstream `queue_worker.py` has no log statements, so
  a working build produces no output. Confirm progress through task status or
  the `service_heartbeats` table rather than `docker compose logs`.
