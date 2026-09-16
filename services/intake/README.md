<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Intake Service

Intake is the telemetry ingestion and read API for NeMo Helix. It stores span
and trace data in ClickHouse, accepts OpenTelemetry traces, and supports
post-hoc annotations and evaluator result lookup.

## API Surface

Active v2 workspace endpoints:

- `GET /apis/intake/v2/workspaces/{workspace}/spans`
- `GET /apis/intake/v2/workspaces/{workspace}/spans/groups`
- `GET /apis/intake/v2/workspaces/{workspace}/spans/{span_id}`
- `GET /apis/intake/v2/workspaces/{workspace}/sessions/{id}`
- `GET /apis/intake/v2/workspaces/{workspace}/traces`
- `GET /apis/intake/v2/workspaces/{workspace}/traces/{id}`
- `GET /apis/intake/v2/workspaces/{workspace}/annotations`
- `POST /apis/intake/v2/workspaces/{workspace}/annotations`
- `DELETE /apis/intake/v2/workspaces/{workspace}/annotations/{annotation_id}`
- `GET`/`POST /apis/intake/v2/workspaces/{workspace}/evaluator-results`
- `POST`/`GET /apis/intake/v2/workspaces/{workspace}/experiment-groups` (+ `GET`/`PUT`/`DELETE .../{name}`)
- `POST`/`GET /apis/intake/v2/workspaces/{workspace}/evaluations` (+ `GET`/`PUT`/`DELETE .../{name}`, `.../{name}/pin`, `.../{name}/sessions`)
- `POST /apis/intake/v2/workspaces/{workspace}/ingest/otlp/v1/traces`
- `POST /apis/intake/v2/workspaces/{workspace}/ingest/chat-completions`
- `POST /apis/intake/v2/workspaces/{workspace}/ingest/atif`
- `POST /apis/intake/v2/workspaces/{workspace}/ingest/spans`

## Logging experiment data (end-to-end)

The Experiments feature captures evaluation runs as leaderboard rows. The flow:

1. Create an **Experiment Group** — `POST /experiment-groups`.
2. Create an **Evaluation** under it — `POST /evaluations`. Use the Evaluation's `name` when
   associating telemetry with it.
3. Log traces + evaluator results to an ingest endpoint. The Evaluation must exist first. Attach
   evaluation identity per endpoint: **ATIF/Harbor** and **chat-completions** carry it in the JSON body
   as `evaluation_context = {evaluation_name, test_case_name}`; **OTLP** carries it as root-span attributes
   `nemo.evaluation.name` (the Evaluation name) and `nemo.test_case.name`.
4. Read the rollups — `GET /evaluations/{name}` and `.../{name}/sessions` — or view them in Studio
   (behind the `VITE_FF_EXPERIMENT` flag).

For the step-by-step guide with copy-pasteable payloads for all three ingest endpoints, the Harbor
mapping, and a troubleshooting table, see the **`nemo-experiments-upload`** agent skill:
`packages/nemo_helix_ext/src/nemo_helix_ext/skills/nemo-experiments-upload/`.

> Naming note: the entity/API is mid-rename. The API surfaces "Evaluation" and "Experiment Group",
> but the stored entity is still `Experiment`. The OTLP evaluation attribute key is
> `nemo.evaluation.name`. This doc follows the current in-code naming.

## Importing historical observability data

`POST /apis/intake/v2/workspaces/{workspace}/ingest/spans` accepts a provider-neutral JSON batch of
1–1000 spans. It supports arbitrary provider IDs, parent references outside the batch, structured
input/output, semantic attributes, and lossless native JSON for unmodeled attributes. Reposting the
same `(workspace, source, trace_id, span_id)` updates the logical span.

Customers should use the `nemo-intake` skill's bundled MLflow, LangSmith, Phoenix, and Braintrust
scripts. They page bounded provider exports, write spans before scores/annotations, retain native
provider records under namespaced raw attributes, deduplicate annotations, and query imported IDs
back before reporting success. Intake rejects spans outside its 90-day observation-time retention
window with `422` before writing any part of the batch. Increase the ClickHouse `spans` and
`trace_index` TTLs before importing older data.

## Local Development

Run these commands from the repository root unless a command says otherwise.
Intake tests rely on shared platform test helpers, so use the root `uv`
environment instead of package-scoped `uv run --package ...` commands.

Prerequisite for the managed local database: Docker must be installed and the
Docker daemon must be running.

Intake is tested and profiled on ClickHouse 26.3 LTS. Other ClickHouse versions
may not be supported.

Start Intake with the platform runner. When `NHX_INTAKE_CLICKHOUSE_URL` is not
set and the configured URL is the default `http://localhost:8123`, Intake
automatically provisions a ClickHouse container before initializing its client:

```bash
uv run nemo services run \
  --services auth,entities,intake \
  --host 127.0.0.1 \
  --port 8000
```

The container is owned by the resolved NeMo data directory, publishes ClickHouse
on a Docker-assigned loopback port, and is reused by platform processes sharing
that data directory. Its data is stored below `$NHX_DATA_DIR/intake-clickhouse/`
(or the normal NeMo data directory when `NHX_DATA_DIR` is unset). Graceful
platform shutdown stops the container without removing it or its data; later
startups restart and reuse it. A hard process termination can leave it running.
The managed lifecycle assumes only one active local platform runner uses a given
data directory. The data directory contains a durable identity marker; Intake replaces rather than
reattaches a stale container after that directory has been deleted and
recreated. Replacing a container does not delete the current directory. Only an
explicitly confirmed full data teardown deletes the default storage under the
NeMo data directory, after removing the managed container. An explicitly
configured `NHX_INTAKE_CLICKHOUSE_DATA_DIR` outside the NeMo data directory is
never deleted by platform teardown.

The managed image and storage location can be overridden through the typed
Intake configuration surface:

```bash
export NHX_INTAKE_CLICKHOUSE_IMAGE=clickhouse/clickhouse-server:26.3
export NHX_INTAKE_CLICKHOUSE_DATA_DIR=/path/to/clickhouse-data
```

Intake does not change permissions on an explicitly configured data directory.
Changing `NHX_INTAKE_CLICKHOUSE_USER` or
`NHX_INTAKE_CLICKHOUSE_PASSWORD` requires removing the existing managed
container so it can be provisioned with the new credentials; removing the
container does not remove its bind-mounted data. If the image is not installed
locally, startup logs announce the synchronous image pull before it begins.

If Docker is unavailable, Intake still starts, but ClickHouse-backed endpoints
return `503`. Start Docker Desktop on macOS/Windows or the Docker service on
Linux, then rerun `nemo setup` or restart `nemo services run`.

To use an operator-managed ClickHouse and bypass Docker provisioning entirely,
set its URL explicitly before starting the platform:

```bash
export NHX_INTAKE_CLICKHOUSE_URL=https://clickhouse.example.com:8443
```

`services/intake/scripts/spans/run_clickhouse.sh` remains available for callers
that need the historical `nhx-intake-clickhouse` name, fixed localhost ports
`8123`/`9000`, and repository-local data under `tmp/intake-clickhouse`. Point
Intake at that compatibility container explicitly:

```bash
services/intake/scripts/spans/run_clickhouse.sh
NHX_INTAKE_CLICKHOUSE_URL=http://localhost:8123 uv run nemo services run
```

Remove a compatibility container with its wrapper so the same data directory
is targeted, or pass the directory explicitly to the underlying command:

```bash
services/intake/scripts/spans/run_clickhouse.sh --remove
uv run python -m nhx.intake.local_clickhouse --remove --data-dir /path/to/clickhouse-data
```

Running `python -m nhx.intake.local_clickhouse` without compatibility flags uses
the normal managed container naming and dynamic loopback port behavior.

In another terminal, start Studio from the `web/` workspace with the Intake
feature flag enabled:

```bash
VITEST=true VITE_FF_INTAKE_ENABLED=true VITE_PLATFORM_BASE_URL=http://localhost:8000 \
  pnpm --filter nemo-studio-ui start -- --host 127.0.0.1
```

The Studio dev server is available at
`http://127.0.0.1:5173/workspaces/default/intake/traces`.

Configure your local OTLP/HTTP trace exporter, NeMo relay, or collector to send
spans to Intake:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:8000/apis/intake/v2/workspaces/default/ingest/otlp/v1/traces
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

Send a minimal OTLP trace after the service is running:

```bash
uv run services/intake/examples/send_otel_sample.py
```

Read it back:

```bash
curl -i "http://127.0.0.1:8000/apis/intake/v2/workspaces/default/spans?filter[session_id]=sample-session"
```

Seed an Experiment rollup and read it back:

```bash
uv run services/intake/scripts/spans/seed_experiment_rollup_data.py
curl -s "http://127.0.0.1:8000/apis/intake/v2/workspaces/default/evaluations/rollup-smoke-exp" | jq

# Optional larger local workload.
uv run services/intake/scripts/spans/seed_experiment_rollup_data.py \
  --experiment experiment-name \
  --base-url http://127.0.0.1:8000 \
  --runs 10 \
  --cases-per-run 10
```

## Testing

Focused route-surface test:

```bash
uv run --frozen pytest services/intake/tests/integration/test_intake.py -q
```

Focused ingest/read tests:

```bash
uv run --frozen pytest \
  services/intake/tests/integration/spans/test_chat_completions_ingest.py \
  services/intake/tests/test_atif_v17.py \
  -q
```

Run the full Intake service test suite:

```bash
make test-service SERVICE=intake
```

## Generated API Artifacts

Run `make refresh-openapi` after Intake route or schema changes. The Stainless
resource config lives in `sdk/stainless.yaml`.
