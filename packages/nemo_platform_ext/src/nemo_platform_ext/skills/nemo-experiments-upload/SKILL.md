---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-experiments-upload
description: End-to-end guide for getting evaluation data into NeMo Platform Intake so it shows up in the Experiments leaderboard. Create an Experiment, create an Evaluation, then log traces and evaluator results via ATIF (Harbor), chat-completions, or OTLP and view the rollups in Studio. Use when a user wants to create named evaluation runs, publish evaluation results, or compare runs in NeMo Experiments.
triggers:
  - log evaluation traces to experiments
  - upload experiment results
  - ingest evaluation data
  - how do I log to experiments
  - send evaluation traces to nemo experiments
  - publish evaluation results
  - log to experiments
  - upload harbor / atif results
  - get my eval data into nemo
not-for:
  - nemo-intake (use for general instrumentation, telemetry ingestion, trace queries, or evaluator results outside an Experiments leaderboard)
  - nemo-evaluator (use to AUTHOR and RUN evaluations/metrics; this skill UPLOADS results)
  - nemo-status (use for a read-only platform health dashboard)
  - nemo-skill-selection (use for dispatch when intent is unclear)
preconditions:
  - nemo_setup_complete
  - workspace_exists
  - clickhouse_ready
compatibility: nemo-platform >= 0.1.0; needs a reachable local or remote intake service (with auth and entities) backed by ClickHouse for rollups/results; talks HTTP to /apis/intake/v2 (curl only, no Docker); Experiments viewing in Studio is behind the VITE_FF_EXPERIMENT feature flag.
maturity: beta
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read, Write
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Log evaluation data to NeMo Intake

Get evaluation runs into the platform end-to-end: **create an Experiment → create an Evaluation → log traces + scores to an ingest endpoint → see the rollups.** The API calls the parent (the leaderboard) an **Experiment** and each row an **Evaluation**; the whole feature is called **Experiments**.

Everything below uses `${NMP_BASE_URL}` (default `http://localhost:8080`) and a `${WORKSPACE}`
(default `default`). Point `NMP_BASE_URL` at the local platform or a remote HTTPS origin. Reject
non-loopback `http://` targets, and never send authentication across an HTTP redirect. All routes
are under `/apis/intake/v2/workspaces/${WORKSPACE}`.

## Pre-flight

Confirm the target platform is reachable before doing anything. If this fails, report the target as
unreachable and stop; route to `setup`/`nemo-status` only for a local platform.

```bash
set -euo pipefail
: "${NMP_BASE_URL:=http://localhost:8080}"
: "${WORKSPACE:=default}"
nmp_authority=${NMP_BASE_URL#*://}
nmp_authority=${nmp_authority%%/*}
case "${nmp_authority}" in
  *@*) echo "NMP_BASE_URL must not contain userinfo" >&2; exit 1 ;;
esac
case "${NMP_BASE_URL}" in
  https://*) ;;
  http://*)
    case "${nmp_authority}" in
      localhost|127.0.0.1) ;;
      localhost:*|127.0.0.1:*)
        nmp_port=${nmp_authority#*:}
        case "${nmp_port}" in
          ""|*[!0-9]*) echo "loopback NMP_BASE_URL has an invalid port" >&2; exit 1 ;;
        esac
        ;;
      *) echo "HTTP NMP_BASE_URL must use exactly localhost or 127.0.0.1" >&2; exit 1 ;;
    esac
    ;;
  *) echo "remote NMP_BASE_URL must use https://" >&2; exit 1 ;;
esac
if curl -sf "${NMP_BASE_URL}/health/ready" >/dev/null; then
  echo "platform ready"
else
  echo "NOT READY — target platform is unreachable or not ready" >&2
  exit 1
fi
```

## What you do

Run the steps in order. Steps 1–2 create the entities; step 3 logs the data; steps 4–5 verify.

### 1. Create an Experiment

An Experiment is the leaderboard container. You need its `id` for the next step.

```bash
set -euo pipefail
: "${NMP_BASE_URL:=http://localhost:8080}"
: "${WORKSPACE:=default}"
experiments="${NMP_BASE_URL}/apis/intake/v2/workspaces/${WORKSPACE}/experiments"
# Create the experiment. 201 = created, 409 = already exists; any other status is a real failure.
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "${experiments}" \
  -H 'Content-Type: application/json' \
  -d '{"name": "my-experiment", "description": "example run"}')
case "${code}" in
  201|409) ;;
  *) echo "experiment create failed: HTTP ${code}" >&2; exit 1 ;;
esac
# Fetch the id (works whether it was just created or already existed).
EXPERIMENT_ID=$(curl -sf "${experiments}/my-experiment" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
[ -n "${EXPERIMENT_ID}" ] || { echo "could not resolve experiment id" >&2; exit 1; }
echo "experiment id: ${EXPERIMENT_ID}"
```

The POST accepts only `201` (created) or `409` (already exists) — any other status stops the step
instead of masking it. The `id` is then read with a GET, so this works on both first run and re-run;
`set -euo pipefail` + the `[ -n ]` guard keep it from continuing with an empty `EXPERIMENT_ID`.

### 2. Create an Evaluation

Create one Evaluation for each agent/config run against a dataset (one leaderboard row).

```bash
curl -sf -X POST \
  "${NMP_BASE_URL}/apis/intake/v2/workspaces/${WORKSPACE}/evaluations" \
  -H 'Content-Type: application/json' \
  -d "{\"name\": \"my-eval-baseline\",
       \"experiment_ids\": [\"${EXPERIMENT_ID}\"],
       \"dataset_name\": \"my-dataset\",
       \"dataset_version\": \"v1\",
       \"metadata\": {\"model\": \"provider/model\", \"job_name\": \"baseline\"}}"
```

- `experiment_ids` is a list holding the Experiment's **`id`** (from step 1).
- `metadata` values must be **strings** (`dict[str, str]`).
- **You must create the Evaluation before you can log to it** — ingesting with an unknown `evaluation_name` returns `400 "…must be created before it can be logged."`

### 3. Log traces + evaluator results

Pick the ingest endpoint that matches your producer. **Read `../nemo-intake/references/ingest-formats.md` for the full schema and a copy-pasteable example for each.** How you attach evaluation identity depends on the endpoint:

- **ATIF and chat-completions** (JSON body) — add an `evaluation_context = {evaluation_name: "<evaluation-name>", test_case_name: "<test-case-name>"}` object to the payload.
- **OTLP** — there is no body field; set `nemo.evaluation.name` and `nemo.test_case.name` as
  **attributes on the root span**. Spans missing these still
  ingest but won't associate to an Evaluation.

| Producer | Endpoint | Read |
|---|---|---|
| **Harbor / agent trajectories** (most common) | `POST .../ingest/atif` | `references/harbor-quickstart.md` |
| A single captured model call | `POST .../ingest/chat-completions` | `../nemo-intake/references/ingest-formats.md` |
| OpenTelemetry spans | `POST .../ingest/otlp/v1/traces` | `../nemo-intake/references/ingest-formats.md` |

Evaluator **scores** arrive one of two ways (both covered in the references):
- **Automatically** with ATIF — put rewards under `extra.verifier_result.rewards` (one key per criterion).
- **Explicitly** — `POST .../evaluator-results` with `{span_id, session_id, name, data_type, value}` — use `string_value` instead of `value` for `CATEGORICAL`/`TEXT` results (see `../nemo-intake/references/ingest-formats.md`).

### 4. Verify the data landed

```bash
curl -sf "${NMP_BASE_URL}/apis/intake/v2/workspaces/${WORKSPACE}/evaluations/my-eval-baseline" \
  | python3 -m json.tool
```

Then list the ingested sessions:

```bash
curl -sf "${NMP_BASE_URL}/apis/intake/v2/workspaces/${WORKSPACE}/evaluations/my-eval-baseline/sessions" \
  | python3 -m json.tool
```

### 5. View in Studio

Open Studio → the **Experiments** area (behind the `VITE_FF_EXPERIMENT` flag) → your experiment → your evaluation. You'll see the leaderboard row with score/cost/latency rollups and can drill into individual sessions and traces.

## Reference files

Read these before hand-writing a payload:

- **`../nemo-intake/references/ingest-formats.md`** — the shared Intake request schemas, evaluation context, evaluator results, and examples for ATIF, chat-completions, and OTLP.
- **`references/harbor-quickstart.md`** — the Harbor path specifically: mapping a Harbor trial result → an ATIF payload, including verifier rewards → evaluator scores.
- **`references/troubleshooting.md`** — every common `400`/`422`/`503` from the ingest and CRUD endpoints, with the fix.

## Verification

You succeeded when `GET .../evaluations/my-eval-baseline` shows:
- `run_count` ≥ 1 (each ingested session counts as one run), and
- non-empty `evaluator_names` / `aggregate_scores` if you logged rewards, and/or `cost_usd` if your spans carried cost.

If `run_count` is 0 after ingesting, the traces didn't associate — almost always a wrong evaluation identity: `evaluation_context.evaluation_name` for ATIF/chat-completions, or the `nemo.evaluation.name` root-span attribute for OTLP (see Gotchas).

## If verification fails

| Symptom | Cause | Recovery |
|---|---|---|
| `400 "…must be created before it can be logged."` | Ingested before the Evaluation existed, or `evaluation_name` doesn't match | Create the Evaluation (step 2); ensure `evaluation_context.evaluation_name` identifies it |
| `422 Unprocessable` on ingest | Unknown/typo'd top-level key (ATIF/chat-completions are `extra="forbid"`) or bad `schema_version` | Check the exact schema in `../nemo-intake/references/ingest-formats.md`; remove stray keys |
| Ingest 2xx but `run_count` stays 0 | Evaluation context is missing or doesn't match the target Evaluation | Attach the correct `evaluation_context.evaluation_name` (ATIF/chat-completions) or `nemo.evaluation.name` root-span attribute (OTLP) |
| `503` on GET evaluation / sessions | ClickHouse (telemetry store) not running | Start ClickHouse; rollups and sessions require it |
| Scores don't show up | Rewards not under `extra.verifier_result.rewards`, or wrong `data_type` on `/evaluator-results` | See `references/troubleshooting.md` |

## Gotchas

- **Create before you log.** The Evaluation entity must exist before any ingest referencing it — otherwise `400`.
- **The parent lives at `/experiments`; `/experiment-groups` is a deprecated hidden alias.** Prefer `/experiments`. Evaluations are created and logged under `/evaluations`.
- **`metadata` is `dict[str, str]`** — stringify non-string values or you'll get a `422`.
- **ATIF and chat-completions are `extra="forbid"`** (unknown keys → 422); `evaluation_context` itself is lenient (`extra="ignore"`).
