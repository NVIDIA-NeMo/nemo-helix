---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-intake
description: Instrument agents, ingest telemetry into NeMo Intake, and query spans, traces, sessions, and evaluator results. Use when connecting agent code or existing telemetry to Intake, choosing among OTLP, chat-completions, or ATIF, checking Intake and ClickHouse readiness, inspecting agent runs, or attaching evaluation scores outside the Experiments leaderboard workflow.
license: Apache-2.0
preconditions:
  - nemo_setup_complete
  - workspace_exists
allowed-tools: Bash, Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# Use NeMo Intake

Use Intake to normalize agent telemetry into queryable spans and traces. The Intake endpoint may be
local or remote. An agent does not need to run on NeMo Platform; it only needs network access and
credentials, when required, to send a supported format to that endpoint.

## Requirements

Set the target to the local or remote NeMo Platform origin:

```bash
export NMP_BASE_URL=http://127.0.0.1:8080
export WORKSPACE=default

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
```

Require:

- Reachable ClickHouse storage.
- A reachable local or remote NeMo Platform `intake` service with its `auth` and `entities`
  dependencies. ClickHouse must be reachable from Intake; it does not need to be reachable from the
  telemetry producer.
- One supported telemetry source. NeMo Studio is optional.

For a remote deployment, set `NMP_BASE_URL` to its HTTPS origin and skip local startup. Use the
deployment's authentication mechanism. Do not send authentication across an HTTP redirect: validate
the final HTTPS origin and do not add `curl -L` to authenticated requests. For a local source
checkout, follow `SETUP.md`, make sure Docker is running, then start the backend. Intake
automatically provisions local ClickHouse unless `NMP_INTAKE_CLICKHOUSE_URL` is set:

```bash
uv run nemo services run --services auth,entities,intake --host 127.0.0.1 --port 8080
```

Verify the Intake read path before ingesting:

```bash
curl -i "$NMP_BASE_URL/apis/intake/v2/workspaces/$WORKSPACE/spans?page=1&page_size=1"
```

Continue only on `200`; an empty list is healthy. `503` means Intake cannot reach ClickHouse. For
other failures, report the response and route platform startup problems to `setup` or `nemo-status`.

## Choose an ingest path

| Path | Use it for | Endpoint |
|---|---|---|
| **OTLP/HTTP protobuf** | Live, granular telemetry when instrumentation emits **OpenInference** or **OTel GenAI semantic conventions**. Prefer this for ongoing observability and complete agent hierarchies. | `POST .../ingest/otlp/v1/traces` |
| **Chat completions** | Captured OpenAI-compatible request/response logs, proxy instrumentation, or runtimes without OpenInference or OTel GenAI instrumentation. It represents one model interaction, not a full agent trajectory. | `POST .../ingest/chat-completions` |
| **ATIF** | Complete agent trajectories with ordered steps and metadata, especially Harbor evaluation trials. | `POST .../ingest/atif` |
| **Direct spans** | Historical exports from MLflow, LangSmith, Phoenix, Braintrust, or another trace store. | `POST .../ingest/spans` |

All endpoint prefixes are:
`$NMP_BASE_URL/apis/intake/v2/workspaces/$WORKSPACE`.

Read `references/ingest-formats.md` before hand-writing an ATIF or chat-completions payload, or when
you need the semantic attributes and response behavior for OTLP.

Do not translate a native source format without need. Preserve the source's IDs, timestamps,
hierarchy, inputs, outputs, statuses, errors, and semantic attributes.

## Import an existing trace store

When the user names MLflow, LangSmith, Arize Phoenix, or Braintrust, read only that provider's
reference below and run its bundled script. Do not load the other provider references.

| Provider | Reference | Script |
|---|---|---|
| MLflow | `references/import-mlflow.md` | `scripts/import_mlflow.py` |
| LangSmith | `references/import-langsmith.md` | `scripts/import_langsmith.py` |
| Arize Phoenix | `references/import-phoenix.md` | `scripts/import_phoenix.py` |
| Braintrust | `references/import-braintrust.md` | `scripts/import_braintrust.py` |

Every live import requires an explicit `--project`, `--since`, and `--until`. The scripts write
spans first, then provider evaluations and human annotations, verify the imported span IDs, and are
safe to replay. They use the existing SDK client factory, including the active CLI context and OAuth
token refresh; explicit `--nmp-base-url`, `--workspace`, and `NMP_ACCESS_TOKEN` overrides still
work. Use `--dry-run` to inspect the direct JSON projection without writing Intake.

Direct imports preserve provider timestamps and reject a complete span batch with `422` when any
`started_at` falls outside Intake's 90-day ClickHouse TTL. Increase the `spans` and `trace_index`
table TTLs before importing older history.

## Instrument code for OTLP

Only choose OTLP when the instrumentation emits **OpenInference** or **OTel GenAI semantic
conventions**. Generic OpenTelemetry spans may ingest, but they do not provide the semantic model,
tool, token, cost, input/output, and session fields needed for useful Intake telemetry.

1. Recommend **NeMo Relay** when it supports the agent runtime and emits a supported semantic
   format; it avoids hand-written span capture and records agent, model, and tool activity
   consistently. Use the format Relay actually exports: send semantic OTLP to the OTLP endpoint, or
   Relay ATIF output to the ATIF endpoint.
2. Otherwise use the runtime's **OpenInference instrumentor**, or its native **OTel GenAI**
   instrumentation, with an OpenTelemetry SDK OTLP exporter. Choose the framework-specific
   instrumentor from its current documentation; do not invent package names or APIs.
3. If neither semantic convention is available, capture each OpenAI-compatible request and response
   and send it directly to `.../ingest/chat-completions`. Do not recommend generic OTLP as a
   substitute.
4. For supported OTLP instrumentation, point the exporter at Intake:

   ```bash
   export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="$NMP_BASE_URL/apis/intake/v2/workspaces/$WORKSPACE/ingest/otlp/v1/traces"
   export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
   ```

5. Emit a root agent/chain span plus granular model, tool, retrieval, guardrail, and error spans.
   Preserve parent-child IDs and set a stable `session.id` for related traces. Always set
   `gen_ai.agent.name` (or `llm.agent.name` / `agent.name` for instrumentation that emits those
   conventions) on every span to a stable name for the agent. This ensures the spans can be analyzed
   by NeMo Insights in the future.
6. Run one representative interaction, then verify it through the spans query below.

Intake maps OpenInference and OTel GenAI semantic attributes into queryable model, provider, tool,
status, token, cost, and error fields while retaining unhandled attributes.

## Customer-facing data model

The hierarchy is `session -> trace -> span`; evaluator results attach to a span and session.

| Record | Meaning |
|---|---|
| **Span** | One timed operation, such as an agent step, model call, tool call, retrieval, guardrail, evaluator, or chain step. It carries IDs, timing, status/error, input/output, semantic fields, and source attributes. |
| **Trace** | One end-to-end agent run. Its spans share a trace ID and form a parent-child tree. |
| **Session** | Related traces, such as a multi-turn conversation or the traces for one evaluation case. A stable session ID is the main grouping key across ingest paths. |
| **Evaluator result** | A score attached to an exact span and session. `NUMERIC` and `BOOLEAN` use `value`; `CATEGORICAL` and `TEXT` use `string_value`. |

For ATIF, top-level `extra.verifier_result.rewards = {criterion: score}` automatically creates a
`harbor.verifier` evaluator span and one evaluator result per criterion. Stock Harbor keeps rewards
in a separate `reward.json`; enrich the trajectory before ingesting or post results explicitly:

```bash
curl -X POST "$NMP_BASE_URL/apis/intake/v2/workspaces/$WORKSPACE/evaluator-results" \
  -H 'Content-Type: application/json' \
  -d '{"span_id":"<span-id>","session_id":"<session-id>","name":"faithfulness/v1","data_type":"NUMERIC","value":0.82}'
```

## Verify ingestion

Always query the interaction back; a successful POST alone is insufficient:

```bash
curl -g "$NMP_BASE_URL/apis/intake/v2/workspaces/$WORKSPACE/spans?filter[session_id]=<session-id>&page=1&page_size=100"
```

Confirm the response contains the expected session, trace/span hierarchy, inputs and outputs,
status/errors, and any evaluator results. Also query with `filter[agent_name]=<agent-name>` and
confirm the returned `agent_name` matches the stable value emitted by the agent. If the goal is to
create named evaluation runs and compare them in a leaderboard, hand off to
`nemo-experiments-upload`.
