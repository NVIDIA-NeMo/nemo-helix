<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ingest formats

Full request schemas for the three intake ingest endpoints. All are under
`/apis/intake/v2/workspaces/{workspace}/`. Every one attaches evaluation identity the same way — an
`evaluation_context` object (or, for OTLP, span attributes).

## `evaluation_context` (shared by all)

```json
{
  "evaluation_context": {
    "evaluation_name": "my-eval-baseline",
    "test_case_name": "dataset/case-001"
  }
}
```

- `test_case_name` is optional and identifies which task or test case the run covers.
- The referenced Evaluation **must already exist** (create it first) or the request is rejected with
  `400 "…must be created before it can be logged."`

---

## 1. ATIF (Harbor / agent trajectories) — most common

`POST .../ingest/atif` → `201` (empty body). Body model is `AtifIngestRequest` with
`extra="forbid"` (unknown top-level keys → `422`).

**Required:** `schema_version` (one of `"ATIF-v1.0"` … `"ATIF-v1.7"`) and `agent` (`{name, version, ...}`).

**Common optional:** `session_id`, `trajectory_id`, `final_metrics`, `steps[]`, `extra`, `evaluation_context`.

**Evaluator scores** attach via `extra.verifier_result` — either `rewards` (a dict, one key per
criterion → one score row each) or a bare `score` (→ a single `"reward"` row). This happens
automatically; you don't call `/evaluator-results` separately for Harbor runs.

**Terminal execution errors** attach via `extra.error`. Intake marks the trajectory's root span as
`error` and promotes string `type` and `message` values into the span's queryable error fields. ATIF
does not define a standard trajectory-level error field; this is NeMo's documented root-`extra`
convention. Omit it for trajectories without a terminal execution error.

```json
{
  "extra": {
    "error": {
      "type": "AgentTimeoutError",
      "message": "Agent execution timed out after 900.0 seconds"
    }
  }
}
```

```json
{
  "schema_version": "ATIF-v1.5",
  "session_id": "d074dfb7-3691-443c-b137-720d75e40afa",
  "evaluation_context": { "evaluation_name": "my-eval-baseline", "test_case_name": "my-dataset/case-a" },
  "agent": { "name": "my-agent", "version": "1.0.0", "model_name": "provider/model" },
  "final_metrics": {
    "total_prompt_tokens": 51701, "total_completion_tokens": 255,
    "total_cost_usd": 0.264321, "total_steps": 3
  },
  "extra": {
    "task_id": "my-dataset/case-a",
    "verifier_result": { "rewards": { "correctness": 0.75, "structure": 1.0 } }
  },
  "steps": [
    { "step_id": 1, "timestamp": "2026-01-01T00:00:00Z", "source": "user", "message": "..." },
    { "step_id": 2, "timestamp": "2026-01-01T00:00:01Z", "source": "agent",
      "model_name": "provider/model", "message": "...",
      "tool_calls": [ { "tool_call_id": "c1", "function_name": "Bash", "arguments": {} } ],
      "observation": { "results": [ { "source_call_id": "c1", "content": "..." } ] },
      "metrics": { "prompt_tokens": 25773, "completion_tokens": 131, "cost_usd": 0.13123 } }
  ]
}
```

- `steps[].step_id` must be 1-based and sequential; `tool_call_id`s unique; each observation
  `source_call_id` must resolve to a tool call in the same step.
- Two rewards under `extra.verifier_result.rewards` → two `evaluator_results` rows named
  `correctness` and `structure`, aggregated per-evaluator on the read model.
- A root `extra.error` object marks only the trajectory's root span as errored. Failed tool calls
  remain errors on their own tool spans and do not implicitly fail the trajectory.

See `../../nemo-experiments-upload/references/harbor-quickstart.md` for mapping a Harbor trial result
to this shape and publishing it as an Evaluation.

---

## 2. chat-completions (a single captured model call)

`POST .../ingest/chat-completions` → `201` with `{session_id, span_id}`. Body model is
`ChatCompletionsIngestRequest` (`extra="forbid"`); the nested `request`/`response` are `extra="allow"`
(pass provider fields straight through).

**Required:** `request` (OpenAI-style, needs `model` + `messages`) and `response` (needs exactly one of
`choices` or `error`).

**Optional:** `session_id`, `trace_id`, `provider`, `cost_usd` / `cost_input_usd` / `cost_output_usd`
(all ≥ 0), `cost_details` (extra breakdown), `evaluation_context`.

```json
{
  "request": {
    "model": "gpt-4o-mini",
    "messages": [
      { "role": "system", "content": "You are a terse calculator." },
      { "role": "user", "content": "What is 6 times 7?" }
    ],
    "temperature": 0.2
  },
  "response": {
    "id": "chatcmpl-abc123", "object": "chat.completion", "model": "gpt-4o-mini-2024-08-06",
    "choices": [ { "index": 0, "message": { "role": "assistant", "content": "42" }, "finish_reason": "stop" } ],
    "usage": { "prompt_tokens": 24, "completion_tokens": 2, "total_tokens": 26 }
  },
  "session_id": "session-001",
  "provider": "openai",
  "cost_usd": 0.0001,
  "evaluation_context": { "evaluation_name": "my-eval-baseline", "test_case_name": "case-001" }
}
```

- Cost goes in the **top-level** `cost_usd` (not inside `response`). The producer alias
  `cost_total_usd` is **rejected** (`422`) — use `cost_usd`.
- Group related calls into one run with a shared `session_id` (and/or `trace_id`).

---

## 3. OTLP (OpenInference or OTel GenAI spans)

`POST .../ingest/otlp/v1/traces` with `Content-Type: application/x-protobuf` (a standard
OTLP/HTTP trace export). Response `{ "errors": [] }` (per-span errors collected; HTTP stays 200).

Use this path only when spans carry OpenInference or OTel GenAI semantic conventions. If the runtime
has neither, capture its OpenAI-compatible requests and responses and use `ingest/chat-completions`;
generic OTel spans do not provide the semantic fields needed for useful model and tool telemetry.

There is **no JSON `evaluation_context`** here — evaluation identity travels as **span attributes** on
the root span:

| Meaning | Span attribute key |
|---|---|
| Evaluation | **`nemo.evaluation.name`** |
| Test case | **`nemo.test_case.name`** |

These attributes correspond to `evaluation_name` and `test_case_name` in the JSON `evaluation_context`
used by the other ingest endpoints.

Cost/token/model attributes are read from standard GenAI / OpenInference keys (first match wins):

| Field | Source keys |
|---|---|
| input tokens | `gen_ai.usage.input_tokens`, `llm.token_count.prompt` |
| output tokens | `gen_ai.usage.output_tokens`, `llm.token_count.completion` |
| cached tokens | `gen_ai.usage.cached_tokens`, `gen_ai.usage.input_cache_tokens`, `llm.token_count.cached` |
| total tokens | `gen_ai.usage.total_tokens`, `llm.token_count.total` |
| total cost (USD) | `gen_ai.usage.cost`, `llm.cost.total` |
| input / output cost | `llm.cost.prompt` / `llm.cost.completion` |
| model | `gen_ai.request.model`, `gen_ai.response.model`, `llm.model_name` |
| provider | `gen_ai.system`, `gen_ai.provider.name`, `llm.provider` |
| session id | `gen_ai.conversation.id`, `session.id` |

Point any OTLP exporter at the endpoint and force HTTP/protobuf — some SDKs default to gRPC, which
won't reach this `/v1/traces` HTTP endpoint:

```bash
: "${NHX_BASE_URL:=http://localhost:8080}"
: "${WORKSPACE:=default}"
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="${NHX_BASE_URL}/apis/intake/v2/workspaces/${WORKSPACE}/ingest/otlp/v1/traces"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL="http/protobuf"
```

Then set `nemo.evaluation.name` (+ `nemo.test_case.name`) on the root span of each run.

---

## 4. Direct spans (historical imports)

`POST .../ingest/spans` with JSON → empty `201`. The strict body is one source plus 1–1000 spans:

```json
{
  "source": "my-trace-store",
  "spans": [
    {
      "span_id": "provider-span-id",
      "trace_id": "provider-trace-id",
      "session_id": "conversation-id",
      "parent_span_id": null,
      "name": "agent-run",
      "kind": "AGENT",
      "status": "success",
      "started_at": "2026-08-01T00:00:00Z",
      "ended_at": "2026-08-01T00:00:01Z",
      "input": {"question": "Hello"},
      "output": {"answer": "Hi"},
      "attributes": {"gen_ai.request.model": "example-model", "provider.raw": {"native": true}}
    }
  ]
}
```

IDs are arbitrary non-empty strings; a missing session ID defaults to the trace ID, and a parent may
be outside the batch. Timestamps require an offset. Structured inputs, outputs, and unknown
attributes retain native JSON types. Known semantic attributes populate queryable fields; other
attributes appear in detailed reads under `raw_attributes`. Reposting the same
`(source, trace_id, span_id)` updates the existing logical span.

String inputs and outputs are stored verbatim; objects and arrays are serialized as JSON.

The default ClickHouse TTL is 90 days from `started_at`. If any span is outside that window, Intake
returns `422` before writing the batch and instructs the operator to increase the `spans` and
`trace_index` TTLs. Provider timestamps are never rewritten by the endpoint.

Use the bundled provider scripts instead of manually translating MLflow, LangSmith, Phoenix, or
Braintrust exports.

---

## Explicit evaluator results (any producer)

To attach scores without the ATIF verifier path:

`POST .../evaluator-results` → `201`. Model `EvaluatorResultInput` (`extra="forbid"`):

```json
{
  "span_id": "<root or evaluator span id>",
  "session_id": "<the run's session id>",
  "name": "faithfulness/v1",
  "value": 0.83,
  "data_type": "NUMERIC"
}
```

- `data_type` ∈ `NUMERIC | CATEGORICAL | BOOLEAN | TEXT`. `NUMERIC`/`BOOLEAN` need `value`
  (BOOLEAN must be `0` or `1`); `CATEGORICAL`/`TEXT` need `string_value`.
- The id is derived from `(workspace, session_id, span_id, name)` — re-posting upserts (idempotent).

## How results surface on read

`GET .../evaluations/{name}` hydrates rollups from ClickHouse:
`run_count`, `evaluator_names`, `aggregate_scores` (per-evaluator `{mean, median, p90, …, count}`),
`cost_usd`, `latency_ms`, `model_names`, `agent_names`. List sessions at
`GET .../evaluations/{name}/sessions` (`mode=summary|detailed`), each row carrying per-session
`evaluator_scores`, tokens, cost, and the `trace_id` to drill into.
