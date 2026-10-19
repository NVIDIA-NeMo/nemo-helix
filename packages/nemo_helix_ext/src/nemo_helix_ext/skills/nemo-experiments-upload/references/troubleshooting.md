<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Common failures across the create + ingest endpoints, and the fix. Most 4xx bodies include a `detail`
string — read it first.

## Create Experiment / Evaluation

| Status | Meaning | Fix |
|---|---|---|
| `409` on create experiment/evaluation | Name already exists in this workspace | Reuse it — `GET .../experiments/{name}` or `.../evaluations/{name}` |
| `400 "…experiment … does not exist"` on create evaluation | `experiment_ids` is wrong or the experiment was deleted | Use the Experiment's **`id`** from the create-experiment response (not its name) |
| `400 "…parent … does not exist"` | `parent_evaluation_id` doesn't resolve | Omit it, or pass the parent Evaluation's entity **`id`** |
| `422` on create evaluation | Missing required field or non-string metadata value | Required: `name`, `experiment_ids`, `dataset_name`. `metadata` must be `dict[str, str]` |

## Ingest (all endpoints)

| Status | Meaning | Fix |
|---|---|---|
| `400 "Evaluation '…' must be created before it can be logged."` | Ingested before the Evaluation existed, or `evaluation_name` typo | Create the Evaluation first; ensure `evaluation_context.evaluation_name` matches it |
| `400 "Evaluation '…' has been deleted…"` | The referenced Evaluation is soft-deleted | Recreate it or target a live one |
| `422` on ATIF/chat-completions | Unknown top-level key (both are `extra="forbid"`) | Remove stray keys; check the schema in `../../nemo-intake/references/ingest-formats.md` |
| `422` bad `schema_version` (ATIF) | Not one of `ATIF-v1.0` … `ATIF-v1.7` | Use a supported literal |
| `422` non-sequential `step_id` / duplicate `tool_call_id` (ATIF) | Step/tool-call invariants violated | 1-based sequential `step_id`; unique `tool_call_id`; observation `source_call_id` must resolve |
| `422 cost_total_usd` unexpected (chat-completions) | Used the wrong cost key | Use top-level `cost_usd` (not `cost_total_usd`, not nested in `response`) |
| `415` on OTLP | Wrong content type | Send protobuf: `Content-Type: application/x-protobuf` |
| `413` on OTLP | Payload over the size cap | Export smaller batches |

## Data ingested but nothing shows up

| Symptom | Cause | Fix |
|---|---|---|
| Ingest returned 2xx but `run_count` stays 0 | Evaluation context is missing or doesn't match the target Evaluation | Set `evaluation_context.evaluation_name`; for OTLP, set `nemo.evaluation.name` on the root span |
| No scores on the evaluation | Rewards not under `extra.verifier_result.rewards` (ATIF), or wrong `data_type` (`/evaluator-results`) | ATIF: `extra.verifier_result.rewards = {criterion: value}`. Explicit: `NUMERIC`/`BOOLEAN` need `value`, `CATEGORICAL`/`TEXT` need `string_value` |
| No cost on the rollup | The producer never emitted cost | Cost is pass-through — set `cost_usd` (chat-completions / ATIF step `metrics`) or `llm.cost.total` / `gen_ai.usage.cost` (OTLP) |
| `503` on `GET .../evaluations/{name}` or `/sessions` | ClickHouse (telemetry store) not running | Start ClickHouse; rollups, sessions, and metric sorts/filters all need it |
| Studio shows no Experiments area | Feature flag off | Experiments viewing is behind `VITE_FF_EXPERIMENT` |

## Identifier cheat-sheet (the #1 source of bugs)

- `experiment_ids` (on create evaluation) → a list with the Experiment's **`id`**.
- OTLP evaluation attribute key → **`nemo.evaluation.name`** (test case → `nemo.test_case.name`).
- Parent → **`/experiments`** (`/experiment-groups` is a deprecated hidden alias); evaluations → **`/evaluations`**.
