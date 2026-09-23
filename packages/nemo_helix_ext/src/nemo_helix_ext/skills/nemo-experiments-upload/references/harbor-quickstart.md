<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Harbor quickstart

Harbor is the most common producer. A Harbor run executes tasks (test cases); with `n_attempts > 1`
each task runs several times (trials). You upload **one ATIF payload per trial** to
`POST .../ingest/atif`, and Harbor's verifier rewards become evaluator scores automatically.

## Mapping: Harbor run → NeMo entities

| Harbor concept | NeMo entity | How |
|---|---|---|
| A benchmark / sweep | **Experiment** | `POST /experiments` once |
| One agent+config on that benchmark | **Evaluation** | `POST /evaluations` once (its `name` is your `evaluation_name`) |
| A task / test case | `test_case_name` | field inside `evaluation_context` |
| One trial (attempt) of a task | one ingested **session** | one `POST /ingest/atif` |

## Mapping: Harbor trial files → ATIF payload

Per trial, Harbor writes result files (typically `result.json` and `agent/trajectory.json`). Map:

| ATIF field | Harbor source |
|---|---|
| `agent.name` / `agent.version` / `agent.model_name` | the agent under test |
| `steps[]` | the trajectory steps (`agent`/`user`/`system`, with `metrics.{prompt_tokens, completion_tokens, cost_usd}`) |
| `final_metrics.{total_prompt_tokens, total_completion_tokens, total_cost_usd, total_steps}` | trajectory `final_metrics` or the trial's `agent_result` (`n_input_tokens`, `n_output_tokens`, `cost_usd`) |
| `evaluation_context.evaluation_name` | the target Evaluation (constant across the whole run) |
| `evaluation_context.test_case_name` | the task name (constant across that task's trials) |
| `extra.verifier_result.rewards` | the verifier's per-criterion scores → one evaluator score row each |
| `extra.error.{type,message}` | a terminal agent execution error from the trial result → root span error status/details |

**Cost/tokens are pass-through.** NeMo does not recompute cost — it sums the per-call `cost_usd` /
token values Harbor recorded. If Harbor didn't record a cost for a run, that run simply has no cost
(null, not zero).

See the shared `../../nemo-intake/references/ingest-formats.md` reference for the exact
`extra.error` shape and root-span behavior.

## Minimal per-trial payload

```json
{
  "schema_version": "ATIF-v1.5",
  "session_id": "<trial-uuid>",
  "evaluation_context": { "evaluation_name": "my-eval-baseline", "test_case_name": "tau-bench/airline-042" },
  "agent": { "name": "my-agent", "version": "1.0.0", "model_name": "provider/model" },
  "final_metrics": { "total_prompt_tokens": 51701, "total_completion_tokens": 255, "total_cost_usd": 0.264, "total_steps": 3 },
  "extra": {
    "task_id": "tau-bench/airline-042",
    "verifier_result": { "rewards": { "reward": 1.0 } }
  },
  "steps": [
    { "step_id": 1, "timestamp": "2026-01-01T00:00:00Z", "source": "user", "message": "..." },
    { "step_id": 2, "timestamp": "2026-01-01T00:00:01Z", "source": "agent", "model_name": "provider/model", "message": "..." }
  ]
}
```

## Consistency rules (so rollups aggregate correctly)

- **`evaluation_name` is identical** for every trial of the run.
- **`test_case_name` is identical** across all trials of the same task, and **differs** between tasks —
  this is what lets the platform group a task's k attempts.
- Give each trial a distinct `session_id` (one session = one run in the rollup's `run_count`).

## Reference: the scaled-evals integration

`scaled-evals` already does exactly this (raw HTTP to `/ingest/atif`, canonical `evaluation_context`,
`str`-only metadata). If you're wiring a new Harbor-based producer, mirror its intake module
(`src/scaled_evals/intake/`) rather than reinventing the payload.
