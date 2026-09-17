<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# legal_agent_bench_harbor — run Harvey Labs' LAB through the SDK's Harbor runner

Run the public [Legal Agent Benchmark (LAB)](https://github.com/harveyai/harvey-labs) as a
**[Harbor](https://www.harborframework.com) task suite** through NeMo Evaluator's native Harbor
runner. LAB ships **raw** tasks (`tasks/**/task.json` + `documents/`); this example is **self-contained**
— it downloads the pinned source and *generates* the Harbor suite itself, then runs and scores it with
one `AgentEvaluator` call.

## Prerequisites, seams & caveats

- **Not zero-dependency**: Python ≥ 3.12, Docker, and a NeMo Platform source checkout
  (see [SETUP.md](../../../../SETUP.md) for toolchain prerequisites). The SDK is not
  published as a standalone PyPI package. From the repository root, install Harbor with
  `uv sync --frozen --package nemo-evaluator-sdk --extra harbor`. Harbor native runtime is early-access.
- **Reproducing LAB's official reference-agent number** additionally requires wiring **LAB's reference
  agent** (as an `--agent-import-path` adapter) and
  LAB's **exact** `rubric_criterion` judge prompt into `lab_verify.py`. Out of the box this generates a
  *runnable, faithful-in-shape* suite; treat scores as comparable-in-method until you drop those in.
- **Agent-output seam**: `prepare_lab_suite.py --run-dir` sets where the verifier reads the agent's
  deliverables (default `/logs/agent/artifacts/lab-run`, LAB's reference-agent location). Point it at
  wherever your chosen Harbor agent writes.
- **`scores.json` schema**: `LabCriteriaMetric` reads `n_criteria`, `n_passed`, `all_pass`,
  `judge_error_count`, `criteria_results[].verdict` — exactly what `lab_verify.py` writes.
- **Scale**: the SDK runs tasks with async concurrency locally (or a single-container platform job).
  For the full 1,749-task sweep, prefer the governed platform job over a local run.

## Files

- [`prepare_lab_suite.py`](prepare_lab_suite.py) — self-contained: downloads + SHA-verifies the pinned
  LAB source and **generates** one Harbor task folder per LAB task (documents, `instruction.md`,
  `task.toml`, `environment/Dockerfile`, `tests/`).
- [`lab_verify.py`](lab_verify.py) — the **in-container rubric verifier** (SDK-free) that
  `prepare_lab_suite.py` copies into each task; grades each criterion PASS/FAIL and writes
  `verifier/scores.json` + `verifier/reward.json`.
- [`run_legal_agent_bench.py`](run_legal_agent_bench.py) — runs the suite via `HarborAgentTaskRunner`.
- [`lab_criteria_metric.py`](lab_criteria_metric.py) — host-side metric that turns LAB's rubric into
  per-criterion component scores (reads the `verifier/scores.json` the verifier writes).

## 1. Generate the suite (self-contained)

```bash
uv run -m packages.nemo_evaluator_sdk.examples.legal_agent_bench_harbor.prepare_lab_suite \
    --source-dir ./data/lab-source \
    --out-dir    ./data/lab-harbor-suite \
    --judge-base-url https://integrate.api.nvidia.com/v1 \
    --judge-model    "meta/llama-3.3-70b-instruct" \
    --limit 5     # omit for all 1,749 tasks
```

This downloads the pinned LAB archive (verifying `SHA-256`), then writes a plain Harbor suite (no
`all.jsonl` index, no cache markers). `--judge-*` bake the judge endpoint into each task's
`[verifier.env]`; omit `--judge-api-key` and inject the key another way if you'd rather not write a
secret to disk.

## 2. Run and score it

`--agent-name` picks a built-in Harbor agent; use `--agent-import-path` for your own.

```bash
uv run -m packages.nemo_evaluator_sdk.examples.legal_agent_bench_harbor.run_legal_agent_bench \
    --dataset-path ./data/lab-harbor-suite \
    --agent-name oracle \
    --model your-model \
    --mode components --limit 5
```

`run_harbor_eval` / `HarborAgentTaskRunner` discovers the tasks, runs each in a Docker sandbox, and
scores its verifier reward with `HarborRewardMetric`; `--mode components` also attaches
`LabCriteriaMetric` for per-criterion scores. Every run writes an agent-eval bundle (`run.json`,
`trials.jsonl`, `scores.jsonl`, `summary.json`, `report.html`).

## `--mode components` (per-criterion scoring in one run)

LAB is a **rubric** benchmark, so a single pass/fail reward per row throws away most of the signal.
[`LabCriteriaMetric`](lab_criteria_metric.py) reads the verifier's `scores.json` and reports both
the official all-pass reward *and* the component breakdown in one run:

```text
harbor_reward.reward:            mean=0.42   # all-pass reward (1.0 iff every criterion passes)
lab_criteria.criteria_pass_rate: mean=0.78
lab_criteria.all_criteria_pass:  mean=0.42
lab_criteria.n_passed / n_criteria
lab_criteria.judge_error_count:  mean=0.0    # treat > 0 as an infra failure, not a model miss
view.legal_quality:              mean=0.60   # MEAN(reward, criteria_pass_rate)
```

For the **task-driven, bring-your-own-agent** counterpart (native `AgentEvalTask`s + Fabric + a rubric
*metric* instead of an in-container verifier), see [`../legal_agent_bench_fabric`](../legal_agent_bench_fabric).
