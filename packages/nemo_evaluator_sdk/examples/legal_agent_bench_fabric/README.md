<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# legal_agent_bench_fabric — evaluate an agent on LAB, the NeMo Platform way

Run Harvey Labs' [Legal Agent Benchmark (LAB)](https://github.com/harveyai/harvey-labs) as **native
`AgentEvalTask`s** through **NeMo Fabric**, scored by **LAB's own rubric scorer** wrapped in a metric.
This is the task-driven counterpart to [`legal_agent_bench_harbor`](../legal_agent_bench_harbor):
execution (Fabric) and scoring (a metric) are decoupled.

```text
LAB raw task ──build──▶ AgentEvalTask ──Fabric run──▶ Trial(workspace + ATIF) ──LabRubricMetric──▶ Scores
   (title, docs,          (instruction+manuals,        (agent's deliverables       (calls LAB's own
    criteria, skills)      documents/ + skills/          under output/)              evaluation.score_rubric)
                           seeded, criteria=reference)
```

## Two design decisions that make it faithful

**1. Scoring = LAB's own code.** [`lab_rubric_metric.py`](lab_rubric_metric.py) doesn't reimplement the
rubric — it **imports LAB's `evaluation/` module at runtime** (from the pinned source the prep downloads,
never committed here) and calls
`score_rubric(criteria, run_dir, judge, task_desc, parallel)`. LAB's code does the document→text
extraction (incl. pandoc `--track-changes` for redlines), loads LAB's exact `rubric_criterion` judge
prompt, and applies all-pass aggregation — so fidelity comes for free. **The grading model is
pluggable:** LAB's native `Judge` routes by model-name prefix (`gpt-*`/`claude-*`/…) and uses the OpenAI
*Responses* API — neither fits a namespaced NVIDIA model id. So when you pass `--judge-base-url` (an
OpenAI-compatible endpoint like NVIDIA's), the metric swaps in a small adapter that reuses LAB's **exact
prompt + JSON parsing** over `chat.completions`; without it, LAB's native `Judge` is used.

**2. Skills = task inputs, not skill injection.** LAB gives the agent **all three** skills
(docx/pptx/xlsx) on every task, and a skill is just a `SKILL.md` manual + `scripts/`. So
[`prepare_lab_taskset.py`](prepare_lab_taskset.py) **seeds the skill directories into each workspace**
under `skills/<name>/` and prepends the manuals to the instruction — exactly what LAB does. This mirrors
LAB faithfully and **sidesteps Fabric's skill API limitations** (one-skill-per-runtime; container runner
has no skill support — see the limitations log).

## Files

- [`prepare_lab_taskset.py`](prepare_lab_taskset.py) — downloads + SHA-verifies the pinned LAB source and
  builds native tasks: documents → `documents/` seeds, skills → `skills/<name>/` seeds + manuals,
  criteria → grader-only `reference`.
- [`lab_rubric_metric.py`](lab_rubric_metric.py) — `LabRubricMetric`: reads deliverables from `workspace`
  evidence and scores them with **LAB's own `score_rubric`**.
- [`run_legal_agent_bench_fabric.py`](run_legal_agent_bench_fabric.py) — wires the Fabric runner (host or
  container) + `AgentEvaluator`.
- [`rescore.py`](rescore.py) — **re-grade a stored run bundle with a different judge, without re-running the
  agent** (see [Re-score a run](#re-score-a-run-with-a-different-judge)).

## How LAB maps onto the native model

| LAB source | Native construct |
|---|---|
| `title` | `AgentEvalTask.intent` + `reference["task_title"]` |
| `instructions` (+ skill manuals) | `inputs["instruction"]` |
| `documents/` | `inputs["files"]` seeded under `documents/` |
| `harness/skills/{docx,pptx,xlsx}` | `inputs["files"]` seeded under `skills/<name>/` |
| `criteria[]` | `reference["criteria"]` → scored by LAB's `score_rubric` |
| `evaluation/scoring.py` (+ `rubric_criterion` prompt) | imported at runtime & called by `LabRubricMetric`; grading model via a `chat.completions` judge adapter |

## Setup (one-time)

Run from the **`nemo-platform` repo root** with the project venv's Python directly — **not `uv run`**,
which re-syncs `.venv` to the lockfile and drops the out-of-lock `nemo_fabric` + adapters. `$FABRIC_REPO`
/ `$RELAY_REPO` are your NeMo-Fabric / NeMo-Relay checkouts (macOS builds them from source).

```bash
cd nemo-platform
make bootstrap-python     # base SDK env → .venv

# 1. NeMo Fabric. The `runtime` extra provides the importable `nemo_fabric` module (a separate
#    `nemo-fabric-runtime` package); the codex extra pulls prereleases that need explicit pins.
uv pip install --python .venv/bin/python "$FABRIC_REPO/python"                 # nemo-fabric-runtime (the nemo_fabric module)
uv pip install --python .venv/bin/python "$FABRIC_REPO[codex,relay,runtime]" \
    "openai-codex>=0.1.0b3" "openai-codex-cli-bin>=0.137.0a4" "sqlite-vec>=0.1.10a4"
.venv/bin/python -c "import nemo_fabric; print('nemo_fabric OK')"              # must pass before running

# 2. The harness. codex is the DEFAULT and the only one that runs LAB's docx/pptx/xlsx skill scripts under
#    Fabric — it is already installed by the `codex` extra in step 1, so nothing to add here. (deepagents is
#    NVIDIA-native but its shell tool is inert with the host FilesystemBackend, so it can't produce document
#    deliverables for LAB; install it only to experiment:)
# uv pip install --python .venv/bin/python "$FABRIC_REPO/adapters/deepagents"

# 3. LAB's scoring stack — the metric runs LAB's score_rubric in THIS process (mistralai is required
#    because LAB's judge.py imports every provider SDK at module load):
uv pip install --python .venv/bin/python \
    python-docx python-redlines python-pptx openpyxl pdfplumber markitdown pandas openai anthropic mistralai
# `anthropic` is REQUIRED: LAB's scoring.py imports it at module load. `mistralai`/`google-genai` are only
# needed for LAB's *native* prefix-routed Judge; the --judge-base-url adapter path does not import them.
# system tools: pandoc (e.g. `brew install pandoc`); libreoffice for the agent's docx/xlsx skill scripts.
```

## Run it

```bash
# The default codex harness authenticates via your ~/.codex login (real OpenAI); the judge runs on NVIDIA.
# Keep the two credential paths separate — do NOT point OPENAI_API_KEY/OPENAI_BASE_URL at NVIDIA, or codex
# would send the agent to the NVIDIA endpoint. Pass the judge endpoint explicitly instead.
export NVIDIA_API_KEY=...                                    # judge only (NVIDIA gpt-oss-120b)

.venv/bin/python -m packages.nemo_evaluator_sdk.examples.legal_agent_bench_fabric.run_legal_agent_bench_fabric \
    --runtime host --harness codex-cli --model gpt-5.5 \
    --judge-model openai/gpt-oss-120b \
    --judge-base-url https://integrate.api.nvidia.com/v1 --judge-api-key-env NVIDIA_API_KEY \
    --source-dir ./data/lab-source --output-dir ./results/lab-fabric \
    --limit 1 --parallelism 1 --no-trajectory
```

The codex harness is configured **closed-book** (web search disabled, `sandbox=workspace-write`) to match
LAB. The judge endpoint is passed explicitly with `--judge-base-url` (it otherwise defaults from
`$OPENAI_BASE_URL`). Aggregate scores print per run; a real grading of a strong deliverable (a complete
antitrust memo, re-scored from a stored codex run) looks like:

```text
lab_rubric.criteria_pass_rate: 0.76        # fraction of the rubric passed (38/50) — the primary signal
lab_rubric.n_passed / n_criteria: 38 / 50
lab_rubric.score:              0.0         # all-pass reward: 1.0 ONLY if every criterion passes
lab_rubric.all_pass:           mean=None   # boolean outputs don't average — expected, not an error
```

Per-criterion verdicts + judge reasoning are recorded in each row's `diagnostics` (in `scores.jsonl`), so you
can see exactly which of the 50 criteria failed and why — not just the aggregate.

Every run writes a bundle (`run.json`, `trials.jsonl`, `scores.jsonl`, `summary.json`, `report.html`).

## Re-score a run with a different judge

Because **execution and scoring are decoupled** — a run persists each trial's deliverables as durable
`workspace` filesystem evidence, and `LabRubricMetric` grades purely from that evidence + an injected judge
— you can re-grade an existing bundle with a *different* judge model/endpoint, **without re-running the
agent** (no agent invocations, no credits).

[`rescore.py`](rescore.py) does this the idiomatic way — it reloads the stored trials and feeds them to
the SDK's own imported-trials path, `AgentEvaluator().run(tasks=…, trials=…)`, with a fresh
`LabRubricMetric` bound to your judge. No agent runs; you get a **full re-scored bundle** at
`<run-dir>-rescored` (`scores.jsonl` with per-criterion diagnostics, `report.html`, aggregates) plus an
original-vs-rescored table:

```bash
# re-grade the run above with a *different* judge (llama-3.3-70b on inference-api) — agent never re-runs
NVIDIA_API_KEY=...
python -m packages.nemo_evaluator_sdk.examples.legal_agent_bench_fabric.rescore \
    --run-dir ./results/lab-fabric \
    --judge-model nvidia/meta/llama-3.3-70b-instruct \
    --judge-base-url https://inference-api.nvidia.com/v1 --judge-api-key-env NVIDIA_API_KEY \
    --judge-parallel 4 --judge-min-interval 0.5
```

```text
task (area)                      original   rescored   passed/total
----------------------------------------------------------------------
corporate-ma                          0.68       0.81   46/57
employment-labor                      0.81       0.95   56/59
intellectual-property                 0.80       0.85   46/54
real-estate                           0.86       0.85   55/65
```

The shift (a more lenient judge scores higher) makes the **pluggable judge** concrete — the judge is a
constructor argument (`build_lab_judge`), so swapping models/endpoints is a flag, not a code change. It's
also how you recover from a flaky judge endpoint mid-benchmark: re-score the already-produced deliverables
against a different endpoint instead of paying for a full re-run. (Non-reasoning judges like `llama-3.3-70b`
are also *much* faster than reasoning models on LAB's huge redline prompts — seconds vs minutes per call.)

## Harnesses (`--harness`)

| Harness | Runs LAB? | Notes |
|---|---|---|
| **`codex`** (default) | ✅ | The **only** harness whose shell tool actually runs LAB's docx/pptx/xlsx skill scripts under Fabric. OpenAI-provider-locked (auth via your `~/.codex` login, `CODEX_HOME`); configured **closed-book** here (web search off, `sandbox=workspace-write`). Agent runs on OpenAI; the judge still runs on NVIDIA. |
| `deepagents` | ❌ for LAB | NVIDIA-native LangChain Deep Agents, but its `execute` shell tool is **inert** with Fabric's host `FilesystemBackend` — so it can't run the skill scripts or produce document deliverables (it emitted an empty stub for LAB). Fine for non-document agents. |
| `hermes` | ⚠️ | Provider-agnostic, but blocked today by a `requests==2.33.0` pin conflict (PR #778). Usable once resolved. |

## Gotchas we hit (so you don't)

- **Use `.venv/bin/python`, not `uv run`** — `uv run` re-syncs and removes the out-of-lock `nemo_fabric`.
- **`--no-trajectory`** — current nemo-fabric renamed `FabricConfig.enable_relay(config=…)`; without this
  flag the host runtime crashes the trial. LAB scoring doesn't use the trajectory.
- **Agent model** — some NVIDIA models time out (`meta/llama-3.3-70b-instruct` did); `meta/llama-3.1-70b-instruct`
  and `openai/gpt-oss-20b` respond reliably.
- **Judge** — LAB's native `Judge` can't reach NVIDIA (prefix routing + Responses API); the
  `--judge-base-url` adapter handles it with LAB's exact prompt over `chat.completions`.
- **Closed-book** — LAB is a provided-documents-only benchmark: its reference harness has **no web tool**
  and runs `--network=none`. For fidelity, disable the agent's web/search tools (the `codex` harness may
  web-search by default). True network isolation needs the container runtime with a no-network sandbox.
- **Scoring reads `output/`** — LAB's `score_rubric` appends `output/` to the run dir itself; the metric
  hands it the workspace **root** (not `<root>/output`). Passing the wrong level silently grades empty text.

## What runs *where*

- **Agent environment** (host, or the container `--image`): the document toolchain (pandoc,
  libreoffice/`soffice`, node, python-docx/docxtpl/python-redlines/python-pptx/openpyxl) so the seeded
  skill scripts run. For `--runtime container`, pass a prebuilt `--image` (the `image` param of
  `FabricAgentRuntime` in sandbox mode); we deliver skills as workspace seeds, so no skill injection is
  needed.
- **Eval process** (LAB's `score_rubric`): the scoring stack from step 3 above.

## Fidelity & scope

- **Scoring is LAB's own** (exact prompt, extraction, redline handling, aggregation) — high fidelity.
- **This measures *your* agent**, not LAB's reference agent, so it will not reproduce LAB's official
  leaderboard number; for that, run LAB's reference agent (the Harbor path's domain).
- **Scale**: local async concurrency or a governed platform job. The full 1,749-task sweep is best run as
  a platform job rather than locally.

See [`legal_agent_bench_harbor`](../legal_agent_bench_harbor) for the in-container-verifier counterpart.
