<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# harbor_to_intake — run a real benchmark, publish the scores to Intake

Two takes on the same round trip, both pulling a benchmark from **Harbor Hub** rather than using a
toy dataset:

| | |
| --- | --- |
| [`run_harbor_to_intake.py`](run_harbor_to_intake.py) | the whole flow as a script |
| [`harbor_to_intake.ipynb`](harbor_to_intake.ipynb) | notebook — also shows what a Harbor task is made of |

![Harbor Hub to Intake pipeline](pipeline.png)

Harbor runs each task in its own container and a verifier script stamps a reward. Intake stores the
trajectory and its scores under an Evaluation. `publish_to_intake` is what joins them, and it is an
explicit call — `AgentEvaluator.run()` has no Intake side effect and there is no feature flag.

Both default to two **Terminal-Bench 2.1** tasks solved by **`codex`**, a real agent working inside
the task container. Terminal-Bench is Apache-2.0 and its tasks ship prebuilt images, so a run pulls
rather than builds. Tasks are fetched individually — a task id is an `org/task` reference the Hub
serves on its own, so there is no need to pull all 89. What lands in Intake is that agent's
actual trajectory — the problem it was given, its reasoning, the commands it ran, and the answer it
wrote — so the scores are a genuine measurement.

`--agent oracle` replays each task's bundled reference solution instead: no credentials, 1.0 on
everything, and a single-step trajectory, since a replay has no agent turns to record. It is a
pipeline smoke test, not an evaluation.

## Prerequisites

- **Python >= 3.12** with the Harbor extra. The SDK's own floor now matches Harbor's `>=3.12`, so
  Harbor is a plain optional dependency — imported lazily and never installed by default:
  ```bash
  uv pip install "harbor>=0.20,<0.21"
  ```
- **Docker**, running. Harbor needs it for the task containers, and Intake needs it for ClickHouse.
- **The platform**, running at least `auth,entities,intake`.
  The example builds its client from a base URL alone, which is direct mode: it sends no auth
  headers. That suits a local platform; against an auth-enabled deployment the calls would 401.
- **A logged-in `codex`** for the default agent — `codex login`, or `OPENAI_API_KEY`. Both artifacts
  pick up the `auth.json` that `codex login` writes and say so; `--agent oracle` needs neither.
- **A built Studio bundle**, only if you want the Studio links at the end to open anything. In a
  source checkout Studio serves `web/packages/studio/dist`, which does not exist until you build it:
  ```bash
  make bootstrap-studio
  ```
  It needs the Node.js and pnpm versions pinned in the Flox environment. Everything else in the
  example works without it — Studio starts either way and shows a "not built" notice in place of the
  UI.

## Standing it up

**The notebook does this itself** — its first section starts the platform and waits for readiness,
so there is no second terminal to manage. A platform already answering on `BASE_URL` is reused
rather than killed, so it is safe to re-run and safe alongside one you started yourself. It stops
what it started when the kernel shuts down.

For the script, start the platform yourself:

```bash
NHX_BASE_URL=http://localhost:8080 uv run nemo services run --services auth,entities,intake
```

Wait for readiness. First startup provisions ClickHouse, so give it around 30 seconds:

```bash
curl -sf http://localhost:8080/health/ready
```

Intake is ClickHouse-backed and provisions a managed ClickHouse container itself as long as nothing
has pointed it at an operator-owned one — that is, `NHX_INTAKE_CLICKHOUSE_URL` is unset *and* the
resolved URL is still the default `http://localhost:8123`. On a stock checkout both hold, so there
is nothing to start separately.

## Running it

```bash
uv run plugins/nemo-evaluator/examples/harbor_to_intake/run_harbor_to_intake.py
```

The first run downloads the tasks into `~/.cache/harbor-to-intake/tasks/`; later runs reuse them.
Useful overrides:

| Flag | Default | |
| --- | --- | --- |
| `--dataset` | `terminal-bench-2-1` | label only, used to name the Evaluation |
| `--tasks` | two Terminal-Bench tasks | Hub task ids to download and run, as `org/task` |
| `--agent` | `codex` | any Harbor agent |
| `--model` | `gpt-5.6-luna` | must be a model your agent's account can use; ignored by `oracle`/`nop` |
| `--experiment` | `harbor-demo` | Experiment the Evaluations are grouped under |
| `--evaluation` | one per run | pin a name to gather several runs into one Evaluation |

The notebook exposes the same choices as constants in its first cell.

### Running without credentials

`oracle` replays each task's reference solution and scores 1.0; `nop` does nothing and scores 0.0.
Neither calls a model, so neither needs credentials:

```bash
uv run plugins/nemo-evaluator/examples/harbor_to_intake/run_harbor_to_intake.py --agent oracle
```

Neither measures anything, and neither produces a real trajectory — use them to confirm Docker, the
download and the Intake round trip are healthy before spending tokens.

### Using a different agent

`--agent` takes any Harbor agent, and each brings its own credentials. `terminus-2` resolves models
through LiteLLM, so a `nvidia_nim/` model reads `NVIDIA_NIM_API_KEY`:

```bash
export NVIDIA_NIM_API_KEY=...
uv run plugins/nemo-evaluator/examples/harbor_to_intake/run_harbor_to_intake.py --agent terminus-2 --model nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b
```

Only agents that emit ATIF produce a full trajectory in Intake — `codex` does, and so does
`nemo-agent`. An agent without ATIF support still publishes, but with the single-step trajectory.

**Only the `codex` and `oracle` paths have been run.** Other agents are wired but untested here.

## Choosing a benchmark

Harbor Hub carries around 80 datasets, including SWE-bench Verified, GAIA, GPQA-Diamond and
aider-polyglot. Browse them at [hub.harborframework.com](https://hub.harborframework.com/datasets),
or list the legacy registry:

```bash
curl -s https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json | jq -r '.[].name'
```

**Check the licence before you pick one.** The registry records no licence field, and most datasets
are aggregated into `harbor-framework/harbor-datasets`, which declares none — the licence that
matters is the underlying benchmark's. Terminal-Bench is the default here partly because it lives in
its own Apache-2.0 repo (`harbor-framework/terminal-bench-2`).

Two things dominate how long a run takes, and neither is the number of tasks you select:

- **Images.** A task either builds from its own `environment/Dockerfile` or names a prebuilt one.
  Terminal-Bench 2.1 uses prebuilt images of roughly 330 MB each, pulled once and cached;
  SWE-bench-style datasets are far heavier. Run a new task once *before* you rely on it.
- **Task difficulty.** Terminal-Bench tasks are sized for an expert to take an hour or more. Their
  `task.toml` carries `estimated_duration_sec`; sort by it before picking. `largest-eigenval` ran
  the agent for 508s and still scored 0.0, against 67s for `git-leak-recovery`.
- **Agent setup.** An installed agent is installed *into each task container*, once per trial. For
  codex that is 30-68s, and it dominates — more than the agent's own reasoning.

Measured on AIME over 23 codex trials on a warm image cache — the shape holds for any dataset,
since agent setup is per-trial and independent of the task:

| phase | time |
| --- | --- |
| environment (image) | 2-3s |
| agent setup (installing codex in the container) | 30-68s |
| agent execution (the model actually working) | 14-60s, median 20s |
| verifier | <1s |
| **whole trial** | **44-119s, median 75s** |

Two tasks run concurrently, so wall clock is roughly the slower trial rather than the sum.

For the Terminal-Bench defaults, `--agent oracle` ran both tasks in 63s including the first pull of
both images. That is the quickest way to check the plumbing without spending tokens.

## What each step is doing

1. **`harbor download <org/task> --export -o <dir>`** — drops each task folder straight into
   `<dir>`, making `<dir>` itself the "directory whose subdirectories are task folders" the runtime
   discovers. No reshaping, no conversion, and about 3 seconds per task.
2. **`AgentEvaluator().run(tasks=..., target=...)`** — the evaluator needs the tasks to score and a
   target that produces trials for them. `HarborTasksetLoader` supplies the first from the task
   folders, `HarborAgentTaskRunner` the second by running each task in Docker; scoring is
   `HarborRewardMetric`. The caller never imports `harbor`. `run_harbor_eval` is a one-call wrapper
   around exactly this.
3. **Create the Experiment and Evaluation** — `publish_to_intake` references an Evaluation that
   already exists and never creates one; ATIF ingest rejects an unknown name with HTTP 400. Both
   creates pass `exist_ok=True`, so re-running is safe.
4. **`publish_to_intake(result, ...)`** — per trial, posts the ATIF trajectory, resolves its root
   span, and writes one evaluator-result row per metric output. Publishing is not atomic: every
   trial that can land does, and failures are raised together carrying a partial report rather
   than aborting on the first one. The script holds its result in memory only, so a publish that
   fails outright still means re-running the tasks. It is idempotent — the session id and row ids derive
   from the run and trial rather than the clock, so re-publishing replaces rows instead of
   duplicating them.
5. **Read back** — the same rows queried through the Intake API, which is what makes this a round
   trip rather than a print statement.
6. **Studio links** — URLs for the Evaluation and for each trial's trajectory. The notebook renders
   them as clickable links; the script prints them.

## Viewing the run in Studio

Both artifacts end by emitting Studio URLs:

```
View in Studio:
Evaluation:
  http://localhost:8080/studio/workspaces/default/experiment/harbor-demo/terminal-bench-2-1-agent-eval-20260821140131-ca481809
Trial git-leak-recovery__8D328Gd:
  http://localhost:8080/studio/workspaces/default/intake/sessions/<session-id>
```

These mirror two destinations Studio publishes in
[`nhx.studio.studio_links`](../../../../services/studio/src/nhx/studio/studio_links.py) —
`experiment_detail` for the Evaluation and `intake_session` for one trial's trajectory — so they
track Studio's own routing rather than being hand-built paths.

**They only resolve if Studio is running.** The notebook already includes `studio` in the services
it starts; for the script, add it yourself:

```bash
NHX_BASE_URL=http://localhost:8080 uv run nemo services run --services auth,entities,intake,studio
```

Studio also serves a built Vite bundle. Without one it starts fine and shows a "not built" notice
instead of the UI, so build it once:

```bash
make bootstrap-studio
```

The links print either way — a run without Studio produces correct URLs that 404 until it is up.

## Comparing two agents

Both artifacts publish into the same `harbor-demo` Experiment under their own Evaluation, which is
what an Experiment is for. To compare agents, run the same tasks twice under different Evaluation
names:

```bash
uv run plugins/nemo-evaluator/examples/harbor_to_intake/run_harbor_to_intake.py --agent oracle --evaluation tb-oracle-ceiling
uv run plugins/nemo-evaluator/examples/harbor_to_intake/run_harbor_to_intake.py --agent nop    --evaluation tb-nop-floor
```

Intake can then roll the two up over identical tasks. Those two are the ceiling and the floor —
1.0 everywhere against 0.0 everywhere — needing no credentials and about 10 seconds per leg on a
warm cache, so it is a quick way to see two Evaluations sitting side by side under one Experiment.
Substitute a real agent for either one when you want a genuine comparison.

## Teardown

Stop the platform with Ctrl-C. A graceful shutdown stops the managed ClickHouse container without
removing it or its data, and the next startup reuses it — so the second demo run is faster than the
first. Killing the runner by signal instead can leave the container running; it is still reused, so
this costs nothing but a stray container.

## Troubleshooting

**`RewardFileNotFoundError`.** Harbor bind-mounts the container's `/logs` back to the job directory
to collect the verifier reward, and some macOS Docker backends (colima, for one) only reflect
bind-mount writes for paths they share — often `$HOME` but not `/tmp`. That is why both artifacts
put their job directory under `~/.cache/`; move it with the script's `--jobs-dir` or the notebook's
`WORK_DIR`. Linux Docker shares all paths.

**Intake returns 503.** ClickHouse is not reachable. Note that this does *not* show up as an
unready platform: Intake starts and reports itself ready either way, and serves its
ClickHouse-backed endpoints with 503 until ClickHouse turns up. The script's preflight query
catches it before the evaluation runs; in the notebook it surfaces at the publish cell, where you
can fix ClickHouse and re-run that cell alone, since the evaluation result is still in memory. If
`NHX_INTAKE_CLICKHOUSE_URL` is set, Intake uses that instance rather than provisioning one, so
check that it is actually up.

**Ingest rejects the Evaluation.** The Evaluation must exist before publish. If you renamed it via
`--evaluation`, that name has to be created first — which is what step 3 does.

**The agent scores 0.0 on everything.** `nop` does this by design — it is the floor, not a fault.
For a real agent, check the model slug resolves and the key is live before assuming the agent is
bad; an auth failure surfaces as a failed trial, not as a crash. A `--agent oracle` run over the
same tasks tells you whether the task and verifier are fine.

Read the task's `instruction.md` and the agent's trajectory in the job directory before concluding
the model is weak. A verifier checks for a specific end state, so an agent that solved the problem
but wrote its result somewhere unexpected still scores 0.0.

## Related

- [`packages/nemo_evaluator_sdk/examples/harbor/`](../../../../packages/nemo_evaluator_sdk/examples/harbor/) —
  the Harbor runtime on its own, without the Intake half, over a bundled local dataset.
- [`plugins/nemo-evaluator/tests/integration/test_publish_to_intake.py`](../../tests/integration/test_publish_to_intake.py) —
  the same round trip as an integration test, including the idempotency and skipped-score cases.
