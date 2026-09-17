<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Run a NeMo Gym environment through NeMo Evaluator

`run_gym_eval.py` runs an **existing** NeMo Gym environment (the `mcqa` benchmark by default) through the Evaluator's `GymAgentTaskRunner` and scores it with `AgentEvaluator`. Use it when you already have a Gym environment and want to run and score it through NeMo Evaluator without migrating it. Gym owns execution *and* scoring; the runner shells out to the `gym` CLI and adapts the rollout bundle into trials, and `GymRewardMetric` surfaces Gym's per-attempt reward.

Mapping: one Gym dataset → one run; each distinct row → one `AgentEvalTask` (id = content hash of the row); each attempt (`--num-repeats`) → one `AgentEvalTrial`. So `--num-repeats 2` over a 5-row dataset yields 5 tasks × 2 = 10 trials.

## Prerequisites

**1. NeMo Gym installed in its own environment, with that environment's `bin` on `PATH`**, plus the target environment's own dependencies. Gym imports Ray at module load and nemo-platform excludes Ray by constraint, so the two generally cannot share a virtualenv; the runner resolves `gym` from `PATH` only. Environments ship in the `nemo-gym` wheel — `resources_servers` and friends install beside `nemo_gym`, configs and `data/example.jsonl` included — so no checkout is needed:

```bash
uv venv ~/gym-env --python 3.12
# tiktoken is mcqa's own dependency; each resources_server ships a requirements.txt
uv pip install --python ~/gym-env/bin/python nemo-gym tiktoken
export PATH="$HOME/gym-env/bin:$PATH"
```

The runner shells out to whatever `gym` is on PATH. There is deliberately no setting for a checkout, another venv, or a search root: this config becomes a serialized job spec when Gym runs as a platform job, and a local path means nothing on the other side of that boundary.

**2. Model credentials for the collector** — a gitignored `env.yaml` in the directory you run from:

```bash
cat > env.yaml <<'YAML'
policy_base_url: https://<your-openai-compatible-endpoint>/v1
policy_api_key: <key>
policy_model_name: <model, e.g. nvidia/meta/llama-3.3-70b-instruct>
YAML
```

> Gym searches the working directory, then its install root. This SDK never reads that file — the `gym` subprocess does.

## Run

From the directory holding `env.yaml`:

```bash
uv run python -m packages.nemo_evaluator_sdk.examples.gym.run_gym_eval
```

Working from a Gym checkout also makes its components take precedence over the packaged ones, which is how you reach an environment whose data the wheel does not carry (bulk train/validation splits are not git-tracked, so they are not in it).

Useful flags: `--resources-server`, `--agent`, `--model-type` (`inference_provider` for OpenAI-compatible **chat** endpoints; `openai_model` uses the OpenAI **Responses API** and 500s against chat-only endpoints), `--num-repeats`, `--dataset`, `--output-dir`.

For the full set of knobs the underlying `gym env start` / `gym eval run` commands accept, see the [NeMo Gym documentation](https://github.com/NVIDIA-NeMo/Gym). Anything `GymRuntimeConfig` does not expose as a field can be passed through with its `hydra_params` escape hatch — nested data such as `{"model": {"temperature": 0.7}}`, flattened to Hydra's override grammar and applied to `gym env start`.

Each run writes its bundle to a fresh temporary directory by default. Pass `--output-dir` to choose one, but give every run its own: the runner refuses to reuse a directory that already holds Gym rollout output (Gym appends to its failures sidecar, so reusing one would mix runs) and raises rather than clearing a prior run's results.

Expected output:

```text
discovered 5 tasks from .../resources_servers/mcqa/data/example.jsonl
=== RESULT ===
tasks: 5  trials: 10
aggregate scores:
  gym_reward.reward: mean=0.6

Run bundle (run.json, trials.jsonl, scores.jsonl, report.html): /var/folders/.../gym-eval-ab12cd34
```

## Read the results

`inspect_results.py` reads `summary.json` and shows each result layer: run aggregates from
`summary.scores`, ordered per-task values from `summary.task_outcomes("<metric_type>.<output>")`, and
runner-owned aggregates under `runner.gym.*`. That accessor returns models rather than nested dicts —
each row names its own `task_id` and `metric_name` — so the example needs no per-task accessor of its
own. A `null` value is a trial that failed before scoring, while an empty `trials` list means the
metric produced no usable measurement; a task the metric never measured is not returned at all.

Each record names the trial that produced it, so `trial_id` — not list position — is what joins two
outputs of the same task, or looks up `trials.jsonl`. A trial whose metric failed is absent
rather than `null`, so two lists for one task need not be the same length.

Values keep the metric's own type (a count stays an int, a judge's verdict stays a label), and each
carries a `value_type` of `number` / `label` / `missing`. Use `numeric_metric_values` before doing
arithmetic — it drops labels and keeps a dead trial's `null`.

No bundle is checked in; the run above produces one. Give it a stable `--output-dir` and point the reader at the same path:

```bash
uv run python -m packages.nemo_evaluator_sdk.examples.gym.run_gym_eval \
    --output-dir /tmp/gym-eval
uv run python -m packages.nemo_evaluator_sdk.examples.gym.inspect_results --bundle /tmp/gym-eval
```

Aggregates named `runner.gym.*` are Gym's own figures, imported into `summary.scores` so they are addressable exactly like the SDK's — the prefix is what tells you which side computed them. The script cross-checks Gym's `pass@1` against the SDK's natively-computed one (Gym reports accuracy on a 0-100 scale where the SDK uses 0-1). It works on any agent-eval bundle: pass `--metric-type`/`--output-name` for a run scored with a different metric.

## How it runs Gym

The runner uses Gym's **two-step** flow, which reads a dataset file directly (no split-driven data-prep, no HuggingFace downloads):

1. `gym env start …` — brings up the resources-server + agent + model servers.
2. `gym eval run --no-serve --input <dataset> …` — collects rollouts against them.

It does **not** import `nemo_gym` and does **not** handle secrets — it invokes whichever `gym` executable is on `PATH`, resolved once to an absolute path so the subprocesses cannot pick up a different one, and Gym reads credentials from the `env.yaml` in the directory you run from.

The dataset handed to step 2 is not your source file. The runner **materializes** a normalized one into the run's work directory: one row per requested task, with `_ng_task_index` stamped explicitly. Gym honors a caller-supplied `_ng_task_index` (it only assigns one when a row lacks it) and echoes it back on every rollout record, so rollouts join back to tasks through a map the runner owns rather than a guess about Gym's internal row ordering. It also means running a *subset* of tasks only rolls out that subset.

### Logs

Gym's subprocess output is streamed to files in the run's work directory — `gym_env.log` for `gym env start`, and `gym_eval.stdout.log` / `gym_eval.stderr.log` for the collection — and mirrored to the `nemo_evaluator_sdk.agent_eval.runtimes.gym` logger at `DEBUG`. Startup and collection failures name the relevant files and inline the last lines. To watch Gym's output in your own terminal, turn that logger up:

```python
logging.getLogger("nemo_evaluator_sdk.agent_eval.runtimes.gym").setLevel(logging.DEBUG)
```

## Notes & caveats

- **One distinct row is one task.** Task identity is the row's content hash, so duplicate rows collapse into a single task and the runner warns. Repeating a row is *not* how you ask for repeated attempts — `num_repeats` is, since attempts are a run-level concern. Duplicates usually mean a data problem.
- **Per-env deps are heterogeneous.** mcqa needs only `tiktoken`; other Gym envs pull `torch`/COMET/GPU or docker. The caller is responsible for a Gym runtime whose deps are installed.
- **`--no-serve --input` bypasses Gym's data-prep** (prompt templating / dataset materialization). This example's `example.jsonl` rows are already complete, so it's faithful; an env whose rows need templating would need that step first.
- This example covers the local subprocess path. Platform deployments can run the same target
  colocated or provision a separate sandboxed Gym host; custom environment FileSets require the
  sandboxed path. See the
  [Evaluator Gym documentation](https://docs.nvidia.com/nemo-platform/documentation/evaluate-models/agent-eval/gym-runner).

## Next steps

- [`../harbor/`](../harbor/) — the same `AgentEvaluator` seam driven by the Harbor runtime.
- [`../fabric_harness_runtimes.py`](../fabric_harness_runtimes.py) — the Fabric runtime equivalent.
- [`gym/config.py`](../../src/nemo_evaluator_sdk/agent_eval/runtimes/gym/config.py) — `GymRuntimeConfig` field reference.
- [`gym/runtime.py`](../../src/nemo_evaluator_sdk/agent_eval/runtimes/gym/runtime.py) — the attribution and teardown rationale.
- [NeMo Gym documentation](https://github.com/NVIDIA-NeMo/Gym) — authoring environments, agents, and datasets.
