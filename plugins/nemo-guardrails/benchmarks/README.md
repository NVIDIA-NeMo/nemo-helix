<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Guardrails Plugin Benchmarks

Local harness for benchmarking the `nemo-guardrails` Inference Gateway
middleware against the upstream NeMo Guardrails benchmark suite (mock LLMs +
AIPerf sweep).

The implementation lives in `nemo_guardrails_plugin.benchmarks` (under
`plugins/nemo-guardrails/src/`). The harness does **not** copy benchmark code
from the NeMo Guardrails repository; it expects a local checkout and runs its
benchmark modules with `PYTHONPATH` pointed at that checkout.

## Layout

```text
plugins/nemo-guardrails/benchmarks/
  configs/
    nhx_igw_guardrails_sweep_concurrency.yaml   # AIPerf sweep template
    mock_llm/                                   # in-repo mock LLM env files
  artifacts/                                    # per-run outputs (gitignored)
plugins/nemo-guardrails/src/nemo_guardrails_plugin/benchmarks/
  run.py             # entrypoint: `python -m nemo_guardrails_plugin.benchmarks.run`
  analyze.py         # post-run analysis; checks latencies against baseline values
  paths.py           # filesystem layout
  constants.py       # workspace / VM / provider names
  processes.py       # subprocess supervision (process groups + ExitStack)
  seeding.py         # NHX SDK calls to create the required entities
  aiperf_runner.py   # rewrite AIPerf config + invoke upstream sweep + collect results
  bootstrap.py       # manage the isolated venv that hosts the `aiperf` CLI
  shim.py            # tiny HTTP shim that satisfies AIPerf's `/v1/models` pre-check
```

## Prerequisites

- This repo bootstrapped via `make bootstrap-python` (the harness runs in
  `.venv` and imports the NHX SDK from the workspace).
- A local NeMo Guardrails checkout. By default the harness looks at
  `../NeMo-Guardrails` relative to the NHX repo root.
- `uv` available on `PATH`.
- Ports `8000`, `8001`, `8080`, and `8090` available — unless you opt into
  reusing an already-running local NHX via `--reuse-services`. Port `8090` is
  used by an internal shim that satisfies AIPerf's hard-coded `/v1/models`
  health probe.

The harness has its own dependencies that are declared as
the `bench` extra on `nemo-guardrails-plugin`. The `make benchmark-guardrails`
target installs them automatically via `uv run --extra bench`; they are not
part of the plugin's runtime install.

The upstream `aiperf` CLI itself pins older transitive dependencies. To avoid
downgrading the shared workspace venv, the harness creates an isolated venv at
`plugins/nemo-guardrails/benchmarks/artifacts/venvs/aiperf/` on first run and
reuses it on subsequent runs. CI gets a fresh one each invocation; locally
this caches across runs for fast iteration.

## Run locally

From the NHX repo root:

```bash
make benchmark-guardrails
```

If your NeMo Guardrails checkout is somewhere else:

```bash
NEMO_GUARDRAILS_REPO_ROOT=/path/to/NeMo-Guardrails make benchmark-guardrails
```

To pass through arbitrary harness flags:

```bash
make benchmark-guardrails BENCHMARK_ARGS="--verbose --reuse-services"
```

The default sweep runs concurrency levels:

```text
1, 2, 4, 8, 16, 32, 64
```

With the default 60-second benchmark duration, expect the benchmark to run for ~10 minutes after service bootstrap.

### Monitoring progress

After bootstrap and seeding, the terminal prints `Running aiperf sweep: ...` while
AIPerf runs. AIPerf stdout/stderr is redirected to `logs/aiperf.log`, not the
harness terminal. When the run finishes successfully, the last harness line looks
like:

```text
Sweep summary: 7 run(s), 0 failure(s); per-sweep outputs under ...
```

**Tail the sweep log** (replace `<run-id>` with your `--run-id` or timestamp
directory name):

```bash
tail -f plugins/nemo-guardrails/benchmarks/artifacts/runs/<run-id>/logs/aiperf.log
```

Look for lines like `Run 3/7` and `Run 3 completed successfully`.

**Watch completed sweep directories**:

```bash
ls plugins/nemo-guardrails/benchmarks/artifacts/runs/<run-id>/aiperf_results/*/*/
```

Each finished level appears as `concurrency1/`, `concurrency2/`, etc., with
`process_result.json` and `profile_export_aiperf.csv` inside. A directory with
an empty `profile_export.jsonl` is usually the sweep currently in progress.

**Confirm the process is still running**:

```bash
pgrep -fl "benchmark.aiperf"
```

**Normal log noise during bootstrap**:

- With the `--verbose` flag, `Connection refused` on `:8080` for up to ~1–3 minutes while `nemo services run` starts.
- `409 Conflict` during seeding when resources from a prior run already exist.
- One failed smoke-test attempt (`404` on the VirtualModel) before the IGW route propagates.

## What the harness starts

- The upstream benchmark **mock app LLM** on `http://localhost:8000`,
- The upstream benchmark **mock content-safety LLM** on `http://localhost:8001`,
- Local **NHX services** on `http://localhost:8080`, unless `--reuse-services`.

It then seeds NHX via the SDK with:

- workspace `benchmark`,
- app model provider `benchmark-app-llm`,
- content-safety model provider `benchmark-content-safety-llm`,
- guardrail config `content-safety-local`,
- VirtualModel `benchmark/guardrails-vm` with `nemo-guardrails` attached to
  both request and response middleware.

The benchmark target for inference requests is:

```text
http://localhost:8080/apis/inference-gateway/v2/workspaces/benchmark/openai/-/v1/chat/completions
```

## Useful flags / environment

The harness accepts both CLI flags and environment variables:

| CLI flag                          | Environment variable             | Default                |
|-----------------------------------|----------------------------------|------------------------|
| `--nemo-guardrails-repo-root`     | `NEMO_GUARDRAILS_REPO_ROOT`      | `../NeMo-Guardrails`   |
| `--reuse-services`                | `NHX_BENCHMARK_REUSE_SERVICES=1` | start `nemo services run` |
| `--keep-running`                  | `NHX_BENCHMARK_KEEP_RUNNING=1`   | tear down on exit      |
| `--mock-workers`                  | `NHX_BENCHMARK_MOCK_WORKERS`     | `4`                    |
| `--run-id`                        | _n/a_                            | current timestamp      |

`--keep-running` leaves child processes alive for post-mortem inspection; the
harness logs each child's PID as it starts.

## Outputs

Each run writes artifacts under:

```text
plugins/nemo-guardrails/benchmarks/artifacts/runs/<timestamp>/
  logs/
    mock-app-llm.log
    mock-content-safety-llm.log
    nhx-services.log
    aiperf.log
  generated/
    app_provider.json
    content_safety_provider.json
    virtual_model.json
    content_safety_local_nhx_request.json
    nhx_igw_guardrails_sweep_concurrency.yaml   # runtime AIPerf config
  aiperf_results/<batch>/<timestamp>/<sweep-label>/
    run_metadata.json
    process_result.json
    profile_export*.json                         # written by aiperf
```

## CI

Two jobs in `.github/workflows/ci.yaml`:

- `guardrails-benchmark` — matrix of two parallel jobs, one per variant
  (`with-guardrails`, `without-guardrails`), each on its own NHX instance.
  Uploads per-variant artifacts (`logs/`, `generated/`, `aiperf_results/`).
- `guardrails-benchmark-analyze` — joins the two matrix jobs, downloads both
  artifacts, prints a side-by-side comparison via
  `nemo_guardrails_plugin.benchmarks.analyze`, and runs the baseline check
  (see below). Fails the build on a latency regression beyond tolerance. The
  analyzer is stdlib-only by design, so this job runs on the runner's stock
  `python3` without bootstrapping the uv workspace.

### Baseline and gating

CI compares the run's delta_p50 (with-guardrails minus without-guardrails
p50, in ms) against a checked-in baseline. The baseline lives as
module-level constants in:

```text
plugins/nemo-guardrails/src/nemo_guardrails_plugin/benchmarks/analyze.py
```

Why only delta_p50 (and not absolute with-guardrails p50)? delta_p50
keeps the check focused on the guardrails overhead relative to the
without-guardrails path. This cancels shared local-run noise when both variants
run on the same machine, but CI matrix jobs may land on separate runners, so
the tolerances also account for inter-runner variance.

#### Baseline constants

- `CONCURRENCIES_TO_VALIDATE: list[int]` — concurrency levels to gate on.
  Other levels still appear in the analyzer's output tables, but pass/fail
  is decided only by these.
- `DEFAULT_DELTA_P50_TOLERANCE_MS: int` — default tolerance (in ms) applied
  to every validated concurrency. A check fails only on a regression, when
  `observed - baseline > tolerance`. Faster results pass regardless of how
  far below the baseline they are.
- `DELTA_P50_TOLERANCE_OVERRIDES_MS: dict[int, int]` — per-concurrency
  tolerance overrides (in ms). Levels without an override fall back to the
  default.
- `DELTA_P50_BASELINE_BY_CONCURRENCY: dict[int, int]` — expected delta_p50
  (in ms) per concurrency level. Edit by hand when a real change shifts
  the numbers.

Worked example: at c=16 the override is 200 ms, so runs with observed
delta_p50 = 1190 (diff -200) or 1589 (diff +199 from baseline 1390) pass;
observed delta_p50 = 1591 (diff +201) fails.

Notes on the current values:

- c=16 and c=32 use wider tolerances than the default because their
  absolute delta_p50 is larger. Over time, we can tighten these values
  if latencies in CI produce less variance.
- Any change to mock-LLM latencies, the guardrails config, or the runner
  class invalidates the current baseline values. The benchmark should be
  re-run in CI several times to establish updated baseline values.

#### Running the analyzer locally

```bash
python3 plugins/nemo-guardrails/src/nemo_guardrails_plugin/benchmarks/analyze.py \
    plugins/nemo-guardrails/benchmarks/artifacts/runs/<run-id>
```

Local runs print both tables and the baseline-check table.
CI passes `--strict` to make any out-of-tolerance check fail the job.

#### Updating the baseline

When a real change shifts the numbers (ex. a deliberate middleware change,
a mock-LLM config change, or a runner-class change), edit the constants at
the top of `analyze.py` by hand and reference the PR / CI run that
justifies it in the commit.

## Cleanup

The harness only stops the processes it started. It will not kill unrelated
processes on ports `8000`, `8001`, `8080`, or `8090`.

Across runs, NHX's data dir is reused so subsequent benchmarks start
faster. The harness redirects NHX's writes via the `NHX_DATA_DIR` env var to
a per-checkout directory:

```text
plugins/nemo-guardrails/benchmarks/artifacts/nhx-data
```

This holds NHX's SQLite database, Secrets vault, and other persistent
service state. It is gitignored and isolated from `~/.nhx/`, so the
benchmark will not pollute your normal local NHX setup. Reuse is what
lets repeat runs skip the workspace / provider / VirtualModel creation
work (the harness treats `409 Conflict` as "already exists, carry on").

If a benchmark misbehaves and you suspect stale state (ex. after pulling
a schema change), delete that directory for a fully fresh run:

```bash
rm -rf plugins/nemo-guardrails/benchmarks/artifacts/nhx-data
```

To remove outputs from a specific run (logs, generated configs, AIPerf results):

```bash
rm -rf plugins/nemo-guardrails/benchmarks/artifacts/runs/<run-id>
```

To clear all run outputs:

```bash
rm -rf plugins/nemo-guardrails/benchmarks/artifacts/runs/*
```

In CI this is automatic — every job gets a fresh runner, so `nhx-data`
does not exist.
