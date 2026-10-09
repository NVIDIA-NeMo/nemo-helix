<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Documentation Notebook E2E Tests

Runs documentation notebooks (from `docs/`) as pytest test cases against a live NHX backend.
Each notebook marked with `@nemo-nb: process` becomes an individual test.

## Quick Start

```bash
# CPU notebooks against Docker quickstart
uv run pytest e2e/notebooks --docker -v

# Use host kernel instead of isolated sandbox (faster iteration)
uv run pytest e2e/notebooks --docker -v --notebook-kernel python3

# GPU-requiring notebooks
uv run pytest e2e/notebooks --docker -v --feature gpu --notebook-language all

# Run a specific notebook
uv run pytest e2e/notebooks --docker -v -k "safe-synthesizer-101"
```

## Backend Options

By default, tests spin up the Docker quickstart automatically. You can also connect to an existing backend:

```bash
# Against an already-running quickstart (e.g. started via `nemo quickstart up`)
uv run pytest e2e/notebooks -v --docker --cluster-url=http://localhost:8080

# Against a deployed Kubernetes cluster
uv run pytest e2e/notebooks -v --kubernetes --cluster-url=http://localhost:8080
```

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--notebook-language` | `python` | Cell types to execute: `python`, `shell`, or `all` |
| `--notebook-kernel` | `sandbox` | Jupyter kernel. `sandbox` creates an isolated venv with only the SDK; any other value (e.g. `python3`) uses that kernel directly |
| `--feature gpu` | — | Include GPU-requiring notebooks (skipped by default) |
| `--cluster-url` | — | Connect to an existing backend instead of starting one |

## Notebook Categories

### CPU Notebooks (default)
Run against a bare CPU quickstart. No special flags needed, but most likely need 1 or more keys exported.

### GPU Notebooks (`--feature gpu`)
Require a GPU-enabled quickstart with model deployment or NIM access:
- `run-inference/tutorials/deploy-models.md`
- `run-inference/tutorials/run-inference.md`
- `audit/tutorials/docker-local-nim.md`
- `safe-synthesizer/tutorials/` (differential-privacy, pii-replacement, safe-synthesizer-101)

### Bifurcated Notebooks
Contain both CLI and Python SDK cells. These automatically produce two test cases
(`[shell]` and `[python]`) regardless of `--notebook-language`:
- `run-inference/tutorials/deploy-models.md`
- `run-inference/tutorials/run-inference.md`

```bash
# Run only the shell variant
uv run pytest e2e/notebooks -v -k "shell"

# Run only the Python variant
uv run pytest e2e/notebooks -v -k "python"
```

## Required Environment Variables

`NHX_BASE_URL` is set automatically. Other variables must be set before running
notebooks that need them:

| Variable | Required by |
|---|---|
| `NVIDIA_API_KEY` | evaluator tutorials, example-applications, run-inference (about, deploy-models), data-designer (quickstart, basics, seeding) |
| `NGC_API_KEY` | guardrails tutorials, audit/docker-local-nim |
| `HF_TOKEN` | evaluator/run-an-evaluation, safe-synthesizer-101, run-inference/deploy-models |
| `NIM_API_KEY` | safe-synthesizer/pii-replacement, safe-synthesizer/safe-synthesizer-101 |
| `OPENAI_API_KEY` | run-inference/deploy-models |

Tests that are missing required env vars are automatically skipped with a clear message.

## Kernel Modes

**Sandbox (default):** Creates an isolated virtualenv with only the public SDK
(`sdk/python/nemo-helix/`) and `ipykernel`. This catches accidental imports of internal
packages and mirrors what end-users have installed. `%%bash` cells installing packages via
`pip`/`uv pip` target the sandbox.

**Host kernel (`--notebook-kernel python3`):** Skips sandbox creation and uses the
specified kernel directly. Faster for local iteration.

## CI Jobs

Defined in `.gitlab/ci/docs-acceptance.gitlab-ci.yml`. All jobs run on `main` (allow_failure)
or can be triggered manually on MRs.

| Job | Runner | Flags | Notes |
|---|---|---|---|
| `e2e-nb` | DinD | `--notebook-language all` | CPU notebooks |
| `e2e-nb-gpu` | `nemollm-a100-1gpu` | `--feature gpu --notebook-language all` | GPU notebooks via standalone dockerd |
| `e2e-nb-kubernetes` | `nemollm-low-conc` | `--notebook-language all` | CPU notebooks on k8s review cluster |
| `e2e-nb-kubernetes-gpu` | `nemollm-low-conc` | `--feature gpu --notebook-language all` | GPU notebooks on k8s review cluster |

Executed notebooks are saved as artifacts at `docs/**/*.executed.ipynb` for debugging.

## Running Skipped Notebooks

Some notebooks carry a `skip-test` marker that excludes them from automated test runs.
The notebook still appears in the docs build but is not discovered by pytest.

**Markdown files** (`.md`): the marker is an HTML comment near the top of the file:
```
<!-- @nemo-nb: skip-test -->
```

**Jupyter notebooks** (`.ipynb`): the marker is a comment in the first code cell:
```python
# @nemo-nb: skip-test
```

To run a skipped notebook locally, remove the marker line, run the test, then restore it
(or leave it removed if the notebook is now ready to be part of the suite):

```bash
# After removing the marker from the file:
uv run pytest e2e/notebooks -v -k "<notebook-name>"
```

> **Note:** The marker is intentional — notebooks are often skipped because they are
> still under development, require infrastructure not available in CI, or are known to be
> flaky. Check the notebook or its git history for context before permanently removing
> the marker.

## Artifacts

After execution, each notebook is saved as `<original-name>.executed.ipynb` alongside the
source file. In CI these are collected as job artifacts (7-day retention) for debugging
failures.
