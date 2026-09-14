<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Agentic-use Evals for NeMo Platform (AUT + NAT)

This directory contains agentic benchmark tasks for evaluating a NeMo platform
agent-under-test (AUT).  The current orchestration path runs task instructions
through the AUT (via `nemo agents invoke`) and verifies success with existing
`pytest` assertions in each task's `tests/test_outputs.py`.

## Overview

Our goal is to evaluate whether the AUT can successfully complete platform tasks.
Each test provides:
- A task description (what to do)
- An environment (where to do it)
- A verification script (how to check if it worked)

In practice this translates to:
1. a markdown task description (`instruction.md`),
2. a Docker environment that starts NeMo Platform services,
3. a Python verifier (`tests/test_outputs.py`) that checks API state.

This benchmark currently uses `tests/agentic-use/nat_runner.py` in AUT mode.
`nat run` task-local workflows remain supported as a fallback (`--agent-backend workflow`),
but are no longer the default evaluation target.

Harbor re-integration is intentionally deferred until NAT/Harbor compatibility
work lands.

## Run -> Gate -> Optimize

```bash
# 0) Initialize plugin-backed agents CLI commands (required once per env)
uv pip install -e packages/nemo_platform_plugin -e plugins/nemo-agents
uv run _nemo agents evaluate --help

# 1) Run correctness checks with task-specific pytest verifiers
#    nat_runner.py is the canonical execution path: it runs each task's
#    pytest verifier, captures token usage, and writes nat-jobs/<task>/result.json.
#    `nemo agents evaluate` is diagnostic-only (LLM-judge); it is *not*
#    consumed by the canonical gate below — see step 3 notes.
python tests/agentic-use/nat_runner.py --all --agent-backend aut \
  --aut-agent-name <your-agent> --aut-agent-config <path-to-config.yml>

# 2) Deterministic benchmark gate (canonical)
#    quality: verifier pass-rate
#    cost: token totals (runtime is tie-breaker when token totals tie)
python tests/agentic-use/nat_runner.py \
  --manifest manifests/baseline_supported.txt \
  --agent-backend aut \
  --aut-agent-name <your-agent> \
  --aut-agent-config <path-to-config.yml> \
  --jobs-dir "nat-jobs/$(date -u +%Y%m%dT%H%M%SZ)-baseline-supported"

python tests/agentic-use/passrate_token_policy_gate.py \
  --jobs-dir nat-jobs \
  --manifest manifests/baseline_supported.txt \
  --min-pass-rate 1.0 \
  --require-token-metrics \
  --output eval-out/deterministic-gate-summary.json

# 2b) Candidate-vs-baseline composite policy check
#     hard constraint: pass-rate >= baseline - tolerance
#     optimization target: minimize tokens
#     secondary tie-breaker: runtime (only when token totals tie, which is rare)
python tests/agentic-use/passrate_token_policy_gate.py \
  --jobs-dir nat-jobs \
  --manifest manifests/baseline_supported.txt \
  --baseline-summary eval-out/baseline-summary.json \
  --max-pass-rate-drop 0.0 \
  --max-token-regression-pct 0.0 \
  --max-runtime-regression-pct 0.0 \
  --require-token-metrics \
  --output eval-out/candidate-summary.json

# Optional: include every rerun instead of latest result per task
python tests/agentic-use/passrate_token_policy_gate.py \
  --jobs-dir nat-jobs \
  --manifest manifests/baseline_supported.txt \
  --no-latest-per-task

# 3) Optimize AUT parameters
nemo agents optimize prepare-fileset \
  --source tests/agentic-use \
  --optimize-config aut-optimize.yml \
  --fileset agentic-use-aut-optimize \
  --agent <your-agent>
nemo agents optimize \
  --strategy nat \
  --optimize-config-fileset default/agentic-use-aut-optimize \
  --optimize-config aut-optimize.yml \
  --agent <your-agent>
```

`nemo agents evaluate` with LLM-as-judge can still be run for diagnostic signal,
but it is no longer the canonical quality gate for agent performance or
benchmark numbers.

Optional diagnostic-only eval (not used for promotion decisions). The
`_nemo` binary is the pre-vendor entry point for the `nemo_platform_ext`
CLI (see `packages/nemo_platform_ext/pyproject.toml`); we use it here so
the example matches the binary that's actually wired up in dev. The
explicit `--base-url` keeps the run pointed at your local platform even
if your shell exports a remote `NMP_BASE_URL`:
```bash
uv run _nemo --base-url http://localhost:18080 agents evaluate --spec '{
  "agent": "<your-agent>",
  "eval_config": "tests/agentic-use/aut-eval-diagnostic.yml",
  "output": "./eval-out-diagnostic",
  "workspace": "default"
}'
```

Notes:
- `nat_runner.py` defaults to local platform URL `http://localhost:8080` and ignores `NMP_BASE_URL` from your shell unless you explicitly pass `--nmp-base-url`.
- `_nemo agents evaluate` still reads shell environment when `--base-url` is omitted, so if your shell exports a remote `NMP_BASE_URL` the run will hit the wrong cluster — always pass `--base-url http://localhost:18080` when running diagnostics locally.
- `passrate_token_policy_gate.py` is intended for fresh, comparable artifacts (same manifest/runner generation). Mixed old and new `nat-jobs` outputs can report missing runtime fields.
- Runtime tie-breaker is conditional by design and only applies when baseline and candidate token totals are exactly tied.
- For faster local dev loops, prefer `--no-aut-seed-providers` after provider setup is already known-good. This skips per-task provider bootstrap/wait and significantly reduces runtime.

### Direct agent backend auth

`nat_runner.py` can also run direct container-backed agent backends for smoke
testing and cross-agent comparisons:

```bash
python tests/agentic-use/nat_runner.py workspace-basic-cli-easy \
  --agent-backend codex \
  --agent-model gpt-5.1

python tests/agentic-use/nat_runner.py workspace-basic-cli-easy \
  --agent-backend cursor-agent \
  --agent-model <cursor-model>
```

Codex auth:
- Preferred for CI: set `OPENAI_API_KEY` in the runner environment.
- Local development: run `codex login` on the host once, then opt in to using
  that session by passing `--codex-auth-json ~/.codex/auth.json`. The file is
  mounted read-only and copied into a fresh, task-local `CODEX_HOME` for the
  Codex invocation. Host `config.toml`, MCP servers, and project agent
  instructions are not copied into the task container; auth is the only
  optional host Codex state.

Cursor Agent auth:
- Set `CURSOR_API_KEY` in the runner environment, for example
  `export CURSOR_API_KEY='crsr_...'`.
- The container backend does not mount host Cursor session state; use an API
  key for reproducible local and CI runs.

Claude Code auth:
- Set `ANTHROPIC_API_KEY` in the runner environment. For NVIDIA-hosted Claude
  access, set `ANTHROPIC_BASE_URL=https://inference-api.nvidia.com` and pass a
  concrete allowed model such as `--agent-model aws/anthropic/bedrock-claude-sonnet-4-6`.
- Smoke the model outside the benchmark before a slow matrix run:
  `claude -p 'Reply with OK only.' --model aws/anthropic/bedrock-claude-sonnet-4-6 --output-format json`.
  Claude Code's default model may not be allowed by NVIDIA inference keys.

Placeholder values such as `null`, `none`, or an empty string are treated as
unset by the runner.

### Agent matrix benchmark runbook

Use `agent_matrix_benchmark.py` when comparing direct coding agents over the
same task set. It writes `benchmark_summary.json`, `benchmark_summary.md`, and
`benchmark_report.html` under the selected run directory.

Known-good local flow:

```bash
# Build or refresh the base and task images first.
docker build -f Dockerfile.agentic-base -t nmp-agentic-base:latest .

python tests/agentic-use/nat_runner.py \
  --manifest manifests/evaluator_agent_benchmark_mvp.txt \
  --agent-backend codex \
  --build-only

# Smoke Claude Code against the NVIDIA endpoint before a full matrix run.
export ANTHROPIC_BASE_URL=https://inference-api.nvidia.com
claude -p 'Reply with OK only.' \
  --model aws/anthropic/bedrock-claude-sonnet-4-6 \
  --output-format json

# Run Codex, Cursor Agent, and Claude Code concurrently, one task at a time per agent.
python tests/agentic-use/agent_matrix_benchmark.py \
  --manifest manifests/evaluator_agent_benchmark_mvp.txt \
  --agent codex:gpt-5.5 \
  --agent cursor-agent \
  --agent claude-code:aws/anthropic/bedrock-claude-sonnet-4-6 \
  --codex-auth-json ~/.codex/auth.json \
  --jobs-dir nat-jobs/agent-matrix \
  --run-id pr470-three-agent \
  --skip-build \
  --allow-dirty \
  --parallel-candidates 3
```

Notes:
- `--parallel-candidates` parallelizes across candidates only. Each candidate's
  `nat_runner.py` invocation still runs the selected task list serially.
- Values greater than `1` require `--skip-build`; build the selected task images
  before the parallel matrix run to avoid Docker tag races.
- For Codex, the runner copies only auth into a fresh container-local
  `CODEX_HOME`; host config and MCP servers are not reproduced. Codex receives
  the same task instruction as the other direct agents, runs from `/app`, and
  writes task files directly under `/app/workspace`.
- For Claude Code with NVIDIA inference, always pass a concrete allowed model.
  The default Claude Code model may not be enabled for a given NVIDIA key.
- Open the generated HTML report at
  `nat-jobs/agent-matrix/<run-id>/benchmark_report.html`.

### Task metric authoring direction

Task-local verification should prefer Evaluator SDK `Metric` implementations.
Keep reusable mechanics in `shared/evaluator_agent_eval/` helpers and keep
task-specific files focused on task semantics.

Task-local diagnostics can be emitted as additional numeric metric scores, but
richer structured context should be attached via `MetricResult.diagnostics`
(`list[MetricDiagnostic]`: required `message`, optional `details`). See
`DiffDiagnosticExactMatchMetric` in `packages/nemo_evaluator_sdk/examples/examples.py`
for the recommended pattern.

## Planned follow-ups

The current runner/gate path is stable for deterministic benchmarking. These
refactors are intentionally tracked as follow-up work to keep this PR scoped;
they're cheap to do later and the cost-of-deferral is low because these
benchmark utilities are exploratory and likely to be replaced wholesale once
NAT exposes a structured per-response token trace:

- extract shared manifest parsing/resolution utility used by runner, dataset builder, and gate (`_read_manifest` lives in three files today)
- collapse the wrapper-side `_extract_usage` and runner-side `_extract_usage_metrics` once NAT surfaces token usage in agent-trace events (currently the wrapper extracts from LangChain message metadata and the runner reads from agent log payloads — different sources, similar shape)
- split `runtime_sec` into `build_sec`/`agent_sec`/`verify_sec` (keep total for compatibility)
- move inline AUT invoke heredoc Python into a dedicated helper module for readability/testability
- set up a CI signal for the agentic-use benchmark **once the gate contract is stable** (post-Stage-4.5 IGW routing). Wire as a `.github/workflows/` workflow — `.gitlab/ci/tot-e2e.gitlab-ci.yml` was deliberately not restored after upstream deletion in #253; the GitHub Actions surface is where new CI lives.

Rapid iteration example:
```bash
python tests/agentic-use/nat_runner.py \
  workspace-basic-mcp workspace-basic-cli-easy secrets-crud-cli-easy \
  --agent-backend aut \
  --aut-agent-name <your-agent> \
  --aut-agent-config <path-to-config.yml> \
  --no-aut-seed-providers
```

## Inference Gateway Routing

AUT LLM traffic routes through the platform Inference Gateway (IGW). The
agent config is rewritten at runtime by `_prepare_aut_config_for_runtime`,
which calls the same `inject_gateway_url()` function used by production
`nemo agents deployments create`. This sets `base_url` to the IGW
OpenAI-compatible endpoint and keeps `model_name` in entity form (dashes,
e.g. `aws-anthropic-claude-opus-4-5`). IGW resolves entity names to served
model names internally and retrieves upstream credentials from the secrets
service.

Provider seeding is declarative: `providers.yaml` describes the inference
providers and their secrets; `seed_providers.py` reads the manifest and
creates them via the NeMo SDK. The `--aut-seed-providers` flag (default:
enabled) triggers seeding inside the container before the AUT is deployed.

**Baseline comparison caveat:** results produced with IGW routing are not
directly comparable to earlier results that bypassed IGW. The `routing_mode`
field in `result.json` provenance (value: `igw`) distinguishes the two.
Compare only results with matching `routing_mode`.

## Task Manifests

Stage 1/2 benchmark execution is manifest-driven:

- `manifests/baseline_supported.txt`: curated AUT baseline tasks for regular checkpoints.
- `manifests/nightly_extended.txt`: larger non-compose nightly coverage.
- `manifests/gpu_or_compose_only.txt`: compose/GPU-oriented tasks kept out of baseline/nightly CPU-only gates.

These manifest files are consumed by CI and can also be reused locally:

```bash
# Build an eval dataset for baseline-supported tasks only
python tests/agentic-use/build_aut_eval_dataset.py \
  --manifest manifests/baseline_supported.txt
```

By default this generates a baseline-supported dataset (`aut-eval-data.json`, 10 rows in the current suite). If you want broader optimization/eval coverage, build from a larger manifest (for example, nightly).

The generated eval rows now include immutable benchmark contract metadata
(`instruction.md`, `task.toml`, verifier file, setup scripts, and runtime
requirements) so dataset artifacts capture task assumptions explicitly.

## Plugin initialization preflight

`agents evaluate` and `agents optimize` are provided by the
`plugins/nemo-agents` plugin. If `nemo/_nemo agents ...` is missing:

```bash
uv pip install -e packages/nemo_platform_plugin -e plugins/nemo-agents
uv run _nemo agents evaluate --help
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic-use AUT Test Flow                     │
└─────────────────────────────────────────────────────────────────┘

1. BUILD PHASE
   ┌──────────────────┐
   │ Dockerfile.agentic-base│ (repo root, required to package NeMo Platform)
   │ - Install deps   │
   │ - Setup NeMo Platform API  │
   │ - Install NAT    │
   │ - Install agents │
   └────────┬─────────┘
            │ docker build
            ▼
   ┌──────────────────┐
   │ nmp-agentic-base:latest│ (base image)
   └────────┬─────────┘
            │
            │ Referenced by
            ▼
   ┌──────────────────┐
   │ environment/     │
   │   Dockerfile     │ (in each test)
   │ FROM nmp-agentic-base  │
   └────────┬─────────┘
            │ Harbor builds
            ▼
   ┌──────────────────┐
   │  Test Container  │
   │ - NeMo Platform API ready  │
   │ - MCP configured │
   │ - AUT + CLI      │
   └────────┬─────────┘

2. AGENT PHASE (AUT mode, default)
            │
            ▼
   ┌──────────────────┐
   │ instruction.md   │ → AUT reads task
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  Nemo AUT Agent  │ → Uses Nemo tools/skills
   │  + Gateway/SDK   │ → Calls NeMo Platform API
   │  + NeMo Platform API       │ → Creates resources
   └────────┬─────────┘

3. VERIFICATION PHASE
            │
            ▼
   ┌──────────────────┐
   │ tests/test.sh    │ → Runs pytest
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │tests/test_*.py   │ → Checks API state
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │ Reward: 0.0/1.0  │ → Pass/Fail
   └──────────────────┘
```

## Directory Structure

```
tests/agentic-use/
├── README.md                          # This file
├── Dockerfile.agentic-base (at repo root)   # Base image for all tests
├── shared/
│   └── verify-tests.sh                # Shared test runner (sourced by test.sh)
└── workspace-basic-mcp/               # Example test
    ├── task.toml                      # Harbor task configuration
    ├── instruction.md                 # Task prompt for Claude Code
    ├── README.md                      # Test-specific documentation
    ├── environment/
    │   └── Dockerfile                 # References base image
    └── tests/
        ├── test.sh                    # Sources shared/verify-tests.sh
        └── test_outputs.py            # Pytest verification (required)

```

## Component Details

### 1. Base Image: `Dockerfile.agentic-base` (Repo Root)

**Location**: `Dockerfile.agentic-base` (at repository root)

**Purpose**: Creates a reusable base image with everything needed for Harbor tests.

**Contains**:
- Ubuntu 24.04 with Python 3.11
- NeMo Platform code and dependencies (via `uv sync`)
- NeMo Platform API server (auto-starts via ENTRYPOINT with 3-check health validation)
- MCP server configuration (`.mcp.json`)
- Non-root `harbor` user (UID 1001) for Claude Code
- Claude Code CLI with wrapper script that auto-adds `--dangerously-skip-permissions`
- Required directories: `/app`, `/logs`, `/installed-agent`

**Build once, use for all tests**:
```bash
docker build -f Dockerfile.agentic-base -t nmp-agentic-base:latest .
```

**Why at repo root?**
- Needs access to entire NeMo Platform codebase (`COPY . /app`)
- Provides proper build context for `uv sync`

### 2. Environment: `environment/Dockerfile` (Per Test)

**Location**: `tests/agentic-use/<test-name>/environment/Dockerfile`

**Purpose**: Harbor requires each test to have an environment Dockerfile. We use a wrapper that references the base image.

**Contents**:
```dockerfile
FROM nmp-agentic-base:latest
# All configuration inherited from base image
```

**Why needed?**
- Harbor looks for `environment/Dockerfile` or `environment/docker-compose.yaml`
- We use Dockerfile approach to inherit everything from the base image
- Keeps test-specific configuration minimal

### 3. Task Configuration: `task.toml`

**Location**: `tests/agentic-use/<test-name>/task.toml`

**Purpose**: Configures Harbor test parameters.

**Key sections**:
```toml
[metadata]
difficulty = "easy"
category = "mcp-workspace"
tags = ["workspace", "crud", "mcp"]

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
allow_internet = true
# Note: No docker_image field - Harbor builds from environment/Dockerfile

[verifier]
timeout_sec = 60.0
```

**Important**:
- Don't specify `docker_image` - let Harbor build from environment/Dockerfile
- The wrapper script in Dockerfile.agentic-base auto-injects `--dangerously-skip-permissions`

### 4. Task Prompt: `instruction.md`

**Location**: `tests/agentic-use/<test-name>/instruction.md`

**Purpose**: Describes the task for Claude Code to complete.

**Example**:
```markdown
# Task: Create and Verify NeMo Platform Workspace

Your goal is to complete the following workspace operations using the NeMo Platform MCP tools:

1. Create a new workspace with ID: `harbor-test-workspace`
2. Verify that the workspace was successfully created

## Available Tools

You have access to MCP tools for workspace management:
- `create_workspace` - Create a new workspace
- `list_workspaces` - List all workspaces

## Success Criteria

The test passes when a workspace named `harbor-test-workspace` exists in the system.
```

**Guidelines**:
- Be clear and specific about what needs to be done
- List available MCP tools
- State success criteria explicitly
- Keep it concise (Claude Code has context limits)

### 5. Verification: `tests/` Directory

**Location**: `tests/agentic-use/<test-name>/tests/`

**Contains two files**:

#### `tests/test.sh` - Test Runner
This is required by Harbor and emits either 1 (pass) or 0 (fail). We use a shared test runner to avoid duplication:

```bash
#!/bin/bash
source /app/tests/agentic-use/shared/verify-tests.sh
```

The shared script at `shared/verify-tests.sh` runs pytest on `test_outputs.py` and writes the reward.

#### `tests/test_outputs.py` - Verification Logic (Required)
**This file MUST exist** when using the shared test runner. It contains pytest tests that verify the agent completed the task correctly (typically by querying NeMo Platform API).

**How it works**:
1. Harbor copies `tests/` directory contents to `/tests/` in the container
2. After agent completes, `nat_runner.py` executes the verifier phase (`pytest` against `/tests/test_outputs.py`)
3. The shared script runs pytest on `/tests/test_outputs.py`
4. Pytest checks if the agent accomplished the task (by querying NeMo Platform API)
5. Result written to `/logs/verifier/reward.txt` (1 = pass, 0 = fail)

## Test Execution

### Prerequisites
```bash
# Required for NAT/AUT LLM calls
export NVIDIA_API_KEY='nvapi-...'
```

### Running Tests

```bash
# Step 1: Build base image (do this once, or when Dockerfile.agentic-base changes)
docker build -f Dockerfile.agentic-base -t nmp-agentic-base:latest .

# Step 2: Run a specific task against AUT
python tests/agentic-use/nat_runner.py \
  workspace-basic-mcp \
  --agent-backend aut \
  --aut-agent-name <your-agent> \
  --aut-agent-config <path-to-config.yml>

# Step 3: View results
# Results are written to nat-jobs/<timestamp>-<task>/result.json
```

## Debugging

### View agent logs
```bash
# Find most recent run
ls -lt jobs/ | head -5

# View AUT invoke logs
cat nat-jobs/<timestamp>-<task>/agent/nat_agent.log

# View test output
cat nat-jobs/<timestamp>-<task>/verifier/test-stdout.txt
```

### Test API manually
```bash
# Run container interactively
docker run -it --rm -p 8000:8000 nmp-agentic-base:latest

# In another terminal:
curl http://localhost:8000/health
curl http://localhost:8000/v2/workspaces

# Test MCP server
docker exec -it $(docker ps -q --filter ancestor=nmp-agentic-base:latest) \
  /app/.venv/bin/nemo-mcp --base-url http://localhost:8000
```

## Creating New Tests

### 1. Copy the template
```bash
cp -r tests/agentic-use/example-test-template \
      tests/agentic-use/your-new-test
```

### 2. Modify test files
- **task.toml**: Update metadata (category, tags, difficulty)
- **instruction.md**: Write new task description
- **tests/test_outputs.py**: Write verification logic for your task
- **environment/Dockerfile**: Usually no changes needed (just `FROM nmp-agentic-base:latest`)

### 3. Test it
```bash
# Rebuild base if Dockerfile.agentic-base changed
docker build -f Dockerfile.agentic-base -t nmp-agentic-base:latest .

# Run your test
python tests/agentic-use/nat_runner.py your-new-test \
  --agent-backend aut \
  --aut-agent-name <your-agent> \
  --aut-agent-config <path-to-config.yml>
```

## Key Principles

1. **AUT is the primary target**: benchmark the deployed platform agent, not task-local workflow agents
2. **Thin environments**: each task's `environment/Dockerfile` should remain minimal
3. **API-based verification**: tests query NeMo Platform API to verify AUT actions
4. **Clear success criteria**: `instruction.md` should unambiguously state expected outcomes
5. **Baseline metrics matter**: track pass/fail and token/latency signals for optimization loops

### Backend capability notes

- `aut` backend is the canonical benchmark path and should be used for coverage across platform tasks.
- `workflow` backend is currently a migration/debug fallback. Today, `nemo-mcp` exposes workspace tools only, so many non-workspace tasks are expected to fail under workflow mode.
- When using workflow mode, pass `--model` (or `NAT_MODEL`) to override the task default model without editing task files.

## Troubleshooting

### "Connection refused" errors
- API server may not be ready yet
- Base image uses 3 consecutive health checks to ensure stability
- If still seeing issues, increase health check timeout in Dockerfile.agentic-base

### AUT deployment/invocation errors
- Ensure the AUT exists (or pass `--aut-agent-config` so runner can create it)
- Verify `NVIDIA_API_KEY` is set for model calls
- Check AUT logs in `nat-jobs/<timestamp>-<task>/agent/nat_agent.log`

### Test gets 0.0 reward but agent seems to work
- Check verifier test logic in `tests/test_outputs.py`
- Verify API endpoint URLs are correct
- Look at verifier output: `nat-jobs/<timestamp>-<task>/verifier/test-stdout.txt`

### Docker image not found
- Build base image: `docker build -f Dockerfile.agentic-base -t nmp-agentic-base:latest .`
- Verify image exists: `docker images | grep nmp-agentic-base`
- Run build and test together to avoid image disappearing

## References

- [NeMo Agent Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) - NAT runtime/eval/optimization
- [MCP Protocol](https://github.com/modelcontextprotocol/specification) - Model Context Protocol specification
- [Dev Journal](../../architecture/devjournal/3294-devjournal-harbor-experiment.md) - Detailed implementation notes

## Examples

See `workspace-basic-mcp/` for a complete working example that:
- Creates a workspace using MCP tools
- Verifies it exists via API query
- Achieves 1.0 reward consistently
- Demonstrates all components working together
