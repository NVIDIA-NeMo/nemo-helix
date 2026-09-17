<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Hermes optimize examples

Runnable demos for `nemo agents optimize` using the Hermes Fabric harness.

**Convention:** files named `optimize-*.yaml` are passed to `--optimize-config`.
Agent entity YAML lives under `agents/` and is passed to `--agent-config`.

**Layout:** this directory is a self-contained **optimize bundle**. Every path
inside the `optimize-*.yaml` files (`dataset`, `base_dir`, MCP server `args`)
is relative to *this folder*, not to the repo root. That is what makes the
bundle portable when the platform sees only the files you staged into a fileset.

| Example | What it does | `--optimize-config` | Other |
|---------|--------------|---------------------|-------|
| **Chat-only** | Tunes temperature on a short Q&A agent (no tools) | [`optimize-chatonly.yaml`](optimize-chatonly.yaml) | [`dataset-chatonly.json`](dataset-chatonly.json) |
| **Chat-only + `--agent`** | Same study; agent body from a platform entity | [`optimize-chatonly-via-agent.yaml`](optimize-chatonly-via-agent.yaml) | [`agents/chatonly/agent.yaml`](agents/chatonly/agent.yaml) |
| **MCP** | Tunes temperature / top_p on a phishing agent that calls the email-phishing-analyzer MCP server, scoring accuracy, exactly-one tool call, and verbatim tool input | [`optimize-mcp.yaml`](optimize-mcp.yaml) | [`dataset-mcp.json`](dataset-mcp.json), `PHISHING_MCP_BIN` |
| **MCP, mock analyzer** | The same study against a bundled fixture that replays recorded analyzer results; needs no checkout or analyzer credential | [`optimize-mcp-mock.yaml`](optimize-mcp-mock.yaml) | [`phishing_analyzer_mcp/`](phishing_analyzer_mcp/) |

Official docs: [Optimize Agents](../../../../docs/agents/optimization.mdx).

---

## One-time setup (platform)

### 1. Install the agents CLI

From the **`nemo-platform` repo root**:

```bash
uv sync --package nemo-agents-plugin
source .venv/bin/activate
nemo --help   # should list `agents`
```

### 2. Install the Hermes harness

The lockfile cannot pull `hermes-agent` yet (dependency pin conflict). Install it
into the same venv:

```bash
uv pip install --python .venv/bin/python "hermes-agent==0.19.0" --no-deps
python -c "import hermes_cli; print('ok')"
```

### 3. API key

```bash
export NVIDIA_API_KEY=...   # required for inference-api.nvidia.com
```

The example YAMLs call `https://inference-api.nvidia.com/v1` with full model ids
such as `nvidia/nvidia/nemotron-3-nano-30b-a3b`. Confirm your key can list those
models (`GET /v1/models`).

### 4. Shell env used by every example

```bash
export REPO_ROOT="/path/to/nemo-platform"
export BUNDLE="$REPO_ROOT/plugins/nemo-optimization/examples/hermes-optimize"

export NMP_BASE_URL="${NMP_BASE_URL:-http://localhost:8080}"
# Optional alias used by some CLI paths:
export NEMO_BASE_URL="${NEMO_BASE_URL:-$NMP_BASE_URL}"

# Point Fabric at the platform venv so Hermes adapters resolve. Without this,
# Fabric may pick a system Python and fail with
# `No module named 'nemo_fabric_adapters'`.
export ADAPTER_PYTHON="$REPO_ROOT/.venv/bin/python"
```

### Common bundle rules

- **`prepare-fileset` reads the bundle from your filesystem; `optimize` reads it
  from a fileset.** Paths inside the YAML are bundle-relative, so `cd "$BUNDLE"`
  first when validating examples locally. The platform command requires
  `--optimize-config-fileset` and resolves paths against the downloaded bundle
  root; see [Platform submission](#platform-submission) below.
- Local Hermes output lands in `./artifacts/` under this folder (safe to
  delete). Do **not** stage `artifacts/` into a fileset.
- Re-run the `hermes-agent==0.19.0 --no-deps` install after any fresh
  `uv sync` — sync does not install Hermes and can leave `hermes_cli` missing.

---

## Example 1 — Chat-only

No MCP, no extra checkouts. Good first smoke for optimize.

```bash
source "$REPO_ROOT/.venv/bin/activate"   # if not already
cd "$BUNDLE"

nemo agents optimize prepare-fileset \
  --source "$BUNDLE" \
  --optimize-config optimize-chatonly.yaml \
  --fileset hermes-optimize-chatonly \
  --workspace default

nemo agents optimize \
  --optimize-config-fileset default/hermes-optimize-chatonly \
  --optimize-config optimize-chatonly.yaml \
  --workspace default
```

**Success:** job finishes with `status: completed` and `n_trials: 2`.

Local-only Python run of the same config:

```python
import os
from pathlib import Path

from nemo_optimization.jobs.optimize import OptimizeJob
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.scheduler import NemoJobScheduler

WORKSPACE = "default"
bundle = Path(os.environ["BUNDLE"]).resolve()
os.chdir(bundle)  # the config's dataset / base_dir are relative to the bundle

client = NeMoPlatform(
    base_url=os.environ.get("NMP_BASE_URL", "http://localhost:8080"),
    workspace=WORKSPACE,
)
print(
    NemoJobScheduler().run_local(
        OptimizeJob,
        {"optimize_config": str(bundle / "optimize-chatonly.yaml"), "workspace": WORKSPACE},
        workspace=WORKSPACE,
        sdk=client,
    )
)
```

---

## Example 1b — Chat-only with `--agent`

Same smoke as Example 1, but the agent body is a **platform-managed** entity.
[`optimize-chatonly-via-agent.yaml`](optimize-chatonly-via-agent.yaml) is an
overlay (optimizer + eval only). Optimize resolves `--agent`, translates
`nemo-agents-spec-v1` → Fabric, and merges the overlay.

### 1. Register the agent (once)

Point `--agent-config` at the **slim**
[`agents/chatonly/agent.yaml`](agents/chatonly/agent.yaml) file — not the
parent `hermes-optimize/` directory (that tree includes `artifacts/` and will
fail the fileset size check).

```bash
source "$REPO_ROOT/.venv/bin/activate"

# Optional: retarget models to your platform IGW before create, e.g.
#   model: <your-igw-model-id>
#   base_url: http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1
#   api_key_env: NEMO_AGENTS_IGW_API_KEY
# (Replace host/model with your NMP_BASE_URL and IGW model id; values are
# stored as-is at create time — no ${...} expansion for this path.)
# Defaults in agent.yaml use inference-api (same as optimize-chatonly.yaml).

nemo agents create \
  --name hermes-optimize-chatonly \
  --agent-config "$BUNDLE/agents/chatonly/agent.yaml" \
  --workspace default
```

For a local IGW, also export a key env (any non-empty value is fine if the
gateway does not check it):

```bash
export NEMO_AGENTS_IGW_API_KEY="${NEMO_AGENTS_IGW_API_KEY:-not-used}"
```

### 2. Run optimize against the stored agent

```bash
cd "$BUNDLE"

nemo agents optimize prepare-fileset \
  --source "$BUNDLE" \
  --optimize-config optimize-chatonly-via-agent.yaml \
  --fileset hermes-optimize-chatonly-via-agent \
  --agent hermes-optimize-chatonly \
  --workspace default

nemo agents optimize \
  --optimize-config-fileset default/hermes-optimize-chatonly-via-agent \
  --optimize-config optimize-chatonly-via-agent.yaml \
  --agent hermes-optimize-chatonly \
  --workspace default
```

**Success:** same as Example 1 (`status: completed`, `n_trials: 2`), with log
line `Resolved agent 'hermes-optimize-chatonly' to platform agent ...`.

To replace the stored config after editing `agent.yaml`:

```bash
nemo agents delete hermes-optimize-chatonly --workspace default -y
# then re-run create
```

`delete` prompts for confirmation unless you pass `-y`. Create returns
**HTTP 409** if the name already exists; optimize will keep using the **old**
stored config until you delete + recreate.

---

## Platform submission

`optimize` runs the study **on the platform**, which cannot read your
filesystem. So it takes no host paths at all: you stage the bundle into a
fileset first, and the job references that fileset plus a bundle-relative config
path.

### 1. Stage the bundle

`prepare-fileset` validates before it uploads — it parses the YAML, confirms
there is an Agent under Test, and checks that every path the config references
(dataset, `base_dir`, hook modules, MCP configs) is relative and present under
`--source`. An absolute path that would only exist on your laptop is rejected
here rather than failing minutes into the study.

```bash
nemo agents optimize prepare-fileset \
  --source "$BUNDLE" \
  --optimize-config optimize-chatonly.yaml \
  --fileset hermes-optimize-chatonly \
  --workspace default
```

Add `--dry-run` to validate without uploading, and `--no-check-models` to skip
resolving the config's models against the platform.

> Delete `artifacts/` and `.tmp/` from the bundle before staging — the upload is
> recursive, and previous Hermes executions can leave a large tree behind.

### 2. Launch the study

The command that `prepare-fileset` prints, with `--optimize-config` now relative
to the fileset root:

```bash
nemo agents optimize \
  --optimize-config-fileset default/hermes-optimize-chatonly \
  --optimize-config optimize-chatonly.yaml \
  --workspace default
```

For the overlay example, add `--agent hermes-optimize-chatonly`.

### 3. Watch it

```bash
nemo jobs list --workspace default
nemo jobs logs <job-id> --workspace default
```

Pass `--output <fileset-or-dir>` to `optimize` to have the study's artifacts
(optimized config, trials dataframe, ATIF evidence) published somewhere you can
read them back from.

**Where the study runs:** optimize compiles to the `subprocess` execution
profile when the platform registers one, and otherwise to the `cpu` profile
(docker or `kubernetes_job`, whichever the deployment registered) using the
`nmp-cpu-tasks` image. Either way the fileset is the only input, so both
backends see the same tree. See
[Operator notes](../../../../docs/agents/optimization.mdx) for what each backend
needs installed.

---

## Example 2 — MCP (phishing analyzer)

Same optimize flow, but the agent calls an **MCP email-phishing analyzer** on
each dataset row. The analyzer lives in a **separate** repo
(`email-phishing-analyzer-harnesses`) with its own virtualenv, and
[`optimize-mcp.yaml`](optimize-mcp.yaml) declares it the way any production MCP
server is declared: its console script as the stdio `url`, its credential in
`env`, both expanded from the environment when the optimizer loads the config.
Nothing is configured per task and no hook runs; the study tunes the Hermes
coordinator that calls the tool.

Three evaluators score each trial: the judge compares the final classification
with the dataset label (`average_score`); `tool_call_count` reads the ATIF
trajectory to check the analyzer was called exactly once
(`tool_call_count_matches`); and `tool_argument_matches_input` checks the
`text` the agent passed to the tool equals the task instruction
(`tool_argument_matches_input`), the "copy verbatim" requirement scored directly
rather than inferred from the tool's answer. All three are study objectives in
`optimizer.eval_metrics`.

### Extra setup (once)

1. Clone and sync the agent checkout (adjust the path if yours differs):

   ```bash
   export PHISHING_AGENT_ROOT="${PHISHING_AGENT_ROOT:-$HOME/work/email-phishing-analyzer-harnesses}"
   cd "$PHISHING_AGENT_ROOT"
   uv sync
   ```

2. Point the config at that checkout's MCP binary:

   ```bash
   export PHISHING_MCP_BIN="$PHISHING_AGENT_ROOT/.venv/bin/email-phishing-analyzer-mcp"
   test -x "$PHISHING_MCP_BIN"
   ```

`optimize-mcp.yaml` reads that variable. The analyzer calls
`integrate.api.nvidia.com` (a build.nvidia.com key), while the coordinator and
judge in this config call `inference-api.nvidia.com`, so the analyzer's key is
passed separately as `PHISHING_ANALYZER_API_KEY` and reaches it as
`NVIDIA_API_KEY` through the server's `env`:

```bash
export PHISHING_ANALYZER_API_KEY=...   # build.nvidia.com key for the analyzer
```

The analyzer's model comes from the checkout's `configs/common.yaml`. Its
original default `nvidia/nemotron-3-nano-30b-a3b` was renamed and now returns
410 Gone; the checkout must name `nvidia/nemotron-3.5-lightning-30b-a3b`
(fixed in the harness repo's Fabric MR). The dataset is the agent's full eval
set (5 emails: 3 phishing, 2 benign).

### Variant: the mock analyzer

[`optimize-mcp-mock.yaml`](optimize-mcp-mock.yaml) is the same study with the
tool swapped for a fixture that ships in this bundle
([`phishing_analyzer_mcp/server.py`](phishing_analyzer_mcp/server.py)). The
config spawns it as `python3 phishing_analyzer_mcp/server.py`; the optimizer
makes that bundle-relative path absolute for each trial, so it runs locally and
as a platform job with no checkout and no analyzer credential, and CI exercises
it. Any bundle can ship its own MCP server the same way.

The fixture replays the `analysis` stored on each row of `dataset-mcp.json`,
keyed on the email text: an agent that passes the email verbatim gets the
recorded verdict, one that edits it gets `unknown`. Those values were recorded
from a run of `optimize-mcp.yaml` against the real analyzer (2026-09-16, `nvidia/nemotron-3.5-lightning-30b-a3b`). To add
an email, run `optimize-mcp.yaml` on it once and store the analyzer's result as
the row's `analysis` (the fixture reads `PHISHING_ANALYZER_DATASET` if you keep
the dataset elsewhere). For the mock, `python3` must resolve to the platform
`.venv` (activate it, as above) so the server can import `mcp`.

### Run

```bash
source "$REPO_ROOT/.venv/bin/activate"
cd "$BUNDLE"

# Re-export if this is a new shell:
export PHISHING_AGENT_ROOT="${PHISHING_AGENT_ROOT:-$HOME/work/email-phishing-analyzer-harnesses}"
export PHISHING_MCP_BIN="$PHISHING_AGENT_ROOT/.venv/bin/email-phishing-analyzer-mcp"
export PHISHING_ANALYZER_API_KEY=...   # build.nvidia.com key

nemo agents optimize prepare-fileset \
  --source "$BUNDLE" \
  --optimize-config optimize-mcp.yaml \
  --fileset hermes-optimize-mcp \
  --workspace default

nemo agents optimize \
  --optimize-config-fileset default/hermes-optimize-mcp \
  --optimize-config optimize-mcp.yaml \
  --workspace default
```

**Success:** job finishes with `status: completed`, `n_trials: 4`, the two tool
objectives at `1.0`, and `average_score` near `1.0` when the model follows the
“call the analyzer once” prompt.

**Flakiness:** some models return an empty final message after a successful
analyzer tool call, or re-call the tool. Those trials score low on
`average_score` and `tool_call_count_matches` respectively; that is the signal
the study optimizes over. If every sample fails outright, check
`$BUNDLE/artifacts/.fabric/hermes/runtimes/*/logs/`.

Local-only Python run of the same config:

```python
import os
from pathlib import Path

from nemo_optimization.jobs.optimize import OptimizeJob
from nemo_platform import NeMoPlatform
from nemo_platform_plugin.scheduler import NemoJobScheduler

WORKSPACE = "default"
bundle = Path(os.environ["BUNDLE"]).resolve()
agent_root = Path(
    os.environ.get("PHISHING_AGENT_ROOT", Path.home() / "work/email-phishing-analyzer-harnesses")
)
os.environ.setdefault("PHISHING_MCP_BIN", str(agent_root / ".venv/bin/email-phishing-analyzer-mcp"))
os.chdir(bundle)

client = NeMoPlatform(
    base_url=os.environ.get("NMP_BASE_URL", "http://localhost:8080"),
    workspace=WORKSPACE,
)
print(
    NemoJobScheduler().run_local(
        OptimizeJob,
        {"optimize_config": str(bundle / "optimize-mcp.yaml"), "workspace": WORKSPACE},
        workspace=WORKSPACE,
        sdk=client,
    )
)
```

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `nemo: command not found` | `source .venv/bin/activate` after `uv sync --package nemo-agents-plugin` |
| `No module named hermes_cli` | Re-run the `hermes-agent==0.19.0 --no-deps` install (needed after every fresh `uv sync`) |
| `No module named 'nemo_fabric_adapters'` | `export ADAPTER_PYTHON="$REPO_ROOT/.venv/bin/python"` |
| Mock MCP server fails to start / `No module named mcp` | `python3` must be the platform `.venv` interpreter: `source .venv/bin/activate` before running |
| Every tool call `analyzer request failed` | `PHISHING_ANALYZER_API_KEY` is unset or not a build.nvidia.com key, or the harness checkout's `configs/common.yaml` still names the retired `nvidia/nemotron-3-nano-30b-a3b` (see Example 2 setup) |
| LLM 401 | Confirm `NVIDIA_API_KEY` works on inference-api |
| Dataset / config file not found | `cd "$BUNDLE"` — paths in the YAML are relative to the bundle, not the repo root |
| `optimize` rejected with `optimize_config_fileset is required` | Stage the bundle with `prepare-fileset`, then pass the ref it prints |
| `prepare-fileset` reports an absolute path | Move the file into `--source` and make the YAML entry relative to the bundle root |
| `optimize` fails with `... was not found in fileset` | `--optimize-config` must be relative to the fileset root (e.g. `optimize-chatonly.yaml`), not an absolute path |
| `No 'subprocess' or 'cpu' execution profile named 'default'` | The platform registered neither backend under that profile; check `nemo jobs list-execution-profiles` with your operator |
| Agent create fails on fileset size / too many files | Pass `--agent-config` to `agents/chatonly/agent.yaml` (slim dir), not the parent examples folder |
| `delete` hangs / `Aborted!` | Pass `-y` (`nemo agents delete NAME -y`) |
| Create `409 Conflict` / stale models | Delete with `-y`, then create again; optimize always uses the **stored** agent config |
| Optional `--agent ...` rejected for `http://` / `file://` | Pass a workspace agent name (e.g. `hermes-optimize-chatonly`), or omit `--agent` and use `--optimize-config` only |
| MCP: many samples `trial_status: failed` / `no completed trials` | Inspect `artifacts/.fabric/hermes/runtimes/*/logs/`. Empty finals or repeat tool calls do not fail a trial, they score low on `average_score` / `tool_call_count_matches`; a hard failure usually means `PHISHING_MCP_BIN` is unset, the analyzer's model is retired, or `max_turns` < 4 |
| Judge / best scores look like `4.5` not `~1.0` | `tunable_rag_evaluator` with `default_scoring` can sum component scores; compare trials relative to each other |

Trajectory capture (`capture_trajectory`) is off in the chat-only YAMLs so you
do not need the Relay gateway for a first smoke. `optimize-mcp.yaml` turns it on
because `tool_call_count` reads the ATIF trajectory, so that example needs the
`nemo-relay` gateway on `PATH` (`script/dev-install-fabric.sh`).
