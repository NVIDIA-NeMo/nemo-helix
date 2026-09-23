<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Getting started — agent improvement workflow

This guide walks through using the `nemo agents` plugin to improve an agent
end-to-end: **run eval suites → analyze failures → optimize skills → verify**.

If your agent has Harbor (`task.toml`) or NAT (`workflow.yml`) eval tasks, you
can use this directly. The same workflow improves NeMo itself; the Platform repo
ships `.agent-improver.yml` as an annotated config.

## What you need

| Tool | What for | Required for |
|------|----------|--------------|
| `docker` (daemon running) | Building / running eval container | All commands |
| `harbor` CLI on PATH | Harbor task execution | `evaluate-suite`, `optimize-skills` |
| `claude` CLI authenticated | The coding agent that edits skills | `optimize-skills` only |
| `ANTHROPIC_API_KEY` + `ANTHROPIC_BASE_URL` | Agent-under-test inference + LLM analyzer pass | All commands |

The plugin runs preflight checks before any slow work, so missing prereqs
fail fast with actionable errors.

## What your agent's repo needs

```
my-agent/
├── .agent-improver.yml             # config (copy from examples/)
├── Dockerfile.agentic-base               # container with your agent installed
├── .skills/                        # OR wherever your agent's skills live
│   └── ...
└── tests/
    └── agentic/                    # eval suite — one subdir per task
        └── my-task-1/
            ├── task.toml           # Harbor marker, agent timing, etc.
            ├── instruction.md      # what the agent is told to do
            └── tests/
                └── test_outputs.py # pytest assertions on agent output
```

The eval task layout is unchanged from PR #38 / Harbor convention.

## Quick start

### 1. Copy the example config into your agent's repo

```bash
cp plugins/nemo-agents/examples/agent-improver.example.yml \
   /path/to/my-agent/.agent-improver.yml
```

Edit it to point at your eval directory and skills path. The fields are
documented inline.

### 2. Run an eval suite (sanity check)

```bash
cd /path/to/my-agent
export ANTHROPIC_API_KEY='<key>'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'
printf '%s' "$ANTHROPIC_API_KEY" | nemo secrets create anthropic-api-key --from-file -

nemo agents evaluate-suite --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic" \
  --agent "$(pwd)" \
  --output "$(pwd)/runs/batch-smoke" \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL"
```

Output lands in the directory passed to `--output` with `report.md` / `report.csv`
/ `report.json` and `baselines.json`. With `repeats > 1`, per-eval trial
data is also written to `<batch_dir>/<eval-name>__trials.json`.

### 3. Inspect the results

```bash
BATCH_DIR="$(ls -td "$(pwd)"/runs/batch-* | head -n1)"
nemo agents analyze --spec "{\"batch\": \"$BATCH_DIR\", \"format\": \"md\", \"anthropic_api_key_secret\": \"anthropic-api-key\", \"anthropic_base_url\": \"$ANTHROPIC_BASE_URL\"}"
```

You'll get a markdown report with:
- failing evals (clustered by shared error patterns when possible)
- regressions vs the prior baseline
- LLM hypotheses (root cause, category, proposed fix, confidence)
- mechanical details (slowest evals, tool usage distribution)

For just the mechanical pass (no LLM, no API key needed):

```bash
BATCH_DIR="$(ls -td "$(pwd)"/runs/batch-* | head -n1)"
nemo agents analyze --spec "{\"batch\": \"$BATCH_DIR\", \"mechanical_only\": true}"
```

### 4. Run the optimize-skills loop

```bash
cd /path/to/my-agent
export ANTHROPIC_API_KEY='<key>'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'

nemo agents optimize-skills --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic" \
  --agent "$(pwd)" \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL"
```

The loop strips `CLAUDE_CODE_*` env markers when spawning `claude --print`,
so it works whether or not it's launched from inside an active Claude Code
session. For unattended runs, wrap in `tmux` so you can detach.

What it does, per iteration:

1. **Initial batch** — runs the configured evals (skipped when
   `initial_batch: <existing-dir>` is set in the YAML)
2. **Analyze** — clusters + LLM hypotheses
3. **Pick a hypothesis** — non-overlapping, sorted by confidence, skipping
   ones already tried this loop
4. **Apply** — creates an isolated git worktree, invokes `claude --print`
   with the hypothesis as the prompt, captures the diff
5. **Guard** — reverts any change outside `skills_path` and inside the
   `evals` directory (the eval immutability invariant)
6. **Verify** — re-runs the affected evals (or all evals when
   `full_verification: true`)
7. **Verdict** — pass/fail transitions take priority; then duration delta
   (±5%); then token delta (±10%) as tiebreaker. Status: `improved` /
   `regressed` / `neutral` / `error`
8. **Keep or discard** — improvement keeps the worktree branch (locally,
   or auto-opened as PR/MR when `open_pr: true`); regression / neutral
   discards

The final `loop_state.json` records every iteration's hypothesis, files
changed, before/after metrics, and verdict.

## Common patterns

### Improve NeMo itself

NeMo Helix ships an annotated `.agent-improver.yml` at the repo root. Run from
the repo root and override the checked-in relative paths with absolute host
paths:

```bash
nemo agents optimize-skills --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic-use" \
  --agent "$(pwd)" \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL"
```

This is also the cheapest way to validate the workflow end-to-end on a
real agent.

### Scope to a single eval (debug / fast iteration)

Edit `.agent-improver.yml` (or a copy) to set:

```yaml
filter_glob: my-task-1
iterations: 1
```

Then re-run with `--spec-file <path>`.

### Use an existing batch as the baseline (skip the initial run)

Add to the YAML:

```yaml
initial_batch: /path/to/my-agent/runs/batch-2026-04-30__09-10-42
```

### Run with variance reduction

Single-trial verdicts can commit noise as progress on containerized
agentic runs (cold-cache effects routinely produce >5% wallclock variance
trial-to-trial). Recommended `repeats: 3` in the YAML:

```yaml
repeats: 3
```

Median across trials for duration / tokens / tool-calls; majority vote
for pass/fail with ties going to FAIL (conservative).

### Open a PR/MR automatically on improvement

Set `open_pr: true` in the YAML. The loop detects `gh` (GitHub remote)
or `glab` (GitLab remote) automatically and dispatches accordingly.
With neither installed, falls back to "branch pushed; open manually."

## Troubleshooting

### "Coding agent 'claude' not found on PATH"

Install Claude Code (https://claude.com/claude-code) and authenticate.
Affects `optimize-skills` only; other commands work without it.

### "ANTHROPIC_API_KEY environment variable must be set"

The agent-under-test runs inside the harbor container and needs API
credentials to reach the LLM. Same key drives the LLM analyzer pass.

### "evals_dir must be inside agent_root for v0"

The optimize-skills loop creates a worktree from `agent_root` and re-runs
the evals from inside it; this requires the eval directory to live
within the agent's repo. (For evaluate-suite alone, this constraint
doesn't apply — but optimize-skills does require it.)

### "No eval tasks found in <path>"

Each subdirectory of `--evals` must contain either `task.toml` (Harbor) or
`workflow.yml` (NAT) plus `instruction.md` and `tests/test_outputs.py`. A
top-level directory with no marker files in its children is empty as far
as the runner is concerned.

### Worktree from a previous run blocks a new one

The loop reuses the path `<repo>/../nhx-worktrees/self-improve/iter-N`
and a branch name `self-improve/iter-N`. After a failed run, clean up
manually:

```bash
git worktree remove --force <repo>/../nhx-worktrees/self-improve/iter-1
git branch -D self-improve/iter-1
```

(Lockfile + Ctrl-C cleanup is on the v1 list.)

### LLM analyzer hypothesis quality drops for non-Claude-Code agents

``TraceParser`` is the protocol; ``ClaudeCodeTraceParser`` (session.jsonl
→ typed tool calls, errors, skills) is the only implementation today.
Callers pick a parser at the call site — ``analyze-job`` defaults to
Claude Code and runs without a parser when the batch's agent didn't
emit claude-code-shaped traces. Without a parser, trace-derived cluster
quality drops to near-zero. Mechanical analysis (failing/slowest,
regressions, baselines) is unaffected — those signals come from the
runner, not the trace.

NAT-hosted agents emit ``IntermediateStep`` records to the
``nemo-agent-telemetry`` fileset via the ``nemo_files`` telemetry
exporter today. A parser reading those records is the planned next
implementation, gated on plumbing ``workspace`` / ``agent_name`` /
``run_id`` through ``EvalResult`` first.

## What's where

- **`.agent-improver.yml`** at the repo root — your agent's config
- **`<batch_dir>/`** — per-batch artifacts (results, traces, reports)
- **`baselines.json`** at the repo root — historical eval performance
- **`loop_state.json`** at the repo root — last loop's iteration history
- **`<repo>/../nhx-worktrees/self-improve/iter-N/`** — isolated worktree
  used during a loop iteration
- **`plugins/nemo-agents/examples/agent-improver.example.yml`** —
  annotated template for any agent's repo
- **`plugins/nemo-agents/src/nemo_agents_plugin/skills/nemo-agent-skills-optimization/SKILL.md`** —
  the skill that teaches a Claude session how to drive these commands

## Command verbs

Each improvement command is a direct NemoJob submit command:

```bash
# Submit to the active platform
nemo agents optimize-skills --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic" \
  --agent "$(pwd)" \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL"

# Submit to a configured cluster
nemo agents optimize-skills --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic" \
  --agent "$(pwd)" \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL" \
  --cluster <name>
```

`evaluate-suite` and `analyze` follow the same direct-command pattern.
