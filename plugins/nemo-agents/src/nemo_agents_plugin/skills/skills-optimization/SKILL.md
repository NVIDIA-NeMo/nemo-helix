---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: skills-optimization
description: >-
  Improve agent skills via the `nemo agents` plugin (evaluate-suite / analyze / optimize-skills). Use when the user wants to improve an agent's skills using a Harbor or NAT eval suite, run a batch of agentic tests, analyze why evals fail, or kick off an automated skill-optimization
  loop. Trigger keywords - optimize skills, evaluate agent suite, analyze eval batch, skills optimizer, harbor evals.
allowed-tools: Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo skills improvement workflow

The `nemo agents` plugin exposes three commands for improving an agent that
already has an eval suite (Harbor `task.toml` or NAT `workflow.yml` tasks):

| Command | Purpose |
|---------|---------|
| `nemo agents evaluate-suite` | Run a directory of containerized eval tasks against the agent |
| `nemo agents analyze` | Cluster failures, surface regressions, generate hypotheses |
| `nemo agents optimize-skills` | Full loop: run evals → analyze → have Claude edit skills → verify → keep or discard |

All three are direct NemoJob submit commands. The spec is supplied via
`--spec-file <path.yml>` (YAML or JSON file) or `--spec '{...}'` (JSON inline).
When both are given, `--spec-file` wins.

## When to recommend each command

- "How are my agent's evals doing?" → `evaluate-suite` (collect data) then `analyze` (interpret it).
- "Why did these evals fail?" / "What's slow?" → `analyze` on an existing batch directory.
- "Improve / optimize / fix my agent" → `optimize-skills` (only after confirming a `.agent-improver.yml` exists or asking the user for the `evals` / `agent` / `skills_path` values).

## Self-referential example: improve NeMo itself

The Platform repo ships a canonical `.agent-improver.yml` at its root. Running
`nemo agents optimize-skills --spec-file .agent-improver.yml` with absolute
path overrides from the repo root improves the skills under `.agents/skills/`
based on the `tests/agentic-use/` Harbor evals. This supplants the older
standalone tools/self_improve/ package.

```bash
export ANTHROPIC_API_KEY='<key>'
export ANTHROPIC_BASE_URL='https://inference-api.nvidia.com'
printf '%s' "$ANTHROPIC_API_KEY" | nemo secrets create anthropic-api-key --from-file -
nemo agents optimize-skills --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic-use" \
  --agent "$(pwd)" \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL"
```

When the user wants to improve **another** agent, they copy
`plugins/nemo-agents/examples/agent-improver.example.yml` into their agent's
repo, retarget the paths, and run the same command.

## Important constraints — surface these proactively

- **The optimize-skills loop spawns `claude --print` as a subprocess.** The
  loop strips `CLAUDE_CODE_*` env markers internally, so it can be invoked
  from inside an active Claude Code session — `evaluate-suite`, `analyze`,
  and `optimize-skills` all work the same way whether you're in CC or not.
  For long unattended runs (full suite, multiple iterations), wrap in
  `tmux` so the user can detach:

  ```bash
  tmux new -s improve
  nemo agents optimize-skills --spec-file .agent-improver.yml \
    --evals "$(pwd)/tests/agentic-use" \
    --agent "$(pwd)" \
    --anthropic-api-key-secret anthropic-api-key \
    --anthropic-base-url "$ANTHROPIC_BASE_URL"
  # detach: Ctrl-B D
  # reattach: tmux attach -t improve
  ```

- **Variance:** the default verdict is single-trial. Cold-cache effects on
  containerized agentic runs commonly produce 20-40% wallclock variance
  trial-to-trial, so single-trial verdicts can commit noise as
  "improvement". For verdicts you can trust, set `repeats: 3` in the YAML.
  Median aggregation + majority-vote pass/fail.

- **The eval directory is immutable.** The loop's post-edit guard reverts
  any change inside the `evals` path. If the user asks to "fix the eval",
  redirect them: the loop improves the **agent**, not the **evals**. Eval
  authoring is a separate workflow.

- **`open_pr` is opt-in.** Default behaviour is "verified diff producer" —
  the loop produces a local branch with a clear diff for human review.
  Recommend `open_pr: true` only when the user explicitly wants automation
  end-to-end. With `open_pr: true`, the loop detects `gh` (GitHub) or `glab`
  (GitLab) and dispatches accordingly.

## Prerequisites — check / surface in the order most likely to fail

1. `claude` CLI on PATH and authenticated (only needed for `optimize-skills`).
2. `docker` daemon running and `harbor` CLI on PATH.
3. `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL` exported.
4. Eval directory exists with at least one task containing `task.toml` or
   `workflow.yml` plus `instruction.md` and `tests/test_outputs.py`.
5. For `optimize-skills`: the directory at `skills_path` exists inside
   `agent`.

The plugin's preflight checks fail fast with actionable error messages, so
reading the first error in any failure is usually enough.

## Reading the output

`optimize-skills` prints a JSON `LoopState` at the end. Key fields:

- `iterations[].status`: `improved`, `regressed`, `neutral`, or `error`
- `iterations[].hypotheses[]`: what the LLM analyzer proposed (root_cause,
  category, proposed_fix, confidence, expected_impact)
- `iterations[].changes_made`: files Claude actually modified
- `iterations[].eval_results_before` / `_after`: agent-time per affected eval
- `iterations[].improvement_pct`: aggregate duration delta
- `iterations[].branch_name`: the worktree's branch (kept locally on
  improvement; discarded on regress/neutral)
- `iterations[].mr_url`: set when `open_pr=True` and PR/MR creation succeeded

`evaluate-suite` writes `report.md`, `report.csv`, `report.json`, and
`baselines.json` to the batch directory. Per-eval trial data lands at
`<batch_dir>/<eval-name>__trials.json` when `repeats > 1`.

## Common command shapes

```bash
# Run a single eval to debug
nemo agents evaluate-suite --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic-use" \
  --agent "$(pwd)" \
  --output "$(pwd)/runs/batch-debug" \
  --filter-glob "auth-authorization-cli" \
  --concurrency 1 \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL"

# Run the full suite with variance reduction — set `repeats: 3` in
# .agent-improver.yml, then:
nemo agents evaluate-suite --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic-use" \
  --agent "$(pwd)" \
  --output "$(pwd)/runs/batch-full" \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL"

# Analyze a previous batch
nemo agents analyze --spec "{
  \"batch\": \"$(pwd)/runs/batch-2026-04-30__09-10-42\",
  \"format\": \"md\",
  \"anthropic_api_key_secret\": \"anthropic-api-key\",
  \"anthropic_base_url\": \"$ANTHROPIC_BASE_URL\"
}"

# Scope to one eval — edit `filter_glob` / `iterations` in the YAML
# (or copy and edit a copy), then:
nemo agents optimize-skills --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic-use" \
  --agent "$(pwd)" \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL"

# Full loop with auto-PR — set `open_pr: true` in the YAML, then:
nemo agents optimize-skills --spec-file .agent-improver.yml \
  --evals "$(pwd)/tests/agentic-use" \
  --agent "$(pwd)" \
  --anthropic-api-key-secret anthropic-api-key \
  --anthropic-base-url "$ANTHROPIC_BASE_URL"
```

## Don't do

- Don't suggest editing files under the `evals` directory. Strategies are
  scoped to write only under `skills_path`; eval files are reverted.
- Don't recommend `optimize-skills` without confirming the agent has skills
  to improve. If the agent is stateless / has no skill files, the loop has
  no writable surface.
