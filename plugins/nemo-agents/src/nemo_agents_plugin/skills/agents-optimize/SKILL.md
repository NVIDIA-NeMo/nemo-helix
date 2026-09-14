---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: agents-optimize
description: >-
  Optimizes and accelerates a deployed NeMo agent by suggesting switchyard
  routing splits, skill optimization, prompt/hyperparameter tuning, and
  evaluations against newly available models. Persists snapshots and structured
  suggestions to the NeMo files service for diffing and UI rendering. Use when
  the user asks to optimize, accelerate, tune, route, or reduce the cost of an
  agent or its underlying model. Trigger keywords - agent optimization,
  optimize agent, accelerate agent, smaller model, switchyard router,
  routing split, agents optimize-skills, agents optimize, agents evaluate,
  new model scan, nemotron, frontier model.
---

# NeMo Agent Optimization

Optimize and accelerate a deployed agent by suggesting routing splits, skill
optimization, prompt/hyperparameter tuning, and evaluations against newly
available models. Always anchor proposed changes against an evaluation baseline
so quality regressions are caught before promotion.

## Workflow

1. **Pick an agent.** Run `nemo agents list` and prompt the user to choose one
   to optimize. Use the rest of this skill on that single agent.
2. **Establish a baseline.** Look for prior evaluation results from
   `nemo agents evaluate`. If none exist but the agent has an eval
   dataset+config, run one and save the result. If no eval dataset exists,
   suggest the user create one — recent telemetry can usually be turned into
   a small dataset.
3. **Run the analysis steps below**, emit suggestions, and persist them.
4. **Present suggestions** to the user and let them pick which to apply.
5. **After applying any change**, re-run evaluation and compare to the baseline.
   Surface deltas (per-evaluator average score, wallclock, token cost) and
   confirm before promoting traffic.
6. **Apply via the create-sibling-and-deploy pattern.** Agents v2 has no PATCH
   on `/agents/<name>`, so model swaps and config edits go through:
   create a sibling agent → deploy it → run an eval against the sibling →
   delete old deployments so traffic flows to the new version.

## Analysis steps

### 1. Routing / switchyard split

Inspect `agent.config.llms[*].model_name` from `nemo agents list`. If every
LLM points at a single frontier model and **no** switchyard routing virtual
model is in use, suggest creating a `random_routing` virtual model with an
80% strong / 20% weak split.

- Pick the weak model from `nemo models list --filter.name nemotron` (prefer
  current Nemotron tiers, e.g. `nvidia-nemotron-nano-3b` or
  `nemotron-3-super`-class). Exclude content-safety / topic-control /
  safety-guard / GLiNER models — those aren't general chat replacements.
- **Verify entity names before constructing the VM.** Auto-discovered names
  on the NVIDIA hub sometimes have doubled prefixes (e.g.
  `nvidia-nvidia-nemotron-3-super-v3`, not `nvidia-nemotron-3-super-v3`).
  Always confirm with `nemo -f json models get <candidate>` before referencing
  the name in `--models` or as the `weak`/`strong` model in middleware config.
  A 404 here, caught early, is much cheaper than a failed VM rollout.
- Use **the `inference` skill** for the exact
  `nemo virtual-models create … --request-middleware … nemo-switchyard
  config_type=random_routing …` syntax (`strong_probability: 0.8`).
- **After creating the VM, smoke-test routing before wiring an agent to it.**
  Run ~20 minimal `nemo inference gateway model post v1/chat/completions <vm>`
  calls and tally the response `.model` field. The split should approximate
  the configured probability (e.g. ~16/4 for 0.8/0.2 over 20 calls). If the
  split is 20/0, the middleware isn't actually rewriting the request — stop
  and investigate before deploying the sibling agent. Catches misconfigured
  probabilities, wrong entity IDs in middleware config, and other middleware-
  silently-bypassed failure modes.
- Set `model` on the suggestion to the current source model. Suggested
  follow-up actions: `nemo agents evaluate --agent <agent> --eval-config <yaml>`
  against the routed VM before promoting.

This is pure config inspection — no telemetry download required.

### 2. Skill optimization

If the agent uses skills (its repo has a `--skills-path`, or
`.agent-improver.yml` is present, or its config references skill files) **and**
`nemo agents optimize-skills` has not been run since the last optimizer
snapshot, suggest running it.

Suggested actions:

- `nemo agents optimize-skills --spec-file .agent-improver.yml`
  (or pass an inline JSON spec via `--spec '{...}'` if no config file exists)
- After it returns, apply the resulting skill diff to the agent and redeploy.

See **the `nemo-agent-skills-optimization` skill** for the full optimize-skills loop
(variance, repeats, branch handling, `--open-pr`).

### 3. Prompt + hyperparameter tuning

If `nemo agents optimize` has not been run for this agent since the last
snapshot, suggest running it. The job sweeps prompts and hyperparameters via
`nat optimize` against the agent's eval dataset.

Suggested actions:

- `nemo agents optimize prepare-fileset --source <bundle-dir> --optimize-config <bundle-relative-yaml> --fileset <name>`
- `nemo agents optimize --agent <name> --optimize-config-fileset <workspace/name> --optimize-config <config-path-in-fileset>`
- After it completes, apply the new prompt + hyperparameters to a sibling
  agent, deploy it, and run `nemo agents evaluate` to compare.

### 4. New model scan

Diff `nemo models list --all-pages` against `prevSnapshot.modelNames`. **Never
emit on first run** — there is no prior snapshot to diff against. Emit one
suggestion per new model whose parameter count and family fit the agent's task
profile (skip guardrails / safety / GLiNER models).

Each suggestion sets `model` to the new model name and includes:

- Create a sibling agent pointing at `<new-model>`, then run
  `nemo agents evaluate --agent <sibling> --eval-config <yaml>` to
  compare against the current model on the baseline dataset. There is no
  `--model` override flag on `evaluate`; the model swap goes through the
  sibling agent, or by editing the eval-config YAML.
- `nemo auditor targets create <target> -d '{"model": "<new-model>", "type": "<type>"}'` then
  `nemo auditor audit run --spec '{"config": "default/<config>", "target": "default/<target>"}'`
  to verify the new model is robust against jailbreaks before promotion.

Pure set subtraction — no downloads required.

## Persistence (every run)

Uses fileset `nemo-agent-optimizer`. A downstream web hook also reads/writes
this fileset; **changes here MUST stay backwards compatible** with
`nemo-agents-optimize-sync-web` and the consumer code at
`web/packages/studio/src/routes/AgentOptimizationsRoute/` (kept up to date via
the parity skill above).

| Path | Contents |
|------|----------|
| `optimizer_snapshot.json` | Per-agent snapshot map; one entry per agent ever audited |
| `optimizer_suggestions.jsonl` | One JSON object per line; survives applies |

### Snapshot shape (per-agent, single file)

The snapshot is **one JSON file at the fileset root**, keyed by agent name so
running the skill against agent A never overwrites agent B's snapshot:

```json
{
  "agents": {
    "support-bot": {
      "modelNames": ["..."],
      "agentNames": ["support-bot"],
      "updatedAt": "2026-05-07T12:34:56Z"
    },
    "triage-bot": {
      "modelNames": ["..."],
      "agentNames": ["triage-bot"],
      "updatedAt": "2026-05-06T09:11:02Z"
    }
  }
}
```

The legacy flat shape (`{ modelNames, agentNames, updatedAt }`) is **read-only
backwards-compatible** — if you load a file without the top-level `agents`
key, treat the whole object as the snapshot for the *current* selected agent
and migrate to the keyed shape on the next save. Never write the legacy shape
back out.

### Load → update → save flow

1. Try to load the prior snapshot
   (`nemo files download nemo-agent-optimizer --remote-path optimizer_snapshot.json -o …`).
   - 404 → no snapshot exists yet for the workspace; treat as first run.
   - Snapshot exists but `agents[<selected>]` is missing → first run *for that
     agent*. Do not emit `new_model_scan` suggestions (no prior model list to
     diff against). Other agents' entries are preserved as-is on save.
   - Snapshot exists and `agents[<selected>]` is present → use that entry's
     `modelNames` for the new-model-scan diff.
2. After analysis, **load → update only `agents[<selected>]` → save** the
   whole file. Never overwrite blindly with just the current agent's data —
   that would erase every other agent's snapshot.
3. **Merge** the suggestions JSONL (do not overwrite blindly):
   1. Load the existing file.
   2. Compute fresh suggestions for the selected agent.
   3. Preserve every record where `applied === true` as-is.
   4. Preserve every non-applied record whose `agent` field is **not** the
      currently selected agent — those belong to other agents' runs and must
      not be dropped here. (Workspace-scoped suggestions with no `agent`
      field are owned by the most recent run; replace them.)
   5. Drop any fresh suggestion whose identity tuple `(type, agent, model)`
      matches an applied record.
   6. Replace the selected agent's non-applied records with fresh output.
4. On first run the fileset may not exist — create it with
   `nemo files filesets create nemo-agent-optimizer`.

### JSONL record shape

One JSON object per line. Required fields: `type`, `title`, `detail`. Optional:
`agent`, `model`, `severity`, `suggested_actions`, `apply`, `apply_description`,
`applied`, `applied_at`. Keep this backwards compatible — downstream web tooling
consumes the fileset (see "Persistence" above).

`detail` describes WHY the suggestion was raised. `apply_description` (if
`apply` is set) is a short, concrete description of what executing the `apply`
block would do (e.g. resource names that will be created). When a consumer of
the fileset executes an `apply` block, it is expected to set `applied: true`
and `applied_at: <ISO timestamp>` and re-upload the JSONL so the applied state
survives across reads.

| `type` value | Meaning |
|--------------|---------|
| `model_optimization` | Switchyard routing split or smaller-model fit |
| `skill_optimization` | `nemo agents optimize-skills` is recommended |
| `prompt_optimization` | `nemo agents optimize` is recommended |
| `new_model_scan` | New model appeared since last snapshot |

`model_optimization` is reused for both "use a smaller model" and "add a
routing split" so existing UI tiles render without a frontend change. The
`title`/`detail` text describes which variant it is.

### Optional `apply` block (one-click action)

When set, the `apply` block defines a single-action mutation that a downstream
consumer can execute against the Platform API on the user's behalf. The
contract is **strictly same-origin** to the Platform API host — consumers
re-validate this at request time and reject anything else.

Shape — single step (most common):

```json
"apply": {
  "method": "POST" | "PUT" | "PATCH" | "DELETE",
  "path": "/apis/...",
  "body": { ... }
}
```

Shape — multi-step (array, executed in order, stops on first failure):

```json
"apply": [
  { "method": "POST", "path": "/apis/...", "body": { ... } },
  { "method": "POST", "path": "/apis/...", "body": { ... } }
]
```

Rules — these MUST be followed when emitting an `apply` block:

- `method` must be one of `POST`, `PUT`, `PATCH`, `DELETE`.
- `path` MUST start with `/`. It MUST NOT contain `://`, MUST NOT start with
  `//` (protocol-relative), MUST NOT contain query/fragment, and MUST NOT
  contain control characters.
- The path is resolved against the configured Platform API URL; the frontend
  rejects any spec that resolves to a different origin.
- The path's `workspaces/<ws>/` segment MUST equal the workspace the
  suggestion was emitted for. Cross-workspace mutations are rejected.
- The `(method, pathname)` pair MUST be on the optimizer apply allowlist.
  Currently allowed:
  - `POST /apis/agents/v2/workspaces/<ws>/agents` — create sibling agent
    (`body.name` required)
  - `POST /apis/agents/v2/workspaces/<ws>/deployments` — deploy
    `suggestion.agent` OR a name declared by an earlier `POST /agents` step
    in this apply
  - `POST /apis/agents/v2/workspaces/<ws>/jobs/evaluate` — submit an
    evaluate-agent platform job. `body.spec.agent` must be a platform agent
    ref equal to `suggestion.agent` or a name declared by an earlier
    `POST /agents` step in the same apply. Endpoint URLs and cross-workspace
    refs are rejected.
  - In-place `PATCH /agents/<name>` is **not allowed** — Agents v2 only
    implements POST/GET/LIST/DELETE. Model swaps must go through the
    create-sibling-and-deploy pattern below.
  - Any other path is rejected at apply time. New action types require
    updating both this skill and the frontend allowlist in
    `web/packages/studio/src/routes/AgentOptimizationsRoute/api.ts`.
- `body` must be JSON-serializable. Embed workspace/agent/model identifiers
  literally — do not template at apply time.

If you cannot construct a safe, fully-qualified payload, omit `apply`. The
textual `suggested_actions` remain available as a fallback.

#### Example: routing/smaller-model swap

Multi-step — create the sibling, then deploy it:

```json
{"type":"model_optimization","title":"Consider smaller model for support-bot","detail":"Agent uses 70B; 4B Nemotron is available.","suggested_actions":["nemo agents evaluate ...","switchyard for automatic routing"],"apply":[{"method":"POST","path":"/apis/agents/v2/workspaces/default/agents","body":{"name":"support-bot-nemotron-mini-4b","config":{"llms":{"llm":{"model_name":"nvidia-nemotron-mini-4b"}}}}},{"method":"POST","path":"/apis/agents/v2/workspaces/default/deployments","body":{"agent":"support-bot-nemotron-mini-4b"}}]}
```

Three-step — create sibling, deploy, evaluate so quality is measurable before
promoting:

```json
{"type":"model_optimization","title":"Validate smaller-model swap for support-bot","detail":"Agent uses 70B; 4B Nemotron is available. The third step queues an evaluation against the sibling so quality is measurable before promoting it.","apply_description":"Creates 'support-bot-nemotron-mini-4b', deploys it, and submits an evaluate-agent job using the eval config in fileset 'support-bot-eval'.","suggested_actions":["nemo agents evaluate ...","switchyard for automatic routing"],"apply":[{"method":"POST","path":"/apis/agents/v2/workspaces/default/agents","body":{"name":"support-bot-nemotron-mini-4b","config":{"llms":{"llm":{"model_name":"nvidia-nemotron-mini-4b"}}}}},{"method":"POST","path":"/apis/agents/v2/workspaces/default/deployments","body":{"agent":"support-bot-nemotron-mini-4b"}},{"method":"POST","path":"/apis/agents/v2/workspaces/default/jobs/evaluate","body":{"spec":{"agent":"support-bot-nemotron-mini-4b","eval_config":"eval.yaml","eval_config_fileset":"support-bot-eval","output":"support-bot-eval-out"}}}]}
```

Notes when emitting the eval step:

- `body.spec.agent` MUST be `suggestion.agent` or a name declared by an
  earlier `POST /agents` step in the same apply. Endpoint URLs
  (`http(s)://...`) are rejected.
- `body.spec.eval_config` is interpreted relative to the downloaded fileset
  when `eval_config_fileset` is set.
- `body.spec.workspace` is overwritten by the URL workspace at compile time,
  so it can be omitted.
- Only emit the eval step when you have a concrete eval config + fileset to
  point at — half-configured eval submissions just create a failed job.

## CLI reference (verified)

```bash
# List agents and models
nemo agents list
nemo models list --all-pages              # always --all-pages; default paginates
nemo models list --filter.name nemotron   # find Nemotron candidates

# Optimization commands (see also: nemo-agent-skills-optimization skill).
# The optimize command's --optimize-config value must be relative to the staged fileset root.
nemo agents evaluate --agent <name> --eval-config <yaml>
nemo agents optimize prepare-fileset --source <bundle-dir> --optimize-config <bundle-relative-yaml> --fileset <name>
nemo agents optimize --agent <name> --optimize-config-fileset <workspace/name> --optimize-config <config-path-in-fileset>
nemo agents optimize-skills --spec-file .agent-improver.yml
nemo agents evaluate-suite --spec '{"evals": "<dir>", "agent": "<name>"}'

# Files service
nemo files list <fileset>
nemo files download <fileset> --remote-path <remote> -o <local>
nemo files upload <local> <fileset> --remote-path <remote>
nemo files filesets create <name>
nemo files filesets list

# Auditor (jailbreak robustness check on a candidate model)
nemo auditor targets create <target> -d '{"model": "<new-model>", "type": "<type>"}'
nemo auditor audit run --spec '{"config": "default/<config>", "target": "default/<target>"}'
```

## What requires execution vs. what can be reasoned

**Reason directly from already-fetched data — no extra commands needed:**

- **Routing/smaller-model fit**: compare model name in agent config against
  `nemo models list`. Parameter counts are visible in model names
  (`8b` < `30b` < `70b`). Prefer Nemotron candidates.
- **New model diff**: set subtraction between current model list and prior
  snapshot model list. Pure computation.
- **Skill / prompt optimization gating**: check whether
  `iterations[].timestamp` from a prior `nemo agents optimize-skills` /
  `nemo agents optimize` run is newer than the last snapshot.

**Requires execution:**

- **Establishing a baseline** (`nemo agents evaluate`) when none exists.
- **Comparing post-change quality** — re-run the same evaluation on the
  sibling agent and diff against the baseline before promoting.

## Execution checklist

- Run `nemo agents list`; prompt the user to pick one agent.
- Try to load prior snapshot
  (`nemo files download nemo-agent-optimizer --remote-path optimizer_snapshot.json -o …`).
  404 (or `agents[<selected>]` absent) → first run for this agent, no
  `new_model_scan` suggestions. Other agents' entries in the file are
  preserved on save.
- Look up the latest evaluation result for the selected agent. Run
  `nemo agents evaluate` if no baseline exists and a dataset+config is
  available; otherwise suggest the user create one.
- Fetch agents and models in parallel
  (`nemo agents list`, `nemo models list --all-pages`).
- Reason about routing/smaller-model fit, skill-optimization, prompt-
  optimization, and new-model candidates from this data — no telemetry
  download required.
- Create `nemo-agent-optimizer` fileset if it doesn't exist
  (`nemo files filesets create nemo-agent-optimizer`).
- Update only `agents[<selected>]` in the snapshot map and upload the whole
  file (`nemo files upload <path> nemo-agent-optimizer --remote-path optimizer_snapshot.json`).
  Do not write a flat single-agent shape — that erases every other agent.
- Merge + upload suggestions JSONL
  (`nemo files upload <path> nemo-agent-optimizer --remote-path optimizer_suggestions.jsonl`).
- Summarize key suggestions for the user. Let them pick which to apply.
- After each applied change, re-run evaluation and surface the delta vs.
  baseline before promoting.

## Related skills

- `nemo-agents-secure` — guardrails / PII / data-safety analysis. Use
  alongside this skill when the user wants both a perf-and-safety pass.
- `inference` — exact `nemo virtual-models create … --request-middleware …`
  syntax for switchyard routing/passthrough/translate.
- `nemo-agent-skills-optimization` — full reference for `nemo agents
  optimize-skills` / `evaluate-suite` / `analyze`.
- `nemo-agents-optimize-sync-web` — keeps the web-side optimizer hook in
  parity with this skill's JSONL/apply spec.
