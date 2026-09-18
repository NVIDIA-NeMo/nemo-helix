---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-skill-selection
description: Top-level skill selector for ambiguous tasks involving NeMo Platform (NVIDIA's agent platform). Picks the right downstream skill for setup, design, specification, agent configuration, build, deployment, testing, observability, status, teardown, evaluation, optimization, security, or model customization. Use when the user needs help deciding where a NeMo Platform task should start.
triggers:
  - build an agent
  - create an agent
  - deploy an agent
  - set up nemo
  - install nemo
  - try nemo
  - improve my agent
  - help me with nemo
  - nemo platform
  - shut down nemo
  - tear down nemo
  - what is running on nemo
  - help me ship an agent
  - fine-tune embedding
  - retrieval recipe
  - domain retrieval
not-for:
  - setup (use to verify install or to be told how to run the CLI install)
  - nemo-build-agent (use for the actual scaffold/deploy flow)
  - nemo-explore (use to reason about agent design)
  - superpowers:brainstorming (use for design work unrelated to NeMo Platform)
  - running downstream workflow or state-changing platform commands (each downstream skill owns its own commands)
  - loading multiple downstream skills in one turn
compatibility: nemo-platform >= 0.1.0; selection plus a host scan on macOS or Linux; works without an installed CLI (selector can pick setup, which then tells the user how to run the CLI install).
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Bash
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Platform skill selection

Decide which downstream NeMo Platform skill should run. Bash access is unrestricted at runtime and
can execute state-changing commands; the scope below is a behavioral constraint, not an enforced
allowlist. Execute only the host scan in this skill's Pre-flight section, then announce the choice
and hand off. Never run downstream workflow or state-changing platform commands from this skill.

New NeMo Platform agent builds use a Platform-managed `agent.yaml` with
`config_format: nemo-agents-spec-v1` and a supported harness. NVIDIA NeMo Agent
Toolkit (NAT) workflow YAML remains a compatibility path. Do not describe NAT
as the only supported implementation model.

If an existing agent does not fit a supported harness contract, route based on
the user's goal: preserve an existing NAT workflow, identify a custom adapter,
or use `nemo-agent-config` for a best-effort migration. Do not promise that an
arbitrary Python entrypoint can be converted mechanically.

## Decision table

Match the user's intent to one downstream skill. Pick exactly one.

| The user says or implies | Hand off to | Why |
|---|---|---|
| "set up", "install", "get started", "try NeMo", "first time" | `setup` | Verify the platform is installed and running. If not, the skill tells the user how to run the CLI install (`make bootstrap` + `nemo setup`). Install itself is CLI-only. |
| "design an agent", "I want an agent that handles X", "what should my agent do" | `nemo-explore` | Capture the agent's job, audience, categories, tools, model, constraints before any code |
| "write the ethos", "save the design", "capture what we agreed" | `nemo-ethos` | Persist the explore answers as `agents/<name>-ethos/ETHOS.md` |
| "write agent.yaml", "validate agent.yaml", "choose a harness", "migrate this NAT YAML", "convert to nemo-agents-spec-v1" | `nemo-agent-config` | Author or migrate the Platform-managed machine-readable config without running the full build |
| "build the agent", "create the agent", "deploy", "scaffold from ethos" | `nemo-build-agent` | Build from the approved Ethos, default to Platform `agent.yaml`, register, deploy, evaluate, and optionally apply guardrails |
| "ask my agent", "try the agent", "test it", "invoke this agent.yaml" | `nemo-try-agent` | Invoke a named deployment or run a local agent YAML config once |
| "instrument my agent", "send traces", "use Intake", "agent observability", "query spans or traces" | `nemo-intake` | Choose an ingest path, instrument the source, ingest telemetry, and verify spans, traces, sessions, or evaluator results |
| "create an experiment", "publish evaluation runs", "evaluation leaderboard" | `nemo-experiments-upload` | Create Experiments and Evaluations, ingest their telemetry and scores, and verify leaderboard rollups |
| "status", "what is running", "platform health", "is the platform up", "what's deployed", "show me what's running" | `nemo-status` | Read-only dashboard: platform, agents, providers, models |
| "shut down", "stop NeMo", "tear down", "clean up" | `nemo-teardown` | Stop the cluster (keep data, delete platform data, or full cleanup) |
| "fine-tune embedding", "fine-tune rerank", "retrieval recipe", "domain corpus retrieval", "Nemotron embed", "Nemotron rerank", "retrieve-eval", "nDCG on my documents" | `nemo-retrieval-recipes` | Cross-plugin embed/rerank recipe: Data Designer retrieval jobs, Automodel `bi_encoder`/`cross_encoder`, Evaluator `retrieve-eval`. Not chat SFT. |
| "fine-tune", "customize the model", "train on my data", "SFT", "LoRA" | `nemo-customizer` | Chat/SFT/LoRA/DPO/GRPO via customization contributor plugins. Embedding/rerank domain pipelines go to `nemo-retrieval-recipes` first. |
| "why does my agent keep failing", "analyze my agent's traces", "find recurring failure patterns", "generate insights for my agent" | `nemo-analyst` (plugin-owned, in `plugins/nemo-insights`) | Reads an agent's existing telemetry and files each recurring failure pattern as an Insight citing the traces that evidence it. Requires the Insights plugin; produces the Insight `nemo-experimentalist` acts on. |
| "improve the agent's own code", "fix my agent harness", "candidate code change", "optimize from an Insight", "improve on train and validation datasets" | `nemo-experimentalist` (plugin-owned, in `plugins/nemo-experimentalist`) | Source/harness optimization: generate and validate candidate code changes against Harbor-compatible evaluation data. Requires the Experimentalist plugin; use after `agents analyst` has created an Insight, or with explicit datasets. |
| "optimize my agent", "make it cheaper", "reduce latency", "smaller model", "switchyard", "routing split", "compare against a newer model" | `agents-optimize` (plugin-owned, in `plugins/nemo-agents`) | Cost / latency / quality optimization for a **deployed** agent. Routing splits, skill tuning, prompt tuning, new-model scans. |
| "secure my agent", "harden my agent", "check for PII", "leaked secrets", "guardrail coverage" | `agents-secure` (plugin-owned, in `plugins/nemo-agents`) | Safety and security audit for a **deployed** agent. Guardrails, PII, secrets scan. |
| "evaluate my agent", "run a benchmark", "eval suite" | `nemo-evaluator` (plugin-owned, in `plugins/nemo-evaluator`) | Evaluation metrics, LLM-judge, benchmark jobs against a deployed agent or model. |

**Optimize vs build:** Do NOT route optimize asks to `nemo-build-agent`. Build is for creating new agents from a spec. Use `agents-optimize` for a deployed agent's routing, prompts, skills, cost, or latency; use `nemo-experimentalist` when the requested improvement changes the agent's own source or harness and is evaluated with candidate code changes. If the user says "make my agent faster" or "use a cheaper model," that is `agents-optimize`, not `nemo-build-agent`.

If a request includes both config authoring and deployment, choose
`nemo-build-agent`; it delegates the config portion to `nemo-agent-config`.
Choose `nemo-agent-config` when the requested output stops at a validated config
or migration. Otherwise, if two rows fit, pick the earliest one in the
lifecycle. If nothing matches, ask one disambiguating question with the
relevant rows as a numbered list.

## Pre-flight

Before handing off, run a host-wide platform scan. Three signals, in order — the first one that fires wins:

```bash
# 1. Ground truth: is anything listening on the canonical port?
lsof -iTCP:8080 -sTCP:LISTEN 2>/dev/null

# 2. Functional check: does the platform readiness endpoint answer?
curl -sS --connect-timeout 2 --max-time 5 http://localhost:8080/health/ready -o /dev/null -w "%{http_code}\n" 2>/dev/null || echo "no-response"

# 3. Conflict check: other platform processes / data dirs / configs on this host?
ps -eo pid=,user=,comm=,args= 2>/dev/null \
  | awk '$0 ~ /[n]emo services (run|start)|[n]emo-platform run/ {print $1, $2, $3}'
ls -d ~/.local/share/nemo* 2>/dev/null
ls ~/.config/nmp*/config.yaml 2>/dev/null
```

Interpretation:

| What you observe | Hand off to | Why |
|---|---|---|
| (1) returns a listener AND (2) returns `200` | the requested downstream skill | Platform is up and ready. Skip `setup`. |
| (1) returns a listener but (2) returns `no-response` or non-200 | `nemo-status` | Something is bound to :8080 but the platform is not ready. Do not start a second platform. |
| (1) empty but (3) finds another `nemo services` process OR more than one data dir / config | **stop, do not hand off yet** | Another install on this host, possibly on a different port. Surface only the redacted PID, user, and executable inventory emitted above. Ask whether to tear that one down first, pick a different port + data dir, or abort. Two installs writing to the same `~/.config/nmp/config.yaml` is how users end up with one Studio frontend pointing at the wrong backend. |
| (1), (2), and (3) all empty | `setup` | Clean machine, no platform installed. |

Read-only callers (this skill, `nemo-status`, the build/try pre-flights) should not trust `nemo services status` or `nemo services ls` as an up-check. Both report stale "running" from a held instance lock after the underlying process has died. The lock reconciles automatically the next time `nemo services run` is invoked, but until that happens, `lsof` is ground truth. (Tracking a CLI-side fix for this so we can drop the workaround from skills.)

## What to announce

Tell the user, in one sentence, which skill is next and what it will do. For example: "Handing off to `setup` to verify the platform is installed and running. If it isn't, the skill will tell you the CLI command to run; install is a 5-minute shell step that this skill cannot do reliably for you."

Then hand off. Do not run downstream workflow or state-changing platform commands from this skill.

## If nothing matches

If the user's intent doesn't fit any row, do not guess. Read out the available skills and ask which one they want:

```
NeMo Platform skills I can route to:
  setup           verify install or get the CLI install command
  nemo-explore    design conversation: capture goal, audience, tools, constraints
  nemo-ethos      write the design to agents/<name>-ethos/ETHOS.md
  nemo-agent-config  author, validate, or migrate Platform agent.yaml
  nemo-build-agent  build from the Ethos, register, deploy, evaluate, and sign off
  nemo-try-agent  invoke a named deployment or local agent YAML config
  nemo-intake     instrument agents, ingest/query telemetry, attach scores
  nemo-experiments-upload  publish named evaluation runs to an Experiments leaderboard
  nemo-status     read-only platform health dashboard
  nemo-teardown   guided shutdown
  nemo-retrieval-recipes  embedding/rerank domain fine-tune on Platform plugin jobs

Plugin-owned skills:
  agents-optimize   cost / latency / quality optimization for a deployed agent
  agents-secure     safety and security audit for a deployed agent
  nemo-evaluator-plugin  evaluation metrics, LLM-judge, retrieve-eval, benchmark jobs
  nemo-customizer   fine-tuning of chat/SFT/RL models (not the retrieval recipe)
  nemo-analyst      analyze agent telemetry and file recurring problems as Insights
  nemo-experimentalist  source/harness optimization from Insights or evaluation datasets
  guardrails        content-safety middleware via virtual models
  auditor           red-team vulnerability scanning (garak)
  nemo-data-designer-plugin  synthetic dataset generation (tabular `create`; retrieval SDG is the retrieval recipe)
  anonymizer        PII handling for datasets

Which one fits what you're trying to do?
```

For things outside this catalog (for example, "show me how Switchyard routes between models"), point at the relevant repo skill (`nemo-evaluator`, `nemo-auditor`, etc.) or tell the user no skill claims that intent yet. Do not invent a path.

If the pre-flight finds no platform but the user insists they have installed one: ask them to report
the output of `lsof -iTCP:8080 -sTCP:LISTEN` and the redacted scan below from the shell where they ran
setup. The platform may be bound to a non-default port, or the install may be in a venv whose `nemo`
binary is not on `PATH`.

```bash
ps -eo pid=,user=,comm=,args= 2>/dev/null \
  | awk '$0 ~ /[n]emo services|[n]emo-platform run/ {print $1, $2, $3}'
```

## If the user asks about Studio (web UI)

Skills route through CLI commands, not Studio. But customers ask "what's Studio?" or "do you have a web UI?" Answer honestly, do not invent capabilities, and do not steer users into the experimental flows.

What to say:

- Studio is the NeMo Platform web UI. When the platform is running locally, it serves at `http://localhost:8080/studio`.
- Documentation: `docs/studio/index.md` in this repo covers the stable views (Agents, Optimizations, Monitor, Workspaces, Datasets). Point users there rather than enumerating features in-conversation — the docs stay up to date, this skill won't.
- **Honest caveats to flag every time:**
  - The **Optimizations "Apply suggestion"** flow is **incomplete today**. Suggestions render, but the apply action is not reliable end-to-end. Tell the user to apply optimizer suggestions via the CLI (`nemo agents …`) instead, using the suggestion's `apply` block as the source of truth — see `agents-optimize`.
  - Other views may evolve; refer to the docs for the current state rather than promising specific behavior.
- For local development on Studio itself, the source lives at `web/packages/studio/`. The `studio-dev` skill (if available) covers that workflow.

Do not proactively suggest Studio as the path for anything a skill already covers (chat, deploy, status, teardown, optimization). The CLI path is what these skills verify and what we can confidently support.

## Gotchas

- **One skill at a time.** Do not load more than one downstream skill in the same turn. Each downstream skill is a full procedure with its own context budget.
- **Install must happen before any skill can do useful work.** Build, try, and status all assume the platform is up. If the user has not run the CLI install (`make bootstrap` + `nemo setup`), the skills cannot work around that; hand them to `setup` for instructions.
- **NeMo Platform is the product name.** Capital N, e, M, o, P. Not "nemo" or "Nemo." NAT on first mention is "NVIDIA NeMo Agent Toolkit (NAT)."
- **Model customization** for chat/SFT/RL goes to the `nemo-customizer` plugin skill when `nemo-customizer-plugin` (and a training backend) are installed. **Embedding and reranking recipes** (domain corpus, retrieval SDG, `bi_encoder` / `cross_encoder`, `retrieve-eval`) go to `nemo-retrieval-recipes`. If those skills are not available, tell the user to enable the plugins and install skills — do not improvise training with an external library.
- **Execution compatibility.** New Platform configs must select a supported
  harness. Existing NAT workflows may remain on the NAT compatibility path.
  For another framework or an arbitrary Python entrypoint, inspect whether a
  supported harness owns its lifecycle; otherwise identify a custom adapter or
  NAT wrapper instead of claiming direct support.
