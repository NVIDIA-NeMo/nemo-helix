---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-agent-hardener
description: >
  A security war-game for an agent through NeMo Platform: register or upload the agent, resolve a
  war-game manifest, synthesize a benign suite, attack/defend/validate, adopt the hardened image.
  Covers red-teaming, hardening, and every `nemo agent-hardener` command.
triggers:
  - nemo-agent-hardener
  - war-game an agent
  - red-team this agent
  - harden the agent
  - agent hardener
  - attack a deployed agent
  - prove the agent survives prompt-injection attempts
  - war-game a LangGraph Dockerfile project
  - war-game validation_failed result
not-for:
  - auditor (use for a one-shot garak scan without hardening)
  - guardrails-plugin (use to author guardrails directly, without a war-game)
  - nemo-try-agent (use to chat with a deployed agent)
  - nemo-status (use for a read-only platform dashboard)
preconditions:
  - nemo_cli_available
compatibility: >
  nemo-platform >= 0.1.0 with the nemo-agent-hardener plugin; needs Docker, an OpenShell gateway, and
  an inference credential — `nemo agent-hardener doctor` checks all three. Not sandbox-safe: it builds
  images, starts containers, and makes network calls. Give the Docker VM >= 8 GB memory.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: [Bash, Read, Write]
---

# NeMo Agent Hardener — war-game an agent

Agent Hardener attacks a sandboxed copy of an agent (garak), writes guardrails against what got
through (defenders), and replays the attacks plus benign traffic to prove the fix (validators).
**Done** means: a completed run with a scorecard — attacks blocked vs. benign false positives —
and a hardened image/bundle you can adopt.

This skill is an interview: ask the user two questions (steps 2 and 3) and follow only the branch
they pick. Do not dump every path at them.

> **Do not choose the target agent for the user.** Even when the request names a concrete goal
> ("harden an example agent"), you must still ask *which* one in step 2 before any state-changing
> command (`agents create`, `agents package`, `agent-hardener init`). Picking one yourself — because it
> is the default, the first listed, or the one you just built — is the single most common way this
> skill goes wrong. Ask first, every time.

## 1. Pre-flight

```bash
export NMP_BASE_URL=http://localhost:8080   # or wherever the platform runs
nemo agent-hardener doctor
```

- `doctor` failing on the venv → run `nemo agent-hardener setup` once, then `doctor` again.
- The platform must be running **with the jobs controller** (`nemo services run … --controllers
  jobs`). Without it, Studio and job-driven runs sit in `created` forever — silently.
- Verify: `doctor` reports every check green before moving on. If it cannot be made green, stop
  and tell the user what is missing; do not improvise around the sandbox.

## 2. Interview — which agent, and what kind?

**Ask the user which agent to war-game — do not pick for them.** Use an interactive question and
offer, at minimum:

- each bundled example victim as a concrete choice — the plugin ships one per guardable harness
  under `plugins/nemo-agent-hardener/examples/` (`relay-victim` deepagents, `hermes-victim` hermes,
  `langchain-victim`, `langgraph-victim`, `other-victim`); see that directory's README for the
  matrix;
- **an option to point you at their own directory** (a Fabric `agent.yaml` or a BYO Dockerfile
  project).

Once they choose, the agent's kind decides the route below — follow exactly one:

### Route A — a Fabric agent (has an `agent.yaml`, harness `deepagents` or `hermes`)

Register it on the platform first, then point Agent Hardener at the registered agent. Packaging
details (agent.yaml shape, Dockerfile contract): [references/agent-packaging.md](references/agent-packaging.md).

```bash
nemo agents package --agent agent.yaml --dockerfile ./Dockerfile --tag my-agent:v1
nemo agents create --name my-agent --agent-config agent.yaml
nemo agents deploy --agent my-agent --image my-agent:v1   # optional, but init reads the port from it
nemo agent-hardener init --agent my-agent --name my-agent
```

- Stage a **clean directory** before `create`: it uploads everything next to `agent.yaml`, capped
  at ~900 KB / 500 files. Run state (`.agent-hardener/`, venvs) in that directory fails registration.
- Verify: `init` prints the derived egress. `(none)` for an agent with network MCP servers or a
  remote model is wrong — stop and investigate before running; the sandbox is default-deny and the
  agent's tools will silently fail mid-run.

### Route B — your own image / framework (LangGraph, LangChain, DeepAgents library, …)

Bring-your-own mode: Agent Hardener war-games the image the user's own Dockerfile builds.

**First, confirm NeMo Relay is attached** — `--relay-confirmed` is a promise, not proof, and a
victim that *looks* instrumented but runs unguarded corrupts the whole run. Walk
[references/relay-attachment.md](references/relay-attachment.md); if the agent is not instrumented
yet, help the user make the edits it describes (and defer to NeMo Relay's own documentation and
published skills for frameworks beyond those).

```bash
nemo agent-hardener init --project-dir ./my-agent --name my-agent --harness langgraph --relay-confirmed
```

- Everything the Dockerfile states (port, start command, env, secret names) is derived. The
  command fails naming exactly what it could not derive — supply those flags (`--start-command`,
  `--dockerfile`, `--secrets`, `--egress`, …) and re-run.
- Secrets and egress usually cannot be derived: pass `--secrets NAME` for each credential and
  `--egress host[:port]` for each host the agent legitimately reaches (a bare host opens 443
  only).
- Verify: `init` writes the manifest and echoes the derived facts; read them back to the user and
  confirm they match the agent before continuing.

### Route C — a `claude` / `codex` gateway agent

Stop honestly: Agent Hardener refuses these at `init` by design. They run behind a compiled gateway
with no in-process interception point, so a guardrail could never actually block a tool call.
Suggest the `auditor` skill for scan-only coverage.

## 3. Interview — Studio or CLI?

### Studio

1. Collect credentials first: store each secret the agent needs with the platform `nemo-secrets`
   skill (or `nemo secrets` CLI) so the run can reference them.
2. Open `$NMP_BASE_URL/studio/`, pick the workspace, choose **Agent Hardener** in the nav.
3. **Manifests → New manifest** — the source toggle offers *Registered agent* and *Bring your
   own* (BYO takes a project archive upload).
4. Open the manifest page → **Run war-game** → fill env vars → submit.
5. Verify: the run appears on the runs list and its swarm lanes (attacker / sandbox / defender /
   validator) start moving. A run stuck in `created` means the jobs controller is missing
   (Pre-flight).

### CLI

Continue with steps 4–7 below.

## 4. Synthesize the benign suite (CLI)

The benign suite is what proves the defenders did not break the agent for real users. It is
**required** before a first run, and this is also the first victim bring-up — Relay problems
surface here, early and cheap.

**Do not run this command yourself — hand it to the user.** It is interactive by default: it
interviews them about realistic traffic, and only they can answer for their users. Ask them to run
it in this session with the `!` prefix, and wait for it to finish before continuing:

```bash
!nemo agent-hardener synth-benign --manifest-id my-agent --env-file ./secrets.env
```

Only run it yourself if the user explicitly asks you to, or if they want the interview skipped:
`--yes` keeps the interview but auto-accepts each recommended default; `--no-interactive` skips it
entirely (leaner rules-only suite, for CI).

- Verify: it ends with `Cached N benign requests`. A `VictimNotInstrumentedError` here means
  Relay is not attached → [references/relay-attachment.md](references/relay-attachment.md).

## 5. Run the war-game (CLI)

```bash
nemo agent-hardener run --manifest-id my-agent --env-file ./secrets.env
```

- Fast path for iterating on defenses — replay a previous run's hits instead of re-attacking:
  `--replay-hitlog <workspace>/<fileset>`. That is a **fileset reference, not a file path**; create
  the fileset first (`nemo files filesets create …`) and upload into it (`nemo files upload …` does
  not auto-create a named fileset).
- One war-game per host: every victim binds port 8000. If a second run is needed, wait or use
  `--port`.
- Verify: `nemo agent-hardener status --limit 5` shows the run progressing through rounds. A run that
  ends `validation_failed` is a *result* — some attacks still got through — not a crash; read the
  scorecard.

## 6. Read the results

The run directory (printed at the end, under the manifest's `.agent-hardener/` state) holds:

- the scorecard — attacks blocked, benign false positives, per-round detail;
- `plugins.toml` — the guardrails the defenders wrote;
- `mitigations.json` — machine-readable mitigation records;
- the hardened image/bundle — the user's image with the guardrails delivered into it.

Report the scorecard numbers to the user plainly, including what was **not** blocked.

## 7. Adopt the hardened agent

For a registered agent, deploy the hardened image the run produced:

```bash
nemo agents deploy --agent my-agent --mode docker --image my-agent-hardened:<run_id>
```

`--image` is rejected in the default subprocess mode — `--mode docker` is required. For a BYO
agent, hand the user the hardened bundle location; shipping it is theirs.

- Verify: `nemo agents deployments list` shows the new deployment, and a smoke request answers.

## If verification fails

Match the symptom in [references/troubleshooting.md](references/troubleshooting.md) — it maps each
failure we have actually hit (stuck jobs, OOM-killed victims, timeouts, port conflicts, fileset
refs) to its cause and fix. Do not report a step done that you could not verify; surface what you
saw and ask the user.

## Gotchas

- Manifests are **frozen targets**: after the agent changes, `nemo agent-hardener refresh
  --manifest-id my-agent` re-resolves it.
- `nemo agent-hardener manifest show my-agent` / `manifest set` read and tweak the stored config;
  model-group overrides (`--attack-model`, `--attack-key-secret`, `--analysis-model`, …) live on
  `run`.
- Env vars vs secrets: `--env KEY=VALUE` is for plain config; anything credential-shaped goes
  through `--env-file` / stored secrets so it never lands in the manifest.
- Running the platform from a source checkout needs
  `NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION=1` in the platform's environment.
