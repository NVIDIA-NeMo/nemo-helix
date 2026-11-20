---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-agent-hardener
description: >
  A security war-game for an agent through NeMo Helix: set up a clean VM (Helix from source,
  OpenShell, the model key), register or upload the agent, resolve a war-game manifest, synthesize a
  benign suite, attack/defend/validate, adopt the hardened image. Covers red-teaming, hardening, and
  every `nemo agent-hardener` command.
triggers:
  - nemo-agent-hardener
  - set up a VM for agent hardener
  - war-game an agent
  - red-team this agent
  - harden the agent
  - agent hardener
  - attack a deployed agent
  - prove the agent survives prompt-injection attempts
  - war-game a LangGraph Dockerfile project
  - war-game validation_failed result
not-for:
  - garak (use for a one-shot garak scan without hardening)
  - guardrails-plugin (use to author guardrails directly, without a war-game)
  - nemo-try-agent (use to chat with a deployed agent)
  - nemo-status (use for a read-only platform dashboard)
preconditions:
  - nemo_cli_available
compatibility: >
  nemo-helix >= 0.1.0 installed from source (the PyPI package has no Agent Hardener plugin); needs
  Docker, OpenShell >= 0.1.4 with its gateway, and an NVIDIA API key — `nemo agent-hardener doctor`
  checks them, and references/setup.md sets up a clean VM. Not sandbox-safe: it builds images, starts
  containers, and makes network calls. Give the machine >= 16 GB memory (32 GB recommended) and
  50 GB free disk.
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read, Write
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
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
> command (`agents register`, `agents package`, `agent-hardener init`). Picking one yourself — because it
> is the default, the first listed, or the one you just built — is the single most common way this
> skill goes wrong. Ask first, every time.

## 1. Pre-flight

Check the machine before anything else. On a clean VM, or when any check below fails, walk
[references/setup.md](references/setup.md) section by section (VM checks, Helix from source, the
services, OpenShell, the model key, `setup`) and come back here with `doctor` green.

```bash
which nemo                         # must end in nemo-helix/.venv/bin/nemo (a source install)
nemo services status               # Health: healthy, on port 8080
openshell --version                # 0.1.4 or later
# `nemo` resolves its target from the active config context (~/.config/nhx/config.yaml)
# or an explicit NHX_BASE_URL. Do NOT export a localhost fallback here — it would
# override a configured remote. Leave NHX_BASE_URL unset to use your saved context;
# set it explicitly (export NHX_BASE_URL=http://localhost:8080) only to force a local
# platform for this shell when you have no local config context.
nemo agent-hardener doctor         # five ✓ lines, then "All checks passed."
```

- `nemo` not found, or `No such command 'agent-hardener'` → Helix is missing or a PyPI install;
  [references/setup.md](references/setup.md) section 2. Never `apt install nemo` (an unrelated
  file manager).
- `doctor` failing on the venv or the inference credential → export `INFERENCE_API_KEY` (an
  `nvapi-` key) and run `nemo agent-hardener setup` (`--force` to replace a saved key), then
  `doctor` again.
- The platform must run with **all** of: `--controllers models,jobs`, `--host 0.0.0.0`, and
  `NHX_STUDIO_PLATFORM_BASE_URL` set — the exact command is in setup.md section 2. Without the jobs
  controller, Studio and job-driven runs sit in `created` forever — silently.
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
nemo agents register --name my-agent --agent-config agent.yaml
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
- A name is used once: `init` with an existing `--name` fails with `HTTP 409 … already exists`, and
  there is no `manifest delete`. Change an existing manifest's settings with `manifest set`.
  `refresh` re-reads a registered agent (Route A) only; it does not upload edited project files. After
  editing the project, `init` it again under a new `--name`, then run `synth-benign` for it before
  `run`, since a new name has no benign suite.

**Worked example — the bundled `langgraph-victim`, with an NVIDIA Build key.** Its Dockerfile states
none of the secrets, model settings or hosts, so `init` warns "No credential-shaped env names" and
"No hosts named" — expected here, and covered by the flags. Run from the `nemo-helix` folder:

```bash
nemo agent-hardener init \
  --project-dir plugins/nemo-agent-hardener/examples/langgraph-victim \
  --name langgraph-victim \
  --harness langgraph \
  --relay-confirmed \
  --secrets INFERENCE_API_KEY \
  --env INFERENCE_BASE_URL=https://integrate.api.nvidia.com/v1 \
  --env INFERENCE_MODEL=nvidia/nemotron-3-ultra-550b-a55b \
  --egress integrate.api.nvidia.com \
  --start-command "/usr/local/bin/python /app/server.py" \
  --binary "/usr/local/bin/python*"
```

| Value | Why |
|---|---|
| `--secrets INFERENCE_API_KEY` | the agent's model key; its value comes from `~/.agent-hardener/.env`, written by `setup` |
| `--env INFERENCE_BASE_URL=…`, `--env INFERENCE_MODEL=…` | without them the example calls `inference-api.nvidia.com` (its README's setup), which an `nvapi-` key cannot use |
| `--egress integrate.api.nvidia.com` | the sandbox is default-deny; without it the agent cannot reach its model |
| `--start-command "/usr/local/bin/python /app/server.py"` | absolute: OpenShell replaces `PATH`, and the Dockerfile's `CMD` uses a bare `python` |
| `--binary "/usr/local/bin/python*"` | which process the egress rule applies to; the image has no venv, so the default does not match |

### Route C — a `claude` / `codex` gateway agent

Stop honestly: Agent Hardener refuses these at `init` by design. They run behind a compiled gateway
with no in-process interception point, so a guardrail could never actually block a tool call.
Suggest the `garak` skill for scan-only coverage.

## 3. Interview — Studio or CLI?

### Studio

1. Collect credentials first: store each secret the agent needs with the platform `nemo-secrets`
   skill (or `nemo secrets` CLI) so the run can reference them.
2. Open `$NHX_BASE_URL/studio/`, pick the workspace, choose **Governance → Agent Hardening** in
   the nav.
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
!nemo agent-hardener synth-benign --manifest-id my-agent
```

Only run it yourself if the user explicitly asks you to, or if they want the interview skipped:
`--yes` keeps the interview but auto-accepts each recommended default; `--no-interactive` skips it
entirely (leaner rules-only suite, for CI).

- The agent's secrets come from `~/.agent-hardener/.env` (written by `setup`). Add
  `--env-file ./secrets.env` only for secrets that file does not hold; its values win.
- Verify: it ends with `Cached N benign requests` (a few minutes). A `VictimNotInstrumentedError`
  here means Relay is not attached → [references/relay-attachment.md](references/relay-attachment.md).
  `victim_unavailable: the victim answered the Relay probe with HTTP 500` means the agent's own model
  call failed (a wrong key, or the model down) → [references/troubleshooting.md](references/troubleshooting.md).

## 5. Run the war-game (CLI)

```bash
nemo agent-hardener run --manifest-id my-agent
```

A run takes a while (about 20–40 minutes for the bundled examples): it builds the image, starts
the agent in the sandbox, then attacks, defends and validates. Do not start the agent yourself:
Agent Hardener runs its own sandboxed copy and forwards it to port 8000, which must be free.

- Fast path for iterating on defenses — replay a previous run's hits instead of re-attacking:
  `--replay-hitlog <workspace>/<fileset>`. That is a **fileset reference, not a file path**; create
  the fileset first (`nemo files filesets create …`) and upload into it (`nemo files upload …` does
  not auto-create a named fileset). A replay still sends each hit to the agent once, before
  defending, to record its execution trace through Relay: the guardrails defender writes each rail
  from that trace.
- One war-game per host: every victim binds port 8000. If a second run is needed, wait or use
  `--port`.
- Verify: `nemo agent-hardener status --limit 5` lists the run. A finished run that did not pass
  shows there as `failed (exit 1)`; the reason is the `error.category` that `run` prints:
  `validation_failed` is a *result* — some attacks got through, or benign requests failed — not a
  crash; read the scorecard. `attacker_failed` and `victim_unavailable` are crashes →
  [references/troubleshooting.md](references/troubleshooting.md).

### Changing Agent Hardener's models

By default the attacker, the defenders and the guardrail judge use `nvidia/nemotron-3-ultra-550b-a55b`
on the NVIDIA key. To change them — for example on `429` / rate-limit errors, by moving to another
provider such as OpenRouter — set the three model groups on the manifest (`init` and `run` take the
same flags):

```bash
nemo agent-hardener manifest set my-agent \
  --attack-model <model> --analysis-model <model> --safety-model <model> \
  --defender guardrails --defender openshell
# for another provider add --<group>-base-url <url> --<group>-key-secret <secret> per group
```

- `--safety-model` (the guardrail judge) takes effect only with an explicit defender selection;
  without `--defender`, the run warns `safety model … not applied: no defender selection` and keeps
  the default.
- `--env` on `manifest set` **replaces** the whole env map: pass every `--env KEY=VALUE` again, or
  the ones left out are dropped (for the example, losing `INFERENCE_BASE_URL` sends the agent to a
  host outside its egress).
- These do not change the agent's own model: that is its `--env` (for the example,
  `INFERENCE_MODEL`).
- Check with `nemo agent-hardener manifest show my-agent`.

## 6. Read the results

The run directory (printed at the end, under the manifest's `.agent-hardener/` state) holds:

- the scorecard — attacks blocked, benign false positives, per-round detail;
- `plugins.toml` — the guardrails the defenders wrote;
- `mitigations.json` — machine-readable mitigation records;
- the hardened image/bundle — the user's image with the guardrails delivered into it.

The same run is in Studio: `$NHX_BASE_URL/studio/` → **Governance → Agent Hardening** shows the
attack graph, the recommended defenses and what each one blocks.

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
failure we have actually hit (setup, stuck jobs, keys, OOM-killed victims, timeouts, port
conflicts, fileset refs) to its cause and fix. Do not report a step done that you could not verify; surface what you
saw and ask the user.

## Gotchas

- Manifests are **frozen targets**. After a registered agent (Route A) changes, `nemo agent-hardener
  refresh --manifest-id my-agent` re-resolves it (Studio: **Refresh Target**); runs log a warning when
  the agent is newer than the manifest. `refresh` does not apply to a manifest made from a project
  directory (Route B): after editing the project, `init` it again under a new `--name` and run
  `synth-benign` for it (see the note on names above).
- A host service the victim calls (e.g. a remote MCP server) is declared at init with
  `--backend NAME:PORT[,PORT]` (repeatable) and reached via `host.docker.internal`. The gateway
  backend is added automatically (its name is reserved); declared backends survive `refresh`.
- `--env-file` values are merged into the victim's `.env` and win over the operator env.
- A failed run keeps its hitlog, so it can be replayed with `--replay-hitlog`.
- `nemo agent-hardener manifest show my-agent` / `manifest set` read and tweak the stored config.
  Model-group overrides (`--attack-model`, `--attack-key-secret`, `--analysis-model`, …) are on
  `init`, `manifest set` (stored) and `run` (this run only); see "Changing Agent Hardener's models".
- Env vars vs secrets: `--env KEY=VALUE` is for plain config; anything credential-shaped goes
  through `--secrets NAME=workspace/secret` (read from the Secrets store at run time) or `--env-file`,
  so it never lands in the manifest. A deployed agent's own secret references are recorded for you.
- A key exported in the shell that started the platform overrides `~/.agent-hardener/.env`; the run
  logs `the platform environment overrides <NAME>` when that happens.
- Packaging a Fabric agent (Route A, `nemo agents package`) from a source checkout of the platform
  needs `NEMO_AGENTS_ALLOW_UNPUBLISHED_CONTRACT_VERSION=1` in the platform's environment. Route B
  (`init --project-dir`) does not.
