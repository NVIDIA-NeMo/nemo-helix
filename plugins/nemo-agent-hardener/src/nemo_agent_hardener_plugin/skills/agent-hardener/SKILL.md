---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: agent-hardener
description: >
  NeMo Agent Hardener CLI reference for red-teaming and hardening NAT agents.
  Use when the task involves attacking, red-teaming, hardening, or running a security
  war-game against an agent, or any `nemo agent-hardener` CLI command.
---

# NeMo Agent Hardener CLI Reference

Agent Hardener runs a security war-game against a NAT agent: garak attackers probe a sandboxed copy
of the agent, defenders harden it (OpenShell policy + workflow guardrails), and validators replay
the attacks plus benign traffic to confirm the fix. Point it at an agent registered in NeMo Platform
(it does not have to be deployed) or at a local NAT project — no manual manifest editing either way.

## Prerequisites

Agent Hardener needs Docker and an OpenShell gateway, and runs in its own isolated venv. Provision
once per host, then verify:

```bash
# Provision agent-hardener's dedicated venv and check host prerequisites
nemo agent-hardener setup

# Read-only preflight: venv present, Docker daemon up, OpenShell gateway connected
nemo agent-hardener doctor
```

`setup` provisions the Python venv for you. Host-level prerequisites (Docker, the OpenShell
gateway) are checked and the exact install command is printed if missing — they are never
silently installed (they need sudo/brew/systemd).

## Typical flow

Three steps, in this order. **`synth-benign` is not optional** — see below.

```bash
# 1. Save a target. It prints the manifest id every later command takes.
nemo agent-hardener init --agent <agent-name>
nemo agent-hardener init --agent <agent-name> --egress <host>      # if the agent calls the internet

# 2. Generate the benign suite (required, cached on the manifest)
nemo agent-hardener synth-benign --manifest-id <id> --yes

# 3. Run the war-game, then inspect it
nemo agent-hardener run --manifest-id <id>
nemo agent-hardener status --limit 5

# Later: the agent changed and you want the manifest to catch up
nemo agent-hardener refresh --manifest-id <id>
```

For agent source, the agent must already be registered in NeMo Platform — confirm with
`nemo agents list`. It does not need to be deployed; the war-game builds its own victim.

### Targeting a local NAT project instead

```bash
nemo agent-hardener init --project-dir <path>          # asks about workflow, port, secrets
nemo agent-hardener init --project-dir <path> --yes    # CI: accept detected answers
```

This runs agent-hardener's own interactive `init` in the terminal, then uploads the project and saves
the result. From there the flow is identical — same `--manifest-id`. The project supplies its own
dependencies, so its `pyproject.toml`/`requirements.txt` must include `nvidia-nat`.

Secrets and credentials are never uploaded: the victim's secrets come from the platform secret
store, resolved by the names in the manifest.

## `synth-benign` is required

The war-game validates two things: that attacks are blocked, **and** that ordinary requests still
work. Those ordinary requests are the *benign suite*, and `run` is a pure consumer of it — it never
generates one. Without a suite the run fails immediately with
`smart-benign validation requires an explicit benign suite`.

Generate it once and it is cached on the manifest for every later run:

```bash
nemo agent-hardener synth-benign --manifest-id <id>                  # interview, you answer
nemo agent-hardener synth-benign --manifest-id <id> --yes            # interview, defaults accepted
nemo agent-hardener synth-benign --manifest-id <id> --no-interactive # CI: rules only
```

## Two ways to run

- **`run --manifest-id <id>`** — the normal path. Looks up the cached benign suite by id.
- **`run --config <file>`** — a hand-authored `agent-hardener.yaml` against a local project. There is no
  cache to look up, so it also needs `--benign-suite <csv>`.

`init -o` writes a *rendering* of the manifest for reading; editing that file changes nothing, since
the run uses the saved manifest.

## Manifests are frozen targets

`init` resolves the agent once and stores the result, so every run war-games the same thing — which
is what makes two runs comparable. Editing the agent afterwards does **not** change an existing
manifest. Take the change deliberately:

```bash
nemo agent-hardener refresh --manifest-id <id>
```

Egress, secrets, models, defenders and the cached benign suite are preserved; only the target is
rebuilt. Not needed after `apply-mitigation`, which refreshes the manifest itself so
`run → harden → apply → run` measures the change just applied.

Do not report a manifest as picking up an agent edit without a `refresh` (or an `apply`) between them.

## Egress: the silent-failure trap

The sandbox drops outbound traffic that is not allow-listed, and a blocked host **hangs** rather
than erroring. The agent's LLM then answers from its own knowledge, so the run looks like it passed
while the tool path — the thing Agent Hardener exists to attack — was never exercised.

Pass `--egress <host>` at `init` for every external host the agent reaches. A bare host opens
**443 only**; write `host:80` for plain HTTP. Hosts cannot be auto-discovered for a config-only
agent, because its tool code lives in an installed package rather than in the project.

## Environment variables vs secrets

`--env KEY=VALUE` (repeatable) sets non-secret env vars on the victim; only the first `=` splits.
Values are stored in plaintext on the manifest, so credentials must use `--secrets` instead, which
stores only the names and resolves values from the platform Secrets store at run time. Never suggest
putting an API key in `--env`.

## Notes

- Models default to agent-hardener's built-ins. Three overridable groups: **attack** (garak red-team +
  detector), **analysis** (defenders + the benign validator's synth suite-generation and judge), and
  **safety** (the LLM the generated guardrail screens traffic with). Set them as the manifest's stored
  default (`manifest set` / `init`) or per-run (`run`); a custom endpoint's key is supplied by name from
  the Secrets store via `--attack-key-secret` / `--analysis-key-secret`. A chosen model is preflighted
  before the sandbox spins up, and the run fails fast (listing the reachable models) on a bad name/key/URL.
- The victim's own LLM is deliberately **not** a group. In NAT it is declared in the workflow YAML, so
  overriding it would mean rewriting the target — the war-game measures the agent rather than editing it.
  Change it in the project's workflow and re-upload.
- `--safety-model` names the model only; agent-hardener pins that LLM's endpoint and key to NVIDIA's when it
  generates the guardrail. To run the guardrail on **your own provider**, declare `llms.safety_llm`
  yourself in the workflow — a pre-declared one is left untouched, and `--safety-model` is then ignored:

  ```yaml
  llms:
    safety_llm: { _type: inference_nim, model: my/guard, base_url: https://my-endpoint/v1, api_key: ${GUARD_API_KEY} }
  ```

  Its key must reach the sandbox, so name it in the manifest's `secrets` (`init --secrets GUARD_API_KEY`)
  and allow the host with `--egress my-endpoint`, or the call hangs until the defender's 300s timeout.
- Workflow `model_name` values must be entity names the platform knows (lowercase letters, digits,
  hyphens) — a provider id like `vendor/model-name` is rejected for the slash.
- After a run produces mitigations, freeze a chosen subset and replay the recorded attacks:
  `nemo agent-hardener sanity-check --manifest-id <id> --mitigations <json> --replay-hitlog <ref> --keep <id>`
- A war-game makes many LLM calls and bills a real key. Say so before starting long or repeated runs.
- The war-game requires a host with Docker + OpenShell; the run job fails preflight with a clear
  message otherwise. Do not report a run successful without verifying its status.
