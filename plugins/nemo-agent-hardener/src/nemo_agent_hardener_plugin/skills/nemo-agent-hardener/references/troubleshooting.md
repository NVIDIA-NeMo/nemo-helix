<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# War-game troubleshooting

Symptom → cause → fix. Every entry here has actually happened.

## Setup

| Symptom | Cause | Fix |
|---|---|---|
| `flox is required` or `flox: command not found` during `make bootstrap` | Flox is not installed | `curl -fsSL https://get.flox.dev \| sh`, then `make bootstrap` and `flox -q activate` again |
| `nemo: command not found` | the Helix environment is not active in this shell | `cd nemo-helix && flox -q activate`. Never `apt install nemo` — an unrelated file manager |
| `No such command 'agent-hardener'` | `nemo` comes from a PyPI install, which has no Agent Hardener plugin (`which nemo` does not end in `nemo-helix/.venv/bin/nemo`) | remove it and install from source: [setup.md](setup.md) section 2 |
| Port 8080 taken by a Helix that does not answer `/health/ready` | an older or hung Helix instance | confirm with `nemo services status`, then (with the user's OK) `nemo services stop` — `--force` for one running in a terminal's foreground |
| `No VirtualModel named '...' was found` | the model in the inference request is not served by the provider | use a model from `nemo models list`, with its `default/` prefix (the default `nemo setup` printed) |
| `doctor`: inference credential ✗ | `INFERENCE_API_KEY` was not exported when `setup` ran | export it, then `nemo agent-hardener setup --force` |
| `doctor`: openshell gateway ✗ | the gateway is not running or not registered | re-run `nemo agent-hardener setup`; if it cannot restart the service, `systemctl --user restart openshell-gateway`, check lingering ([setup.md](setup.md) section 3), re-run `setup` |

## Keys and models

| Symptom | Cause | Fix |
|---|---|---|
| `synth-benign` or `run` fails with `victim_unavailable: the victim answered the Relay probe with HTTP 500` | the agent's own model call failed. Read the agent's log: `--no-cleanup` keeps the sandbox, then `openshell sandbox exec --gateway auto-defender --name <sandbox> --no-tty -- tail -50 /tmp/agent.log` | `401` → see the next row; `503` or timeouts → the model is down on build.nvidia.com: retry later, or point the agent at another model with `--env` |
| The agent's model calls return `401` in the sandbox although the key works with `curl` (and `setup --force` did not help) | the sandbox gets its secrets from the OpenShell provider `agent-hardener-secrets`, which keeps the key from the run that created it — Agent Hardener never updates an existing one | with the user's OK: delete the sandboxes that use it (`openshell sandbox list`, `openshell sandbox delete <name> --gateway auto-defender`), then `openshell provider delete agent-hardener-secrets --gateway auto-defender` and `openshell provider profile delete agent-hardener-secrets --gateway auto-defender`; the next run recreates it with the current key |
| A run returns `401` from Agent Hardener's own models | the key in `~/.agent-hardener/.env` is wrong for `integrate.api.nvidia.com` | check the key with `curl` ([setup.md](setup.md) section 4), then `nemo agent-hardener setup --force` with the correct key exported |
| A run fails with `429` or `rate limit` errors | the NVIDIA key hit its rate limit | switch Agent Hardener's models to another provider (SKILL.md, "Changing Agent Hardener's models") |
| `safety model … not applied: no defender selection to attach it to` | `--safety-model` is set but the manifest has no defender selection, so the default judge is used | `manifest set <name> --defender guardrails --defender openshell` |
| After `manifest set --env`, the agent cannot reach its model | `--env` replaced the whole env map and dropped the values not passed (for example `INFERENCE_BASE_URL`) | pass every `--env KEY=VALUE` again; check with `manifest show` |
| `init` fails with `HTTP 409: Manifest '…' already exists` | the name is taken; there is no `manifest delete` | change it with `manifest set`, or use a new `--name` and run `synth-benign` for it before `run` |

## Run or job never starts

| Symptom | Cause | Fix |
|---|---|---|
| Studio run (or job) stuck in `created` forever, no error | the platform is running without the jobs controller | restart with `nemo services run … --controllers jobs` |
| Studio shows "404 Error" on the Agent Hardener route, or renders a blank/naked page; browser console shows `Invalid URL: STUDIO_UI_VITE_PLATFORM_BASE_URL/apis/...` | the platform was started without a Studio base URL, so that build-time marker is never substituted and *every* Studio API call fails — it looks like a plugin bug but breaks all of Studio | restart with `NHX_STUDIO_PLATFORM_BASE_URL=$NHX_BASE_URL` set |
| `Port 8000 already in use` at victim start | another war-game, or a stale port-forward from a previous run | `lsof -iTCP:8000 -sTCP:LISTEN`, kill the holder, or pass `--port` |
| `nemo agents register` fails on upload size | run state or venvs staged next to `agent.yaml` (the upload is capped at ~900 KB / 500 files) | register from a clean directory holding only the agent's own files |

## Victim dies or times out mid-run

| Symptom | Cause | Fix |
|---|---|---|
| Attacks report unblocked while the victim answered 502, container SIGKILLed | Docker VM out of memory — each in-flight Fabric session holds ~160 MB | give the Docker VM >= 8 GB; Agent Hardener already injects the session-reclaim flags, but an undersized VM still OOMs |
| `attacker_failed`: garak `ReadTimeout` (`read timeout=600`) on one turn | a single agent turn ran past the per-turn cap (`garak.request_timeout`, default 600 s): the agent looped, or one of its tools never returned (an endless loop or a read that never ends inside a code-running tool) | Route A: bound the agent with `runtime.max_turns` in `agent.yaml` (maps to LangGraph's recursion limit). Route B: bound each tool call (run it in a subprocess with a timeout, as the bundled `langgraph-victim` does). Raising the timeout only moves the wall |
| Benign requests scored as errors: `replay failed: ReadTimeout` after 240 s | the same hang as above, hit while the validators replay | fix the victim first, then re-run |

## Instrumentation

| Symptom | Cause | Fix |
|---|---|---|
| `VictimNotInstrumentedError` (any variant) | Relay not attached, not started, or not in the tool path | [relay-attachment.md](relay-attachment.md) — the error text maps to the exact missing obligation |
| Derived egress `(none)` for an agent with remote tools or a remote model | the config names no hosts, so discovery had nothing to read | pass `--egress host[:port]` per legitimate host (bare host = 443 only), then `refresh` |

## Replay and filesets

| Symptom | Cause | Fix |
|---|---|---|
| `Fileset name is required` (or the ref is not found) on `--replay-hitlog` | a bare name or file path was passed where a `<workspace>/<fileset>` reference is expected | `nemo files filesets create` first, `nemo files upload` into it, then reference `workspace/name` — upload does **not** auto-create a named fileset |
| The guardrails defender wrote no rail for an attack: `no execution trace for this attack` | the attack left no Relay trace it could be matched to when it was sent (live, or re-sent by a replay): Relay is not attached to the victim, the agent never answered it, or — on a hand-built LangGraph graph — the model node calls the model outside Relay, so traces carry no prompt (tool calls still show) | confirm Relay is attached and, on a hand-built graph, that model calls go through `nemo_relay.llm.execute` ([relay-attachment.md](relay-attachment.md), obligation 3); then run or replay again |

## Results read wrong

| Symptom | Cause | Fix |
|---|---|---|
| Run ends `validation_failed` (`status` shows it as `failed (exit 1)`) | some attacks still got through the new guardrails, or benign requests failed — a result, not a crash | read the scorecard: a benign *error* (timeout) is a broken victim, a benign *refusal* is a false positive; iterate with `--replay-hitlog` |
| `smart-benign validation requires an explicit benign suite` | `synth-benign` was not run for this manifest | `nemo agent-hardener synth-benign --manifest-id <name>`, then `run` again |
| Scorecard shows blocks but the user expected guardrail hits | many "blocks" are the model refusing on its own, not the guardrail firing | read each item's badge — *Blocked by guardrail* / *Blocked by network* versus *Agent refused* — or check the ATOF trail, where guardrail refusals carry the `agent-hardener[rail]` signature |
