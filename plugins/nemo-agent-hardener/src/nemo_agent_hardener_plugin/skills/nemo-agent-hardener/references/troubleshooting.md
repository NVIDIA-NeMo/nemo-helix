<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# War-game troubleshooting

Symptom → cause → fix. Every entry here has actually happened.

## Run or job never starts

| Symptom | Cause | Fix |
|---|---|---|
| Studio run (or job) stuck in `created` forever, no error | the platform is running without the jobs controller | restart with `nemo services run … --controllers jobs` |
| Studio shows "404 Error" on the Agent Hardener route, or renders a blank/naked page; browser console shows `Invalid URL: STUDIO_UI_VITE_PLATFORM_BASE_URL/apis/...` | the platform was started without a Studio base URL, so that build-time marker is never substituted and *every* Studio API call fails — it looks like a plugin bug but breaks all of Studio | restart with `NMP_STUDIO_PLATFORM_BASE_URL=$NMP_BASE_URL` set |
| `Port 8000 already in use` at victim start | another war-game, or a stale port-forward from a previous run | `lsof -iTCP:8000 -sTCP:LISTEN`, kill the holder, or pass `--port` |
| `nemo agents create` fails on upload size | run state or venvs staged next to `agent.yaml` (the upload is capped at ~900 KB / 500 files) | register from a clean directory holding only the agent's own files |

## Victim dies or times out mid-run

| Symptom | Cause | Fix |
|---|---|---|
| Attacks report unblocked while the victim answered 502, container SIGKILLed | Docker VM out of memory — each in-flight Fabric session holds ~160 MB | give the Docker VM >= 8 GB; Agent Hardener already injects the session-reclaim flags, but an undersized VM still OOMs |
| `ReadTimeout` kills the attacker on one long turn | a single agent turn looped past the per-turn cap (`garak.request_timeout`, default 600 s) | bound the agent itself with `runtime.max_turns` in `agent.yaml` (maps to LangGraph's recursion limit); raising the timeout only moves the wall |
| Benign requests unscored | usually the same memory/timeout cause as above | fix the victim first, then re-run |

## Instrumentation

| Symptom | Cause | Fix |
|---|---|---|
| `VictimNotInstrumentedError` (any variant) | Relay not attached, not started, or not in the tool path | [relay-attachment.md](relay-attachment.md) — the error text maps to the exact missing obligation |
| Derived egress `(none)` for an agent with remote tools or a remote model | the config names no hosts, so discovery had nothing to read | pass `--egress host[:port]` per legitimate host (bare host = 443 only), then `refresh` |

## Replay and filesets

| Symptom | Cause | Fix |
|---|---|---|
| `Fileset name is required` (or the ref is not found) on `--replay-hitlog` | a bare name or file path was passed where a `<workspace>/<fileset>` reference is expected | `nemo files filesets create` first, `nemo files upload` into it, then reference `workspace/name` — upload does **not** auto-create a named fileset |

## Results read wrong

| Symptom | Cause | Fix |
|---|---|---|
| Run ends `validation_failed` | some attacks still got through the new guardrails — a result, not a crash | read the scorecard; iterate with `--replay-hitlog` |
| Scorecard shows blocks but the user expected guardrail hits | many "blocks" are the model refusing on its own, not the guardrail firing | check the ATOF trail — guardrail refusals carry the `agent-hardener[rail]` signature |
