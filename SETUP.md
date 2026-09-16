---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-setup
description: Set up a local NeMo Platform (`make bootstrap` + `nemo setup`) — services, providers, plugins, default/fast models, and an optional demo agent. Use when the user asks to install, bootstrap, set up, run, or start a local NeMo Platform.
version: "0.1"
---

# NeMo Platform Setup

Get a local NeMo platform running on `localhost:8080`. Work through the prereq questions below before bootstrapping — they shape which services start and what state survives the run. When setup is finished, [What's next?](#whats-next) maps the user's stated goal to the right follow-up skill.

> This document is the canonical setup guide. It lives at the repository root as `SETUP.md`. Unlike the other skills, it is **not** installed by `nemo skills install` — it has to be available before the platform is bootstrapped, when the CLI may not yet exist.

## Question 1 — Is a NeMo platform already running locally?

Before starting `nemo services run`, check for an existing instance:

```bash
lsof -iTCP:8080 -sTCP:LISTEN
ps -ef | grep "nemo services run" | grep -v grep
```

Alternatively, probe the API with `nemo workspaces list` — a workspaces table back means the platform is up. **Don't `curl /v1/workspaces`** to check; that path returns 404 even when the platform is healthy.

If port 8080 is in use **or** a `nemo services run` process exists, do not silently start a new one — the new instance will fail to bind, and it gets confusing which platform is answering requests (the controller piece can still partially start and connect to the old API on `:8080`, producing logs that look like a fresh boot but are reading state from the old DB). Surface the exact PIDs and command line, then ask which path the user wants:

- **(a) Kill the running platform and start fresh.** Use this when the running process is stale (long-running, crashed, or pre-dates the changes the user is now making). Procedure:
  1. SIGTERM the exact PIDs — `kill <pid> [<pid>...]`
  2. Wait ~10s; if still alive, **re-verify** each PID still matches a `nemo services run` command (paranoid check against PID reuse), then SIGKILL only those exact PIDs.
  3. Only **after** the processes are gone, wipe the DB (see Question 3).
- **(b) Keep it running and skip setup.** Use the running platform as-is — skip Switchyard install / DB wipe / startup entirely and proceed straight to your task. Caveats:
  - Middleware plugins like `nemo-switchyard` are loaded at platform **startup**, so a VirtualModel that needs middleware the running platform did not load will fail. Confirm the loaded plugins by grepping the platform log for `Loaded inference middleware plugin` before assuming a middleware is available. The first VirtualModel create with `nemo-switchyard` either succeeds or returns `422 references unknown plugin 'nemo-switchyard'`; a 422 means restart is needed.
  - The running services should include what the task needs: `inference-gateway`, `secrets`, `models`, `entities`, plus `guardrails` if the task uses rails. There is no public "list running services" endpoint — infer from feature attempts (`nemo guardrail configs list` returning 404 means the guardrails service isn't loaded).
  - The user may already have the workspace / secret / provider seeded — check with `nemo workspaces list`, `nemo secrets list --workspace …`, and `nemo inference providers list --workspace …` before re-creating anything (creates may 409).
- **(c) Abort and let the user investigate.**

> ⚠️ **macOS unlinked-inode gotcha:** running `rm -rf ~/.local/share/nemo` while a `nemo services run` process or its managed ClickHouse container is still alive does **not** reset state safely. A process can keep writing to the unlinked inode. Always stop the platform and remove its managed ClickHouse container **before** wiping the data directory.

## Question 2 — Local data directory?

Confirm where the user wants local platform state (entity-store DB, encryption key, files-service uploads) persisted. Use the same prompt language as `nemo setup`:

> **Local data directory:** `~/.local/share/nemo`

Most users accept the default. Override paths follow XDG conventions:

1. **`$NMP_DATA_DIR`** (most explicit) — used as-is, no `/nemo` suffix appended.
2. **`$XDG_DATA_HOME/nemo`** — if `XDG_DATA_HOME` is set in the shell.
3. **`~/.local/share/nemo`** — the default.

If the user picks a custom path, export it before starting services so the spawned platform inherits it:

```bash
export NMP_DATA_DIR=/custom/path/to/state
```

`nemo setup` persists the choice to `~/.config/nmp/config.yaml` under `local_services.data_dir` and re-uses it on subsequent runs. If you're running services manually (not via `nemo setup`), set `NMP_DATA_DIR` yourself each session.

## Question 3 — Wipe local platform data?

Ask whether the user wants to wipe local platform data before startup. This is a destructive operation that requires explicit confirmation. Warn clearly that it deletes the entity-store database, encryption key, files, job history, secrets, and Intake ClickHouse traces stored under the selected platform data directory. An explicitly configured ClickHouse data directory outside it is preserved. Providers and secrets must be re-seeded afterward. If the database and encryption key get out of sync, later runs can fail with decryption errors such as `cryptography.exceptions.InvalidTag`. **Stop every `nemo services run` process before wiping** (see the macOS gotcha under Question 1), and remove the managed ClickHouse container before deleting its bind-mounted data. If the user confirms, run this before `nemo services run`:

```bash
DATA_DIR="${NMP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nemo}"
case "$DATA_DIR" in
  ""|"/"|"$HOME"|"$HOME/"|"."|"./"|"$PWD"|"$PWD/")
    echo "REFUSING_UNSAFE_DATA_DIR: '$DATA_DIR' — abort"; exit 1 ;;
esac
lsof -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 && { echo "PLATFORM_STILL_RUNNING — abort before wipe"; exit 1; }
CLICKHOUSE_DATA_DIR="${NMP_INTAKE_CLICKHOUSE_DATA_DIR:-$DATA_DIR/intake-clickhouse}"
if [ -f "$CLICKHOUSE_DATA_DIR/.nmp-clickhouse-identity" ]; then
  NMP_DATA_DIR="$DATA_DIR" uv run python -m nmp.intake.local_clickhouse --remove || {
    echo "CLICKHOUSE_CONTAINER_CLEANUP_FAILED — start Docker and retry; data was not deleted"
    exit 1
  }
fi
rm -rf "$DATA_DIR"
```

Replace the path with whatever was chosen in Q2 (`$NMP_DATA_DIR`,
`$XDG_DATA_HOME/nemo`, or the default `~/.local/share/nemo`). The cleanup command
validates and removes only the managed container, restoring host ownership when
its data lives under the platform data directory. If cleanup fails because Docker
is unavailable, start Docker and retry—do not proceed to `rm -rf`.

---

## Bootstrap and start

This section is the **source checkout** path: use it to work on NeMo Platform itself, on a local plugin, or on Studio assets. To only *use* the platform, install the published wheel instead — `uv tool install "nemo-platform[all]"` needs no checkout and no toolchain, then continue at `nemo setup`.

The steps below cover prerequisites install, service startup, provider registration, default/fast model selection, and demo agent deployment in one shot. Prefer them over the manual sections further down whenever the task fits:

Before running `make bootstrap`, install Flox from the [Flox installation guide](https://flox.dev/docs/install-flox/install). Flox is the recommended source-development toolchain and does not need to be activated first. Contributors using a preinstalled host toolchain instead need the versions printed by `make toolchain-versions` and a C compiler; they must use `make TOOLCHAIN=system bootstrap`.

=== "Interactive"

```bash
make bootstrap           # installs Python deps, Studio assets, and plugins (including demo calculator agent)
flox -q activate         # enter the managed development environment
nemo setup               # interactive: prompts for provider, picks default/fast models, optionally deploys calculator-agent
```

=== "Non-interactive (CI)"

```bash
export NVIDIA_API_KEY=nvapi...
nemo setup --auto --start-services --install-skills --deploy-agent
```

`make bootstrap` is the umbrella for three finer-grained targets — use these if you only need a subset:

| Target | What it does |
| --- | --- |
| `make bootstrap-python` | Creates `.venv` and runs `uv sync` (Python deps + workspace packages) |
| `make bootstrap-studio` | Installs web deps via `pnpm` and builds Studio assets for FastAPI |

`make clean` removes the venv; `make clean-python` is the venv-only variant.

### Toolchain: uv, Node.js, pnpm

Flox pins all three — uv to satisfy `pyproject.toml`'s `required-version`, Node.js to match `.nvmrc`, and pnpm to match `web/package.json`. `make` invokes tools through Flox, so the pinned versions win over whatever is already on your PATH. Nothing is written to your shell rc.

**Run `make bootstrap` before any other `make` target.** Targets such as `make test-unit`, `make update-licenses` and `make refresh-openapi` call uv through Flox. On a machine that does not have Flox yet, they fail like this:

```text
/bin/sh: flox: command not found
make: *** [test-unit] Error 127
```

`make bootstrap` fixes it, as does `make verify-toolchain` on its own. CI uses `TOOLCHAIN=system` only after it has provisioned the matching toolchain explicitly.

That covers the `make` targets only. To run `uv` or `pnpm` directly, activate Flox first:

```bash
flox -q activate
```

Without Flox, install the versions printed by `make toolchain-versions` and a C compiler, then run `make TOOLCHAIN=system bootstrap`. Docker is required when starting local services, but not for dependency bootstrap.

If `nemo setup` is too high-level for the task (e.g. debugging startup, custom service set, custom plugin install after bootstrap), use the manual sections below.

### Default and fast model selection

`$NEMO_DEFAULT_MODEL` and `$NEMO_FAST_MODEL` **must be hyphenated entity IDs** from `nemo models list` (e.g. `nvidia-llama-3-3-nemotron-super-49b-v1-5` or `default/nvidia-llama-3-3-nemotron-super-49b-v1-5`). The slash-with-dots form (`nvidia/llama-3.3-nemotron-super-49b-v1-5`) is the upstream catalog's `served_model_name` — it's shown for human display but the gateway rejects it as a request input.

`nemo setup --auto` picks the default model in this order:

1. `$NEMO_DEFAULT_MODEL` (if set) — used as-is. The user may have exported this from a previous session; it takes precedence over anything discovered from the registered provider.
2. Otherwise, the first model entity returned by provider discovery.

The fast model uses `$NEMO_FAST_MODEL` when set and otherwise reuses the effective default. Interactive setup prompts for both and defaults the fast choice to the selected default. Existing contexts without `fast_model` also reuse `default_model`.

If the user is surprised by which model got picked, check `NEMO_DEFAULT_MODEL` and `NEMO_FAST_MODEL` first — that's the most common cause.

The first-discovered fallback is intentionally simple — providers like NVIDIA Build expose dozens of models and "first one" rarely matches the user's intent. If the user wants specific models (or wants to compare options before committing), don't rely on the `--auto` fallback. After setup finishes, the `inference` skill's "Step 2 — Discover available models" enumerates entity IDs and shows the jq filters for picking one out by vendor or family. The user can then pin the pair via `NEMO_DEFAULT_MODEL` and `NEMO_FAST_MODEL`, or override `body["model"]` per request.

If `nemo agents invoke …` fails with HTTP 422 in under a second on the first call, the cause is almost always a slash-with-dots model name reaching the gateway (e.g. via a stale `NEMO_DEFAULT_MODEL` or an agent config that hardcoded the upstream catalog form). The `inference` skill's "Common failure: HTTP 422 from chat completion" subsection has the diagnose-and-recover steps — don't conclude the platform is broken.

### Starting the platform (without Switchyard)

If `make bootstrap` has already run, just start the services. `nemo setup` does this for you; the manual equivalent is:

```bash
uv run nemo services run \
  --services entities,models,inference-gateway,secrets \
  --controllers models
```

### Starting the platform with Switchyard middleware

`make bootstrap-python` and bare `uv sync` install the `nemo-switchyard-plugin` distribution (path `plugins/nemo-switchyard`) through the root workspace's `enabled-plugins` group. The Switchyard library is vendored in-tree at `plugins/nemo-switchyard/vendor/switchyard/` (a snapshot pinned in `tool.uv.sources`) — no separate Switchyard checkout, `SWITCHYARD_PATH` env var, or PyPI workaround is needed. Do not `uv add` upstream `nemo-switchyard` into the workspace lock: that replaces May `switchyard` and breaks translation. After the plugin dist rename, a **throwaway** native overlay is `uv pip install --no-deps plugins/nemo-switchyard` then `uv pip install "git+https://github.com/NVIDIA-NeMo/Switchyard.git@v0.3.0-rc.2"` — do not reinstall the plugin `--no-deps` after rust, and do not `uv sync` unless you intend to restore May. Native `stage_router` / `llm_classifier` also need the IGW host from NVIDIA-NeMo/nemo-platform#2091. Start with debug logging to see routing decisions:

```bash
# Start with LOG_LEVEL=DEBUG to see routing decisions.
LOG_LEVEL=DEBUG uv run nemo services run \
  --services entities,models,inference-gateway,secrets \
  --controllers models
```

`nemo-switchyard` is auto-discovered via its `nemo.inference_middleware` entry point once dependencies are installed.

### Local ClickHouse for Intake

When the `intake` service is selected and `NMP_INTAKE_CLICKHOUSE_URL` is unset, Intake automatically
provisions a ClickHouse container owned by the resolved NeMo data directory, with a Docker-assigned
loopback port. A platform process reuses the container for that data directory; graceful shutdown
stops it without removing it, while hard process termination can leave it running. Only the explicitly
confirmed reset in Question 3 or teardown options 2/3 delete its
default data under the NeMo data directory, after removing the managed container. A separately
configured `NMP_INTAKE_CLICKHOUSE_DATA_DIR` is preserved. Run only one active local platform instance
per data directory; stopping it also stops that directory's managed ClickHouse container.

Docker must already be running. If startup logs report `Docker daemon is unavailable`, start Docker
Desktop on macOS/Windows or the Docker service on Linux, then rerun `nemo setup` or restart
`nemo services run`. To use an external ClickHouse and bypass local Docker provisioning, export
`NMP_INTAKE_CLICKHOUSE_URL` before starting the platform.

### Demo agent

`make bootstrap` installs the NeMo agents plugin and the calculator-agent example through the root workspace, so no separate `uv pip install` is needed. After services start, `nemo setup` (or `nemo setup --auto --deploy-agent`) will deploy a demo `calculator-agent` in the default workspace. Verify with:

```bash
nemo agents list
nemo agents invoke --agent calculator-agent --input "What is 12 * 8?"
```

### Local platform environment summary

- **Port**: `8080` (CLI default — do NOT pass a custom `--base-url`).
- **`export NMP_BASE_URL=http://localhost:8080` — required when targeting a local platform.** If your `~/.config/nmp/config.yaml` already points at a remote cluster, the CLI uses that base URL and ignores the local platform entirely. Setting this env var overrides the config file for the current shell session.
- **Reset state:** follow Question 3 above; it requires explicit confirmation and removes the managed
  ClickHouse container before deleting platform data.

---

## What's next?

The platform is running. Don't leave the user with "you're good to go" — offer a menu of what they can do next based on what they originally asked for. Match the user's intent to one of the patterns from the [README's "Coding agent integration" section](/README.md#coding-agent-integration):

| User says… | Goal | Follow-up skill |
| --- | --- | --- |
| "Optimize my agent", "my agent is too slow / using too many tokens" | Cost / latency optimization via routing or skill tuning | `nemo-agents-optimize` |
| "Secure my agent", "my agent is producing dangerous output" | Content safety / red-team / leak audit | `nemo-agents-secure`, `nemo-guardrails`, `nemo-auditor` |
| "Can my agent use multiple models?", "split traffic across N backends" | Multi-backend routing via Switchyard | (inline; see `inference` skill) |
| "Evaluate my model / agent on \<benchmark\>" | Eval against a dataset / harness | `nemo-evaluator`, `evaluator-plugin` |
| "Generate synthetic data", "I have sensitive data and need…" | Data generation / anonymization / safe synthesis | `data-designer`, `nemo-anonymizer`, `nemo-safe-synthesizer` |
| "Just deploy / invoke an agent" | Deploy the demo calculator agent or your own | `nemo-agents-optimize` (later, if needed) |
| "Chat with a model", "call \<model\> via inference" | Plain inference through IGW | `inference` skill |
| "Register an inference provider" (no further use case) | Provider registration only | `inference` skill |

If the user's prompt doesn't already pin one down, ask: *"The platform is up. What would you like to do next — optimize an agent, deploy one, run inference, evaluate, generate data, or something else?"*

If the user wants to **pick or swap the default/fast model pair**, don't guess — hand off to the `inference` skill. Step 2 there enumerates `served_models[].model_entity_id` and shows jq filters for picking by vendor / family. To pin the choices for subsequent runs, export `NEMO_DEFAULT_MODEL=<workspace>/<entity-id>` and `NEMO_FAST_MODEL=<workspace>/<entity-id>` before the next `nemo setup --auto`; fast falls back to default when omitted. For one-off commands, pass the entity ID positionally: `nemo chat <entity-id>`.

### Available skills

`nemo skills list` is the canonical check for what's actually loaded — the entries depend on which plugins are installed. Expected entries after `make bootstrap`:

CLI built-ins (always present):

- **`inference`** — ModelProvider + VirtualModel + Switchyard reference. End-to-end inference flow, routing patterns, middleware ordering, troubleshooting.

> Setup itself (this document) lives at the repo root as `SETUP.md` rather than as a CLI-installable skill, since coding agents need it before the platform is bootstrapped.

Plugin-provided (appear once the plugin is installed):

- **`nemo-agents-optimize`** — optimize a deployed agent (routing splits, skill tuning, prompt tuning, evals against newer models). From `plugins/nemo-agents`.
- **`nemo-agents-secure`** — audit a deployed agent for missing guardrails, PII exposure, leaked secrets/keys. From `plugins/nemo-agents`.
- **`nemo-guardrails`** — guardrail config CRUD, content-safety rails, the `nemo-guardrails` middleware. From `plugins/nemo-guardrails`.
- **`nemo-auditor`** — vulnerability scanning, audit configs/targets/jobs, red-team probes. From `plugins/nemo-auditor`.
- **`nemo-evaluator`** / **`evaluator-plugin`** — metrics, sync/async evaluations, llm-judge, benchmark jobs. From `plugins/nemo-evaluator`.
- **`data-designer`** — synthetic dataset generation pipelines. From `plugins/nemo-data-designer`.
- **`nemo-entities`**, **`nemo-files`**, **`nemo-secrets`**, **`nemo-auth`**, **`nemo-inference-gateway`** — CLI references for the matching services.

### Installing skills into the coding agent on demand

`nemo skills list` lists every skill the **platform** can install — but that's not the same as what's currently loaded in the coding agent (Claude Code, Cursor, Codex, OpenCode). After `make bootstrap` finishes, the relevant subset must still be **installed into the coding agent** for it to actually use them.

Default flow (already wired into `nemo setup`):

```bash
nemo skills install --agent <claude|cursor|codex|opencode>
```

With no `--skill` flag, this installs **all** skills from `nemo skills list` into the chosen agent. Run it again any time `nemo skills list` changes (new plugin installed, plugin updated).

If the user's goal in the table above maps to a skill that you (the coding agent) don't currently have in this session, install only the skills you need rather than all of them:

```bash
# Single skill
nemo skills install --agent claude --skill nemo-agents-optimize

# Multiple skills
nemo skills install --agent claude --skill nemo-agents-secure --skill nemo-guardrails
```

Then invoke the freshly-installed skill — e.g. ask the user "I just installed the `nemo-agents-optimize` skill; want me to use it to walk through optimizing your agent now?"

If a goal-relevant skill is **missing from `nemo skills list` entirely**, the plugin that ships it isn't installed. Install the plugin first, then re-run `nemo skills install`:

```bash
uv pip install -e plugins/<plugin-name>
nemo skills install --agent <agent>            # picks up the new skill(s)
```
