<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# nemo-agent-hardener

Red-team and harden a NAT agent. Point Agent Hardener at **an agent registered in the platform** (it does
not have to be deployed) or at **a local NAT project directory**, and it runs an
**attack → defend → validate** war-game against a sandboxed copy: an attacker swarm probes the agent,
defenders generate guardrails and sandbox policy, and validators check that the attacks are now
blocked *and* that ordinary requests still work. Your running deployment is never touched.

Two ways to drive it — [the Studio UI](#run-it-in-the-ui) or [the CLI](#run-it-from-the-cli). Both
need the same one-time setup below.

---

## Quickstart

The whole path, from a bootstrapped repo to a finished war-game against the bundled example agent.
Assumes Docker and OpenShell are installed — if not, do [What you need](#what-you-need) first
(~15 min, mostly those two). Each step is explained in [One-time setup](#one-time-setup).

```bash
export INFERENCE_API_KEY=<your-nvapi-key>
export NMP_BASE_URL=http://localhost:8080

# 1. Start the platform (logs to a file — backgrounding alone still prints over your prompt)
uv run nemo services run --service-group all --controllers models,jobs \
  --host 0.0.0.0 --port 8080 > /tmp/nemo-platform.log 2>&1 &
until curl -sf http://localhost:8080/health/ready >/dev/null; do sleep 2; done; echo ready

# 2. Give it a model provider  (409 "already exists" just means you've run these before)
printf '%s' "$INFERENCE_API_KEY" | \
  uv run nemo secrets create nvidia-inference-key --from-file - --workspace default
uv run nemo inference providers create nvidia-inference --workspace default \
  --host-url "https://inference-api.nvidia.com/v1" \
  --api-key-secret-name nvidia-inference-key

# 3. Register the example agent. NEMO_DEFAULT_MODEL must be set *now* — it is baked into
#    the stored config. Use a model entity name from `nemo models list`, not a provider id.
uv run nemo models list --workspace default | grep nemotron
export NEMO_DEFAULT_MODEL=nvidia-nvidia-nemotron-3-nano-30b-a3b   # example — use what you saw
uv run nemo agents create --name react-agent \
  --agent-config plugins/nemo-agents/examples/react-agent/react-agent.yml

# 4. War-game it
uv run nemo agent-hardener setup                                      # once per machine
uv run nemo agent-hardener doctor                                     # everything should be green
uv run nemo agent-hardener init --agent react-agent                   # save a target
uv run nemo agent-hardener synth-benign --manifest-id react-agent     # interview — you answer
uv run nemo agent-hardener run --manifest-id react-agent              # attack → defend → validate
uv run nemo agent-hardener status --limit 5
```

Three things that trip people up, each covered in full below:

- **`synth-benign` is mandatory.** `run` only *consumes* a benign suite and never generates one;
  without one it fails immediately. Add `--yes` to accept the interview's suggested answers, or
  `--no-interactive` in CI. [More](#run-it-from-the-cli)
- **`--egress <host>` is how the agent's tools reach the internet.** The sandbox drops anything not
  allow-listed, and a blocked tool looks like a *passing* run because the model answers from its own
  knowledge. Omitted above because `react-agent`'s `current_datetime` needs no network, and its
  `wiki_search` is broken upstream regardless (see the egress note in
  [One-time setup](#one-time-setup)). Your own agent almost certainly needs it.
  [More](#run-it-from-the-cli)
- **The Studio entry appears as soon as the plugin is installed** — the UI ships inside the plugin
  as a Studio bundle, so there is no flag to set. [More](#run-it-in-the-ui)

Got your own agent instead? `init --agent <name>` works for any registered agent, and
`init --project-dir <path>` war-games a local NAT project with nothing registered at all.

---

## What you need

Two environment variables. `nemo agent-hardener setup` installs agent-hardener itself, into its own venv.

```bash
export INFERENCE_API_KEY=<your-nvapi-key>
export NMP_BASE_URL=http://localhost:8080
```

<details>
<summary><b>Installing agent-hardener from a private index</b> — temporary, until it's on PyPI</summary>

Agent Hardener isn't on public PyPI yet, so `setup` has to be pointed at the index hosting it. Two
steps, and the first is done once per machine.

**1. Store your index credentials in `~/.netrc`.** Private indexes require authentication and the
plugin deliberately doesn't handle it — uv reads credentials from your environment. `~/.netrc` is
the least friction: set once, applies to every shell and every tool, and no secret ever appears in a
command line or your shell history.

```bash
cat >> ~/.netrc <<'EOF'
machine <index-host>
  login <username>
  password <access-token>
EOF
chmod 600 ~/.netrc
```

**2. Point `setup` at the index.**

```bash
export NEMO_AGENT_HARDENER_INDEX_URL="<index-url>"
```

Must be `https://` — `setup` refuses a plaintext index so credentials never cross the wire unencrypted
(`http://localhost`/`127.0.0.1` is allowed for local development).

That's all you need. The index is *additional* to PyPI, so agent-hardener resolves from it and every
dependency still comes from PyPI. (If a dependency fails to resolve, see the troubleshooting below —
don't set `NEMO_AGENT_HARDENER_INDEX_STRATEGY` pre-emptively, it weakens dependency-confusion protection.)

Ask whoever publishes agent-hardener for the index host, URL, and how to get a token — or see
agent-hardener's own README.

*Alternatives to `~/.netrc`, if you can't use it:*

- **`UV_INDEX_<NAME>_USERNAME` / `UV_INDEX_<NAME>_PASSWORD`** — these key off the index *name*, so
  you must use uv's named form `NEMO_AGENT_HARDENER_INDEX_URL="<name>=<url>"`. With a bare URL uv
  generates its own name and the credentials silently never apply. Needs re-exporting per shell.
- **Credentials embedded in the URL** (`https://<user>:<token>@<host>/...`) — works, but puts your
  token in shell history and anywhere the URL is echoed. `doctor` masks it, but prefer `~/.netrc`.

| Variable | Purpose |
|---|---|
| `NEMO_AGENT_HARDENER_INDEX_URL` | Extra index to resolve agent-hardener from, `<url>` or `<name>=<url>` |
| `NEMO_AGENT_HARDENER_INDEX_STRATEGY` | uv `--index-strategy`; `unsafe-best-match` when the index shadows PyPI packages |

**Check access in seconds, before installing anything.** uv cannot tell you which of these is wrong —
it reports an unauthorized index as an empty one — so ask the index directly:

```bash
curl -sS -n -o /dev/null -w '%{http_code}\n' "<index-url>/agent-hardener/"
```

| Code | Meaning |
|---|---|
| `200` | Ready — go run `setup`. |
| `401` | Credentials wrong or missing. Most often a **truncated token**: registry UIs display tokens elided, so copy with the button, not by selecting text. Check with `echo "${#TOKEN} chars"`. Also confirm the `~/.netrc` machine matches the index host, and that the username is the one the registry issued (often a service account, not your own). |
| `403` | Authenticated, but the token isn't scoped to this repository. |
| `404` | Authenticated, but agent-hardener isn't published in this repository — check the URL. |

**If `setup` still fails:**

- **`agent-hardener was not found in the package registry`** — this is almost always **authentication**, not
  a missing package. Run the `curl` check above; a `401` confirms it. It also appears when
  `NEMO_AGENT_HARDENER_INDEX_URL` is unset, which is what a default `setup` reports.
- **A dependency that exists on PyPI won't resolve** — the index carries a package shadowing its PyPI
  counterpart. Set `NEMO_AGENT_HARDENER_INDEX_STRATEGY=unsafe-best-match`.

> **Delete this whole section once agent-hardener is on PyPI.** Nothing else in this README, and no code,
> refers to it — plain `nemo agent-hardener setup` already installs from PyPI with no index
> configuration and no credentials.
</details>

**Docker + OpenShell**, which run the victim sandbox:

```bash
brew install colima docker && colima start          # or Docker Desktop

# Use the native installer — `uv tool install openshell` gives you the CLI
# without the gateway service, and the war-game needs the gateway. Pinned to a
# release tag so the script can't change under you; matches the openshell>=0.0.92
# the deployments plugin requires.
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/v0.0.92/install.sh | sh
openshell status                                     # expect "Status: Connected"
```

<details>
<summary>macOS: point OpenShell at Docker so sandboxes can reach the host</summary>

```bash
DOCKER_SOCK=$(docker context inspect --format '{{.Endpoints.docker.Host}}')
brew services stop openshell
launchctl setenv OPENSHELL_DRIVERS docker
launchctl setenv DOCKER_HOST "$DOCKER_SOCK"
brew services restart openshell
openshell status
```
</details>

---

## One-time setup

```bash
cd /path/to/nemo-platform
make bootstrap                       # Python deps + Studio assets

uv run nemo agent-hardener setup         # creates ~/.agent-hardener/venv and ~/.agent-hardener/garak-venv
uv run nemo agent-hardener doctor        # preflight — everything should be green
```

Start the platform:

```bash
# models  → discovers served models (without it, providers show 0 models)
# jobs    → runs the war-game
# 0.0.0.0 → lets the sandbox reach the Inference Gateway via host.docker.internal
# logs go to a file — backgrounding alone still prints them over your prompt
uv run nemo services run --service-group all --controllers models,jobs \
  --host 0.0.0.0 --port 8080 > /tmp/nemo-platform.log 2>&1 &

until curl -sf http://localhost:8080/health/ready >/dev/null; do sleep 2; done; echo ready
```

Watch it with `tail -f /tmp/nemo-platform.log`; stop it with `uv run nemo services stop`.

Register an inference provider. Both commands fail with `409 already exists` if you've run them
before — that's harmless, skip to the next step. To start from a clean platform instead, stop it
(`nemo services stop`), `rm -rf ~/.local/share/nemo`, and restart; the stop must come first or the
wipe silently doesn't take effect.

```bash
printf '%s' "$INFERENCE_API_KEY" | \
  uv run nemo secrets create nvidia-inference-key --from-file - --workspace default

uv run nemo inference providers create nvidia-inference --workspace default \
  --host-url "https://inference-api.nvidia.com/v1" \
  --api-key-secret-name nvidia-inference-key
```

Register an agent to attack. This example ships with the repo and needs no extra packages:

```bash
# The config references ${NEMO_DEFAULT_MODEL}. It must be a *model entity* name the platform
# discovered — not a provider model id. Entity names are lowercase-and-hyphens only; a slash gets
# rejected by the Inference Gateway with "Invalid model".
uv run nemo models list --workspace default | grep nemotron      # pick one
export NEMO_DEFAULT_MODEL=nvidia-nvidia-nemotron-3-nano-30b-a3b  # example — use what you saw

uv run nemo agents create --name react-agent \
  --agent-config plugins/nemo-agents/examples/react-agent/react-agent.yml
```

> **Egress.** The victim sandbox blocks outbound traffic unless the manifest allow-lists it, and
> agent-hardener can only auto-discover hosts by scanning a project's source — a config-only agent keeps
> its tool hosts in packaged code, so you must declare them. Entries are `host[:port]` and a bare
> host opens **443 only**; a tool using plain HTTP needs `host:80` too. Without this the victim's
> calls are dropped and tool-using attacks silently no-op while the run still reports success.
>
> `react-agent`'s `wiki_search` is a known exception: it fails even with egress open, because the
> `wikipedia` package sends no User-Agent and Wikimedia now rejects that
> ([T400119](https://phabricator.wikimedia.org/T400119)). Upstream, not agent-hardener. Its other tool,
> `current_datetime`, needs no network and exercises the tool path fine.

> `NEMO_DEFAULT_MODEL` must be set **when you run `agents create`** — the config references it as
> `${NEMO_DEFAULT_MODEL}` and the platform resolves it into the stored agent. Register it unset and
> the victim later starts with an unresolved model name. It is not needed afterwards; Agent Hardener
> reads the already-resolved config.

---

## Run it in the UI

The UI ships **inside this plugin** as a Studio plugin bundle, so there is no feature flag to set:
installing the plugin is what puts it in Studio. Studio discovers it through the `nemo.studio`
entry point, serves the bundle at `/plugin-ui/agent-hardener/index.js`, and renders it inside its own
React tree.

Start the platform as usual:

```bash
uv run nemo services run --service-group all --controllers models,jobs \
  --host 0.0.0.0 --port 8080 > /tmp/nemo-platform.log 2>&1 &
```

Open **http://localhost:8080/studio/** → **Governance → Agent Hardener**. If the entry is missing,
confirm the plugin is installed (`curl -s localhost:8080/apis/plugins`) and hard-reload (⌘⇧R).

1. **Manifests → New Manifest** — pick `react-agent`, accept the detected port and secrets, add any
   **egress** hosts the agent calls (same rule as the CLI, see the warning above), **Create**.
2. **Run war-game** on the manifest. The run opens on its **Swarm** tab.
3. Watch the graph light up per phase, with the live agent feed beside it. Click any node for its
   prompts and LLM calls.
4. When it finishes, **Harden** lists each recommended defense with the attack that motivated it.
   Toggle the ones you want → **Sanity check** replays the attacks against just that selection and
   reports what it blocks and any ordinary requests it breaks → **Apply to Agent** writes the
   hardened workflow back onto the agent config.

Redeploy the agent for the guardrails to take effect.

---

## Run it from the CLI

```bash
uv run nemo agent-hardener init --agent react-agent                  # manifest + saved entity
uv run nemo agent-hardener synth-benign --manifest-id react-agent --yes   # required, see below
uv run nemo agent-hardener run --manifest-id react-agent             # attack → defend → validate
uv run nemo agent-hardener status --limit 5                          # recent runs
```

`init` only needs the agent **registered**, not deployed. It saves a reusable manifest named after the
agent — that name is the `--manifest-id` every later command takes — and writes `agent-hardener.yaml` as a
*rendering* you can read. Editing that file has no effect; the run uses the saved manifest.

**A manifest is a frozen target.** `init` resolves your agent once and stores the result, so every run
war-games the same thing — which is what makes two runs comparable, and what a "did the hardening
help?" answer depends on. Editing the agent afterwards (new model, new tool, redeploy) does **not**
change an existing manifest. Take those changes deliberately:

```bash
uv run nemo agent-hardener refresh --manifest-id react-agent
```

Your egress, secrets, models, defenders and cached benign suite are all preserved; only the target
itself is rebuilt. You don't need this after `apply-mitigation` — applying a hardened workflow
refreshes the manifest for you, so `run → harden → apply → run again` measures the change you just
made.

If your agent calls the internet, allow-list the hosts at init time — the sandbox drops everything
else, and a blocked tool usually looks like a working run because the model answers from memory:

```bash
uv run nemo agent-hardener init --agent react-agent --egress en.wikipedia.org
```

A bare host opens **443 only**; write `host:80` for plain HTTP. Hosts can't be auto-discovered for a
config-only agent, since its tool code lives in an installed package rather than in your project.

If the agent reads non-secret environment variables — a host-backend URL, a feature flag — set them
at init too:

```bash
uv run nemo agent-hardener init --agent react-agent --env BACKEND_URL=http://host.docker.internal:8086
```

`--env` is repeatable and only the first `=` splits, so values may contain `=`. **Keep credentials out
of it**: `--env` values are stored in plaintext on the manifest, whereas `--secrets` stores only the
*names* and resolves the values from the platform Secrets store when the run starts.

**`synth-benign` is not optional.** The war-game validates two things: that attacks are blocked, and
that ordinary requests still work. Those ordinary requests are the *benign suite*, and `run` is a
pure consumer of it — it never generates one. Without a suite it fails immediately with
`smart-benign validation requires an explicit benign suite`. Generate it once and it's cached on the
manifest for every later run:

```bash
uv run nemo agent-hardener synth-benign --manifest-id react-agent          # interview, you answer
uv run nemo agent-hardener synth-benign --manifest-id react-agent --yes    # interview, defaults accepted
uv run nemo agent-hardener synth-benign --manifest-id react-agent --no-interactive   # CI: rules only
```

### War-game a local NAT project

No deployed agent needed — point `init` at the project directory instead:

```bash
uv run nemo agent-hardener init --project-dir ~/my-nat-agent      # asks about workflow, port, secrets
uv run nemo agent-hardener init --project-dir ~/my-nat-agent --yes  # CI: accept detected answers
```

This runs `agent-hardener init` in your terminal so you answer its questions directly, then uploads the
project and saves the result as a manifest. From there it's the same `--manifest-id` flow as above.
`--workflow`, `--port`, `--egress` and `--secrets` pre-answer individual prompts.

Your project supplies its own dependencies, so its `pyproject.toml`/`requirements.txt` must include
`nvidia-nat` (plus whatever your tools import).

Prefer `run --manifest-id` over `run --config`: the cached suite is looked up by manifest id, so the
`--config` form needs you to pass `--benign-suite <csv>` yourself.

After a run produces mitigations, freeze a chosen subset and replay the recorded attacks against it:

```bash
uv run nemo agent-hardener sanity-check --manifest-id react-agent \
  --mitigations mitigations.json --replay-hitlog <fileset-ref> --keep custom_guardrail_1
```

---

## Troubleshooting

**Agent Hardener missing from the Studio side nav.** The UI ships with the plugin, so this means
Studio did not load its bundle. Check the plugin is registered (`curl -s localhost:8080/apis/plugins`
should list `agent-hardener` with a `bundleUrl`) and that the bundle is served
(`curl -sI localhost:8080/plugin-ui/agent-hardener/index.js`). Then hard-reload the browser.

**`Missing required secrets: <NAME>`.** The agent config references `${<NAME>}` and nothing provides
it. Agent Hardener derives required secrets from the agent's *stored* config, so this means the variable
was unset when the agent was registered. Export it and re-run `nemo agents create`, or supply it via
`--env-file`.

**The victim never becomes healthy, and its log shows a pydantic `union_tag_invalid` for a
`_type`.** The agent references a component the scaffolded victim can't resolve — it installs only
`nvidia-nat[langchain]`, not the platform's NAT plugins. Platform telemetry (`nemo_files`) is
stripped for you; anything else means the agent needs a real project, so pass `--project-dir` to
`init` and let its `pyproject.toml` supply the dependency.

**The victim is healthy but every request 422s with `Invalid model`.** The workflow's `model_name`
isn't a model entity the platform knows. Entity names are lowercase letters, digits and hyphens —
a provider id like `nvidia/nemotron-3-nano-30b-a3b` is rejected for the slash. Check
`nemo models list --workspace default`, then re-register the agent with that exact name (the value
is baked in at `agents create` time).

**`openshell status` says "connection refused" or "no compute driver" (macOS).** The gateway is up but
has no driver — apply the Docker driver block above, then re-run `nemo agent-hardener setup` to
re-register the `auto-defender` gateway.

**OpenShell installed but no gateway.** You installed via `uv tool install openshell`, which is
CLI-only. `uv tool uninstall openshell`, then use the curl installer above.


**Provider shows 0 served models / inference 404s.** Start the platform with `--controllers models`.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `NEMO_AGENT_HARDENER_SPEC` | `agent-hardener` | Package spec `setup` installs. Override to pin a version (`agent-hardener==0.0.2`) or to develop against a local checkout |
| `NEMO_AGENT_HARDENER_VENV_PATH` | `~/.agent-hardener/venv` | agent-hardener venv |
| `NEMO_AGENT_HARDENER_GARAK_VENV_PATH` | `~/.agent-hardener/garak-venv` | garak (attacker) venv |
| `NEMO_AGENT_HARDENER_DEFAULT_WORKSPACE` | `default` | Workspace used by CLI commands |
| `NEMO_AGENT_HARDENER_REQUIRE_SANDBOX` | `true` | Fail `run` when Docker/OpenShell aren't ready |
| `NEMO_AGENT_HARDENER_OPERATOR_ENV_FILE` | `~/.agent-hardener/.env` | Dotenv the war-game subprocess reads |
| `NEMO_AGENT_HARDENER_INDEX_URL` | unset | Extra package index `setup` resolves agent-hardener from |
| `NEMO_AGENT_HARDENER_INDEX_STRATEGY` | unset | uv `--index-strategy` for that install |

All `NEMO_AGENT_HARDENER_*` values can also be set via Helm `platformConfig.agent_hardener.*`.

---

## How it works

agent-hardener and garak run in **their own venvs**, invoked by subprocess and never imported. The
conflicting closure is *garak's* — it pulls `litellm → httpx>=0.28` plus `torch`, against the
platform's `httpx~=0.27` — and agent-hardener keeps garak out of its own dependencies for the same
reason. Importing agent-hardener would remove neither that boundary nor the Docker sandbox it launches,
while permanently fusing both dependency graphs. Most traffic crosses as files (YAML manifests in,
JSON hitlogs out), with HTTP for live events and the human-in-the-loop interview.

`init --agent` resolves a registered agent into a manifest server-side — the same `POST /manifests`
Studio calls — reading the agent registry and injecting the Inference Gateway URL into its LLMs, so
the sandboxed victim needs no raw model key. `init --project-dir` instead runs agent-hardener's own
interactive `init` in your terminal, then uploads the project as a fileset the run re-downloads.
Either way the manifest is stored as an entity and then frozen: `init` resolves once and saves the
resulting scaffold as a fileset the run re-downloads, so two runs of one manifest hit the same
target — which is what a "did the hardening help?" answer depends on. Edits to a registered agent
reach an existing manifest only through `POST /manifests/{name}/refresh`, which `apply-mitigation`
calls for you. The stored settings (egress, secrets, port) are what persist — not the rendered YAML.

---

## Studio UI

The web UI lives in [`web/`](web/) and ships as a Studio plugin bundle: `src/index.ts` exports a
`Root` component and `navItems`, Studio renders `Root` inside its own React tree (its Router,
QueryClient and theme), and `studio.py` points Studio at the built
`src/nemo_agent_hardener_plugin/web/dist/index.js` through the `nemo.studio` entry point.

The contract and its rules — shared singletons, KUI, theme tokens, auth — are documented in
[`plugins/example-plugin/web/AGENTS.md`](../example-plugin/web/AGENTS.md), the canonical template.

```bash
cd plugins/nemo-agent-hardener/web
pnpm install
pnpm gen        # regenerate the API client from ../openapi/openapi.yaml
pnpm build      # emits ../src/nemo_agent_hardener_plugin/web/dist/index.js (shipped in the wheel)
pnpm typecheck && pnpm lint && pnpm test
```

Commit the rebuilt `dist/index.js` — it is the artifact the wheel installs, and the repo's global
`dist` ignore means it needs `git add -f`.

Two things to keep in mind when editing it:

- **Only use styling Studio already compiles.** Studio's Tailwind scans `web/packages/**`, so a
  utility class it does not already emit has no CSS once the code lives under `plugins/`. Nothing
  catches this — build, typecheck and tests all pass, and it shows only as unstyled UI. Use the
  semantic tokens in `src/theme.ts`.
- **Shared deps stay external.** `react`, `react-dom`, `react-router`,
  `@nvidia/foundations-react-core`, `@tanstack/react-query` and `@nemo/common` must remain bare
  imports in the built bundle so the browser resolves them to Studio's single instance:
  `grep -oE 'from *"[^"]+"' ../src/nemo_agent_hardener_plugin/web/dist/index.js | sort -u`.
