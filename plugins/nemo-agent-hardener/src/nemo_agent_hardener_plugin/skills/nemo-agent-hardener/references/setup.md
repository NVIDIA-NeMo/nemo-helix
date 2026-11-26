<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Setting up a clean VM for a war-game

Take a machine from nothing to a green `nemo agent-hardener doctor`: VM checks, NeMo Helix from
source, the Helix services, OpenShell, the model key, and Agent Hardener itself. Do the sections in
order; each check states the result to expect. Run the checks yourself, but **ask the user before**
anything that removes an install, deletes state, needs `sudo`, or needs their API key, and never
print the key.

## 1. VM requirements

| Item | Requirement |
|---|---|
| OS | Linux x86_64 |
| Memory | 16 GB RAM minimum, 32 GB recommended: the sandboxed agent alone needs about 8 GB, on top of the Helix services and the attacker |
| Disk | 50 GB free: the attacker's environment is about 6 GB, Helix about 2 GB, and each run adds Docker images and outputs |
| Software | Python 3.12–3.14, `uv`, Docker (daemon running, usable without `sudo`) |
| Key | One NVIDIA API key (`nvapi-...`, from build.nvidia.com), for the Helix model provider and for Agent Hardener's own models |

| Port | Used by |
|---|---|
| 8080 | NeMo Helix services (API and Studio) |
| 8000 | The agent under test. Agent Hardener builds it, runs it in an OpenShell sandbox and forwards it here during a run. It does not use an agent that is already running, so the user must not start one: keep the port free |
| 17670 | OpenShell gateway (held by OpenShell once it is installed) |

```bash
free -h                                          # Mem: available >= 16G
df -h ~                                          # Avail >= 50G
docker info > /dev/null && echo "Docker OK"      # Docker OK
sudo ss -ltnp '( sport = :8080 or sport = :8000 )'   # only the header line
```

- `permission denied` on the Docker socket: the user must run `sudo usermod -aG docker $USER`, then
  log out and back in. Any other Docker error means the daemon is not running.
- 8080 in use: confirm it is NeMo Helix with `nemo services status` (`Address:` on 8080, `PID:`
  matching `ss`). If it is a Helix instance that does not answer `/health/ready`, ask the user before
  stopping it (`nemo services stop`, or `--force` for one running in the foreground of a terminal).

## 2. NeMo Helix, from source

Helix must be installed **from source**: the PyPI package (`pip install` / `uv tool install
nemo-helix`) has no Agent Hardener plugin.

```bash
which nemo
nemo agent-hardener --help
```

| What you see | Meaning | Next |
|---|---|---|
| the help lists `setup`, `doctor`, `init`, `synth-benign`, `run`, `status` (the path is usually `<checkout>/.venv/bin/nemo`, whatever the checkout is called) | installed from source | skip to **Start the Helix services** |
| `No such command 'agent-hardener'` (for example from `~/.local/bin/nemo`) | a PyPI install, which has no Agent Hardener plugin | ask the user, then remove it, then install from source |
| nothing | not installed | install from source |

Remove a PyPI install:

```bash
nemo services stop                 # stop any running Helix services
uv tool uninstall nemo-helix       # if it was installed with uv
pip uninstall -y nemo-helix        # if it was installed with pip
hash -r
which nemo                         # prints nothing
```

Install from source. [Flox](https://flox.dev/docs/install-flox/install) provides the toolchain
versions Helix needs; if `flox` is missing, install it with `curl -fsSL https://get.flox.dev | sh`.

```bash
git clone https://github.com/NVIDIA-NeMo/nemo-helix.git
cd nemo-helix
make bootstrap                     # ends with "bootstrap completed"
flox -q activate
which nemo                         # ends in nemo-helix/.venv/bin/nemo
nemo agent-hardener --help         # lists setup, doctor, init, synth-benign, run, status
```

Every `nemo` command afterwards runs from this activated environment (in a new terminal:
`cd nemo-helix && flox -q activate`). `flox -q activate` opens an interactive shell; from a
non-interactive tool, run each command as `flox -q activate -- bash -c '<command>'`. Never
`apt install nemo`: that is an unrelated Linux file manager.

### Start the Helix services

The services run in the foreground; the user keeps that terminal open (or run it in the
background and keep it alive for the whole session).

```bash
export NHX_STUDIO_PLATFORM_BASE_URL=http://localhost:8080
nemo services run --service-group all --controllers models,jobs --host 0.0.0.0 --port 8080
```

Every flag is needed: `--controllers models` discovers the provider's models; `--controllers jobs`
runs Agent Hardener's jobs (without it a run stays in `created` forever); `--host 0.0.0.0` lets the
sandboxed agent reach Helix; `NHX_STUDIO_PLATFORM_BASE_URL` makes Studio work.

```bash
nemo services status               # Health:   healthy, Address: 0.0.0.0:8080
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8080/health/ready   # 200
```

## 3. OpenShell 0.1.4 or later

OpenShell runs the agent in a sandbox. It does not depend on Helix.

```bash
which openshell                    # a path that does not contain uv/tools
openshell --version                # 0.1.4 or later
openshell status                   # Status: Connected
```

- Up to date and connected: skip to lingering.
- Not installed: install it.
- Older than 0.1.4, or installed with `uv tool install` (path contains `uv/tools`): ask the user,
  then remove the old version first. Its sandboxes and state block or crash the new version:

```bash
openshell sandbox delete --all --gateway openshell
openshell sandbox delete --all --gateway auto-defender
systemctl --user disable --now openshell-gateway
dpkg -l | grep -i openshell        # then: sudo apt remove <package-name>
uv tool uninstall openshell
rm -f ~/.local/bin/openshell ~/.local/bin/openshell-gateway
rm -rf ~/.local/state/openshell ~/.config/openshell
which openshell                    # prints nothing
```

Install (CLI and gateway service; `uv tool install openshell` installs only the CLI):

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | OPENSHELL_VERSION=v0.1.4 sh
openshell --version                # 0.1.4
openshell status                   # Status: Connected
```

The gateway is a systemd *user* service; over SSH it stops at logout unless lingering is on:

```bash
sudo loginctl enable-linger $USER
loginctl show-user $USER -p Linger # Linger=yes
```

Do not register the `auto-defender` gateway by hand: `nemo agent-hardener setup` (section 5) does it.

## 4. The model key: NVIDIA Build

One NVIDIA API key serves both Helix and Agent Hardener. The user exports it, with the Helix
address, in every terminal that runs `nemo`:

```bash
export INFERENCE_API_KEY=nvapi-...
export NHX_BASE_URL=http://localhost:8080
```

Check the key (expect a JSON body with `choices`; `401` = wrong or incomplete key, `503` = the
model is temporarily down on build.nvidia.com, so retry later):

```bash
curl -s https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $INFERENCE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "nvidia/nemotron-3-ultra-550b-a55b", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 16}'
```

Register NVIDIA Build as the Helix model provider (services running):

```bash
export NEMO_DEFAULT_INFERENCE_KEY=$INFERENCE_API_KEY
export NEMO_DEFAULT_INFERENCE_BASE_URL=https://integrate.api.nvidia.com
nemo setup --auto                  # Registered provider 'nvidia-build', Default model: ..., Setup complete!
nemo inference providers list      # nvidia-build, status READY
nemo models list                   # auto-discovered models
```

`nemo setup --auto` verifies each candidate model and skips any that fail right now (`HTTP 503`),
then prints the default it picked. Check inference through Helix with that default model (keep its
`default/` prefix):

```bash
curl -s http://localhost:8080/apis/inference-gateway/v2/workspaces/default/openai/-/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<default-model>", "messages": [{"role": "user", "content": "Say hello"}]}'
```

Agent Hardener's own models (attacker, defenders, guardrail judge) do not go through this provider:
they call `nvidia/nemotron-3-ultra-550b-a55b` at `https://integrate.api.nvidia.com/v1` directly with
`INFERENCE_API_KEY`, which section 5 saves for them.

## 5. Agent Hardener

With `INFERENCE_API_KEY` exported (once per machine; the first run takes several minutes):

```bash
nemo agent-hardener setup          # "Inference credential provisioned from environment" ... "Setup complete."
nemo agent-hardener doctor         # five ✓ lines, then "All checks passed."
openshell status --gateway auto-defender   # Gateway: auto-defender, Server: https://127.0.0.1:17670, Status: Connected
```

`setup` installs Agent Hardener's and the attacker's environments, saves `INFERENCE_API_KEY` to
`~/.agent-hardener/.env`, points the OpenShell gateway at Docker and registers `auto-defender`.

The machine is ready: return to the skill's step 2.
