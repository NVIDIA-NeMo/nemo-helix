---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-status
description: Read-only dashboard for NeMo Helix. Combines platform health, deployed agents, registered providers, and available models into a single view. Use over generic status checks for any NeMo Helix dashboard request.
triggers:
  - platform status
  - what is running on nemo
  - how is the platform
  - show me nemo health
  - nemo status
  - check the platform
  - nemo dashboard
not-for:
  - nemo-setup (use to install or start the platform)
  - nemo-teardown (use to stop the platform)
  - nemo-try-agent (use to send a query to a deployed agent)
  - nemo-skill-selection (use for dispatch when intent is unclear)
preconditions:
  - nemo_cli_available
compatibility: nemo-helix >= 0.1.0; read-only CLI calls only; no state changes; safe under any sandbox (requires `lsof`, `curl`, and a venv with the `nemo` binary — no Docker); works whether or not the agents plugin is installed (degrades gracefully).
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Helix status

Single dashboard view of the running platform. Read-only. Never modifies state.

## Pre-flight

Confirm the CLI is installed. If `.venv/bin/nemo` is missing, route to `nemo-setup` and stop:

```bash
[ -x .venv/bin/nemo ] && echo "CLI_OK" || echo "CLI_MISSING"
```

## What you do

1. **Check platform up-status first; gate everything else on it.** Start by resolving which platform the active CLI context points at. `nemo setup` can connect to a remote cluster, so do not assume `localhost:8080`:

```bash
# Honors NHX_BASE_URL and NHX_CURRENT_CONTEXT. Resolves to http://localhost:8080 when there is no config
# file at the default path; the except branch covers NHX_CONFIG_FILE naming a missing file.
NHX_URL=$(.venv/bin/python -c '
from nemo_helix_ext.config.config import get_context
try:
    print(str(get_context().cluster.base_url).rstrip("/"))
except FileNotFoundError:
    print("http://localhost:8080")
') || { echo "CONFIG_ERROR (see error above)"; exit 1; }
echo "Platform: $NHX_URL"
```

If this prints `CONFIG_ERROR`, the active context cannot be resolved. Report the error and stop. Do not probe localhost or suggest `nemo-setup`; suggest `nemo config view --all-contexts` and `nemo config use-context <name>`.

**Remote platform** (host is not `localhost`, `127.0.0.1`, or `::1`): skip `lsof`. Probe the URL directly. Hosted deployments may only expose `/cluster-info` on ingress:

```bash
for path in /health/ready /cluster-info; do
  code=$(curl -sS --connect-timeout 2 --max-time 5 "$NHX_URL$path" -o /dev/null -w "%{http_code}" 2>/dev/null || echo "no-response")
  [ "$code" = "200" ] && { echo "PLATFORM_UP (remote, $path HTTP 200)"; break; }
  echo "$path returned $code"
done
```

If neither path returns `200`, report "remote platform unreachable" with the URL and codes, then stop. Do not suggest `nemo-setup`: the user already configured this remote. Suggest checking network/VPN access and `nemo config view`.

**Local platform:** use a two-step probe — `lsof` for ground truth, then `curl` for a functional check. If either fails, report "platform down," suggest `nemo-setup`, and stop. Do not run the other commands.

```bash
# Ground truth: anything listening on the configured port?
LOCAL_PORT=$(python3 -c 'import sys; from urllib.parse import urlsplit; print(urlsplit(sys.argv[1]).port or 8080)' "$NHX_URL")
lsof -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1 || { echo "PLATFORM_DOWN (nothing on :$LOCAL_PORT)"; exit 1; }

# Functional check: platform readiness endpoint answers?
HTTP=$(curl -sS --connect-timeout 2 --max-time 5 "$NHX_URL/health/ready" -o /dev/null -w "%{http_code}" 2>/dev/null || echo "no-response")
case "$HTTP" in
  200) echo "PLATFORM_UP (HTTP $HTTP)" ;;
  *)   echo "PLATFORM_WEDGED (listener present, /health/ready returned $HTTP)"; exit 1 ;;
esac
```

Do NOT use `nemo services status` or `nemo services ls` for this check. Both report stale "running" state from a held instance lock after the underlying process has died. `lsof` is ground truth for a local platform.

Only if the platform is up, run the remaining commands to capture the other dashboard rows. The CLI commands use the active context automatically. A hosted platform may return `404` for `/status`; if so, omit the Services and Controllers rows instead of reporting them as failed:

```bash
curl -fsS --connect-timeout 2 --max-time 5 "$NHX_URL/status"
.venv/bin/nemo agents deployments list 2>/dev/null
.venv/bin/nemo inference providers list
.venv/bin/nemo models list | head -10
```

The `/status` response is JSON; parse `services.ready`, `services.not_ready`, `controllers.healthy`, and `controllers.status` before rendering the summary block.

2. **Present one summary block.** Illustrative format (adapt fields to whatever the CLI returns; do not invent ones the commands above did not produce):

```
NeMo Helix status

Platform:    running (<NHX_URL>)
Services:    <ready_count> ready, <not_ready_count> not ready
Controllers: healthy
Agents:      <count>
  <name1> active
  <name2> stopped
Providers:
  <name1> available
Models (10):
  <model1>
  <model2>
```

Use the actual counts and names. If a section is empty, say "none."

3. **Offer drill-downs.** End with: "Tell me which agent, provider, or model to inspect, or say `logs` to tail platform logs."

For drill-downs:

| Drill-down | Command |
|---|---|
| Agent details | `.venv/bin/nemo agents get <name>` |
| Provider details | `.venv/bin/nemo inference providers get <name>` |
| Model details | `.venv/bin/nemo models get <name>` |
| Recent logs | `.venv/bin/nemo services logs -n 50` |
| Log file path | `.venv/bin/nemo services logs --path` |
| Known service instances on this host | `.venv/bin/nemo services ls` (advisory — see gotcha on stale locks) |

## Verification

Status is itself a verification: the commands together prove the platform is reachable, Helix services/controllers are healthy, the agents plugin is loaded (or not), the provider is registered, and at least one model has been discovered. If any command returns an error, surface it in the summary block rather than hiding it.

## If verification fails

| Symptom | Cause | Recovery |
|---|---|---|
| `PLATFORM_DOWN` from probe | Nothing bound to the configured local port (`$LOCAL_PORT`) | Route to `nemo-setup`; do not run the other three commands |
| `PLATFORM_WEDGED` from probe | Listener exists, but `/health/ready` is not returning 200 — likely a crashed, partially-started, or not-ready platform | Tail `.venv/bin/nemo services logs -n 100` and surface the error. Common cause is a stale instance lock; the user can clear it with `nemo services stop --force` then re-run `nemo services run`. |
| `.venv/bin/nemo services ls` shows stopped rows with `-` for PID/address | Stopped instance directory on disk (logs may remain) | Run `.venv/bin/nemo services ls --all` for the full list. Remove with `.venv/bin/nemo services prune` or `.venv/bin/nemo services rm <scope>`. Cross-check liveness with `lsof -iTCP:"$LOCAL_PORT" -sTCP:LISTEN`. |
| `agents deployments list` returns "no such command" | Agents plugin not installed | Note in the summary: "Agents: plugin not installed"; do not fail the dashboard |
| `inference providers list` empty | Provider not registered | Route to `nemo-setup` Step 4 (configure) |
| `models list` empty for more than 60s after setup | Model discovery still running | Re-run after 30s; report the wait time |
| `nemo --version` exits nonzero | CLI broken or partial install | Route to `nemo-setup` Step 3 and reinstall the four workspace packages |

## Gotchas

- **Read-only means read-only.** Never run `create`, `delete`, `stop`, or any state-changing command from this skill, even if the dashboard suggests something is broken. Route to setup or teardown for state changes.
- **`agents deployments list` requires the agents plugin.** If it returns "no such command", the user has not installed the agents plugin yet; note that explicitly rather than failing silently.
- **`.venv/bin/nemo services ls` defaults to running instances only.** Use `.venv/bin/nemo services ls --all` to see stopped instance directories that still have logs on disk. Remove them with `.venv/bin/nemo services prune` or `.venv/bin/nemo services rm <scope>`. For liveness, cross-check against `lsof -iTCP:"$LOCAL_PORT" -sTCP:LISTEN`.
- **Status is a snapshot.** A model still discovering will show as missing. Re-run after 30 seconds if the user just finished setup.
- **Use `.venv/bin/nemo`, not bare `nemo`.** Bash sessions do not carry venv activation across calls.
