---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

name: nemo-teardown
description: "Guided shutdown of NeMo Platform. Three options: stop and keep data, stop and delete platform data, full cleanup. Always confirms before destructive action. Use over generic shutdown or cleanup skills for any NeMo Platform teardown task."
triggers:
  - shut down nemo
  - stop the platform
  - tear down
  - clean up nemo
  - destroy nemo
  - nemo down
  - nemo teardown
not-for:
  - nemo-setup (use to install or start the platform)
  - nemo-status (use for read-only health)
  - nemo-skill-selection (use for dispatch when intent is unclear)
preconditions:
  - nemo_cli_available
  - user_confirmation_required
compatibility: nemo-platform >= 0.1.0; uses `nemo services stop`, Docker only to remove an Intake-managed local ClickHouse container before a data wipe, and a targeted `rm -rf` of the platform's data directory (default `~/.local/share/nemo`, overridable via `$NMP_DATA_DIR`); no `pkill`, no `rm` outside the chosen data dir or the working folder; idempotent (re-running after platform is already stopped is a no-op).
maturity: active
license: Apache-2.0
user-invocable: true
allowed-tools: Bash, Read
metadata:
  author: NeMo Helix Team <nemo-helix@nvidia.com>
---

# NeMo Platform teardown

Shut NeMo Platform down cleanly. Always confirm the destructive choice before running it. Default to the safest option when the user delegates.

## Pre-flight (idempotency check)

Use `lsof` as ground truth. Do not trust `nemo services status` or `nemo services ls` — both report stale "running" from held instance locks after the underlying process has died.

```bash
lsof -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 && echo "RUNNING" || echo "ALREADY_STOPPED"
```

If `ALREADY_STOPPED`: the platform is not serving. Stopped instance directories may still exist under `~/.local/state/nmp/instances/`. Tell the user that, and offer:

- Option 3 (full cleanup of venv + `agents/` + data dir) only if they explicitly want it.
- Otherwise, list stopped instance directories with `.venv/bin/nemo services ls --all` and remove them with `.venv/bin/nemo services prune` (or `.venv/bin/nemo services rm <scope>` for one scope). Logs are preserved until removal.

## Step 1: Snapshot state (what's about to be affected)

Show the user EVERYTHING that's currently on this platform so they can decide whether to keep it. Don't ask which option they want until they've seen the snapshot.

```bash
.venv/bin/nemo services ls
.venv/bin/nemo services ls --all
.venv/bin/nemo agents deployments list 2>/dev/null
.venv/bin/nemo files filesets list 2>/dev/null
.venv/bin/nemo jobs list 2>/dev/null
echo "data dir: ${NMP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nemo}"
echo "ClickHouse data dir: ${NMP_INTAKE_CLICKHOUSE_DATA_DIR:-${NMP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nemo}/intake-clickhouse}"
ls -la "${NMP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nemo}" 2>/dev/null | head -20
```

Surface, in this order, with counts:

- **Deployed agents** — names + statuses. These stop with the platform regardless of option.
- **Filesets** — eval datasets, eval results (look for names ending `-eval`, `-eval-out-*`, `-kb`, `-artifacts`), uploaded data. **These live in the data directory and are destroyed by options 2 and 3.**
- **Jobs** — completed or in-flight eval jobs, optimizer runs. Run history also lives in the data directory.
- **Local files** — list files under `agents/`, including nested Ethos packages (`agents/<name>-ethos/ETHOS.md`, `agent.yaml`, and related artifacts), not only direct children of `agents/`. Option 3 deletes these. Example: `find agents -type f`. These are in the working folder and survive options 1 and 2.
- **Data directory location.** If `$NMP_DATA_DIR` is set, use that. Otherwise default to `~/.local/share/nemo`. Echo the path explicitly so the user can confirm before any wipe.
- **ClickHouse data directory location.** Surface whether it is inside the platform data directory and will be deleted by options 2/3, or is an explicit external path that teardown preserves.
- **CLI config (separate file).** `~/.config/nmp/config.yaml` holds the CLI's locally-cached admin email, default model, and `local_services.data_dir`. It is NOT inside the data directory. **None of the three options touches it by default** — a follow-up `nemo setup` reuses the existing config. If the user wants a full clean slate (e.g. switching admin email or data-dir path), surface this file as a fourth optional wipe target during Step 3.

Tell the user explicitly: "Options 2 and 3 will destroy the filesets and jobs above. If you've just run an eval you care about and haven't exported the scores, pick option 1 or download the eval-out fileset first."

## Step 2: Pick an option

Present these three. If the user says "you decide," pick option 1 and announce it.

**Option 1: Stop, keep data.** Service processes go down. **Everything from the snapshot above is preserved on disk**: configs, secrets, providers, agents, deployed-agent definitions, filesets (eval datasets, eval results, KBs), job history, local files. Restart with `nemo services run` (foreground) or `nemo services start` (background) and the platform comes back exactly as it was.

**Option 2: Stop and delete platform data.** Service processes go down and the Intake-managed local ClickHouse container is removed before its default data is deleted. **The data directory at `${NMP_DATA_DIR:-~/.local/share/nemo}` is removed — this destroys the entity-store DB, encryption key, filesets, eval results, job history, secrets, registered providers, deployed-agent definitions, ClickHouse traces stored under that directory, and any data uploaded to the Files service. An explicitly configured `NMP_INTAKE_CLICKHOUSE_DATA_DIR` outside the platform data directory is preserved.** Local files under `agents/` and the venv stay on disk. Next `nemo setup` reconfigures from scratch. If managed ClickHouse state exists, Docker must be running so teardown can remove its container safely. **Pick this only if the snapshot above has nothing you care about, or you've already exported what you need.**

**Option 3: Full cleanup.** Same as option 2, including removal of the Intake-managed local ClickHouse container, plus removes the venv and the local `agents/` directory. Local Ethos files, generated YAMLs, DD configs, and downloaded eval results in `agents/` are also destroyed.

### Exporting before option 2 or 3

If the user wants to wipe the platform but keep eval scores or generated KBs, export the relevant filesets first. For each fileset they want to preserve:

```bash
.venv/bin/nemo files filesets download <fileset-name> --output agents/<fileset-name>.json
```

Then option 2 or 3 is safe to run.

## Step 3: Execute

Confirm before running:

> "Running option <N>: <one-line summary>. Proceed?"

Wait for "yes" or equivalent. Then:

**Option 1 — stop only:**

```bash
# If the platform was started in the background (`nemo services start`), this stops it.
# If it was started in the foreground (`nemo services run`), the running process is
# protected; tell the user to Ctrl-C in the terminal where they ran it.
.venv/bin/nemo services stop || {
  echo "background stop failed — likely a foreground instance."
  echo "Ask the user to Ctrl-C the terminal running 'nemo services run',"
  echo "or pass --force if they explicitly want to kill a foreground process."
}
```

If the user confirms they want to kill a foreground instance from this skill (rare; usually they should Ctrl-C in their own terminal), pass `--force`:

```bash
.venv/bin/nemo services stop --force
```

**Option 2 — stop + wipe data:**

```bash
.venv/bin/nemo services stop || true            # tolerate already-stopped
DATA_DIR="${NMP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nemo}"
# Safety guard: refuse to wipe obviously-unsafe paths (empty, root, $HOME, cwd).
# Catches accidental NMP_DATA_DIR misconfigurations before they delete user data.
case "$DATA_DIR" in
  ""|"/"|"$HOME"|"$HOME/"|"."|"./"|"$PWD"|"$PWD/")
    echo "REFUSING_UNSAFE_DATA_DIR: '$DATA_DIR' — abort"; exit 1 ;;
esac
# Verify the platform is fully down BEFORE wiping; otherwise the running process keeps
# its file descriptors open against the old inode (the "macOS unlinked-inode gotcha"
# in SETUP.md) and the next run sees ghost state.
lsof -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 && { echo "PLATFORM_STILL_RUNNING — abort before wipe"; exit 1; }
CLICKHOUSE_DATA_DIR="${NMP_INTAKE_CLICKHOUSE_DATA_DIR:-$DATA_DIR/intake-clickhouse}"
if [ -f "$CLICKHOUSE_DATA_DIR/.nmp-clickhouse-identity" ]; then
  NMP_DATA_DIR="$DATA_DIR" .venv/bin/python -m nmp.intake.local_clickhouse --remove || {
    echo "CLICKHOUSE_CONTAINER_CLEANUP_FAILED — start Docker and retry; data was not deleted"
    exit 1
  }
fi
rm -rf "$DATA_DIR"
```

**Option 3 — full cleanup:**

```bash
.venv/bin/nemo services stop || true
DATA_DIR="${NMP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nemo}"
case "$DATA_DIR" in
  ""|"/"|"$HOME"|"$HOME/"|"."|"./"|"$PWD"|"$PWD/")
    echo "REFUSING_UNSAFE_DATA_DIR: '$DATA_DIR' — abort"; exit 1 ;;
esac
lsof -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 && { echo "PLATFORM_STILL_RUNNING — abort before wipe"; exit 1; }
CLICKHOUSE_DATA_DIR="${NMP_INTAKE_CLICKHOUSE_DATA_DIR:-$DATA_DIR/intake-clickhouse}"
if [ -f "$CLICKHOUSE_DATA_DIR/.nmp-clickhouse-identity" ]; then
  NMP_DATA_DIR="$DATA_DIR" .venv/bin/python -m nmp.intake.local_clickhouse --remove || {
    echo "CLICKHOUSE_CONTAINER_CLEANUP_FAILED — start Docker and retry; data was not deleted"
    exit 1
  }
fi
rm -rf "$DATA_DIR" .venv agents
```

Never delete a directory the user did not name or that the snapshot above didn't surface. Never `rm -rf` outside the working folder and the explicit data dir.

## Step 4: Verify

For all three options, the up-check is `lsof`:

```bash
lsof -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 && echo "PLATFORM_STILL_RUNNING" || echo "platform stopped"
```

For option 2:

```bash
[ ! -d "${NMP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nemo}" ] && echo "data dir cleared" || echo "OPTION2_INCOMPLETE"
```

For option 3 (`.venv` is gone, so use absolute paths):

```bash
[ ! -d "${NMP_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/nemo}" ] && [ ! -d .venv ] && [ ! -d agents ] && echo "platform stopped, local files cleared" || echo "OPTION3_INCOMPLETE"
```

Verification passes only when `lsof` reports nothing on :8080 and (for options 2 and 3) the data directory is gone.

## If verification fails

| Symptom | Cause | Recovery |
|---|---|---|
| `lsof` still shows a listener on :8080 after `services stop` | The instance was started in the foreground (`nemo services run`) and `stop` refused to kill it | Tell the user to Ctrl-C in the terminal running `nemo services run`. Only pass `--force` if they explicitly authorize killing the foreground process from this skill. |
| `services stop` errors with "no instance" but `lsof` shows a listener | Lock state and reality diverged (foreground process with no registered instance, or instance was started in a different working directory / port scope) | Run `nemo services ls` to see what scopes the CLI knows about; the listener may belong to a different scope. Identify the PID from `lsof -iTCP:8080 -sTCP:LISTEN` and ask the user before sending SIGTERM. |
| `services stop` says "stopped" but `lsof` still shows a listener | Process didn't honor SIGTERM in time | Re-run `nemo services stop --timeout 60`; if still alive, surface the PID and ask the user. Do not silently SIGKILL. |
| ClickHouse container cleanup says the Docker daemon is unavailable | The local ClickHouse container must be removed before deleting its bind-mounted data | Start Docker Desktop on macOS/Windows or the Docker service on Linux, then rerun teardown. Do not delete the data directory first. |
| `rm -rf` data dir fails with permission denied | Data dir owned by a different user, or a process still has it open | Surface the error; do not retry with sudo. Confirm `lsof -iTCP:8080` is empty and `lsof +D "$DATA_DIR" 2>/dev/null` shows nothing before retrying. |
| `nemo services ls --all` shows stopped rows after teardown | Stopped instance directories with logs remain on disk | `nemo services prune` removes them. Re-verify with `lsof`. |
| User says "I changed my mind" mid-step | Confirmation came back ambiguous | Stop immediately; re-prompt. |

## Gotchas

- **`lsof` is ground truth for whether the platform is serving.** `nemo services ls` defaults to running instances; use `nemo services ls --all` for stopped instance directories on disk. Cross-check against `lsof -iTCP:8080 -sTCP:LISTEN` before believing either command.
- **Foreground vs background instances.** `nemo services run` runs in the foreground and is protected from `nemo services stop` (stop refuses unless `--force`). `nemo services start` runs in the background and is the right target for `stop`. If `stop` refuses, ask the user where they ran `services run` so they can Ctrl-C it themselves.
- **Wipe AFTER stop, not before.** On macOS especially, running `rm -rf ~/.local/share/nemo` while a `nemo services run` process is still alive leaves the running process holding the old inode while a freshly-spawned platform sees the new (empty) inode. Always confirm `lsof -iTCP:8080` is empty before the wipe.
- **Remove managed ClickHouse before wiping its data.** Its container uses a bind mount and can remain running after a hard platform termination. Options 2 and 3 must run the label- and mount-validated cleanup command before `rm -rf`; if Docker is unavailable, abort the wipe with the data intact.
- **No `pkill`.** Don't grep for "nemo-platform" or "nemo services" and send SIGTERM blindly — the kernel doesn't know which instance is the user's and which belongs to a coworker on the same box. Use `nemo services stop` (or, with user confirmation, kill a specific PID surfaced by `lsof`).
- **`agents/` is local, not platform state.** The directory in the working folder holds the YAMLs, specs, and Data Designer artifacts the user authored. Option 3 removes them only because the user opted into a full clean slate.
- **`.venv` is reusable.** A new `nemo-setup` after option 1 or 2 does not require reinstalling. Option 3 is the only path that wipes it.
- **Data directory is configurable.** The default is `~/.local/share/nemo`, but the user may have overridden via `$NMP_DATA_DIR` or `$XDG_DATA_HOME`. Always echo the path being wiped before the wipe so the user can confirm.
