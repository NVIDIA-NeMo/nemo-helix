#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start gym_host_runtime inside the runtime image with a writable Gym tree.
#
# Usage:
#   gym_host.sh [venv] [git_root] [writable_gym_src] [runtime]
#
# Defaults match sandboxed_gym.host.entrypoint.
set -eu

# Nothing this launcher starts needs execd's token, so it goes before the job's variables can change
# which programs run here. Unsetting it would leave it in /proc/<pid>/environ for a child to read, so
# the launcher starts again without it.
if [ -n "${EXECD_ACCESS_TOKEN+set}" ]; then
    exec /usr/bin/env -u EXECD_ACCESS_TOKEN /bin/sh "$0" "$@"
fi

# The agent bundle's CLI, named by the host provider: its directory goes first on PATH and it is checked
# here, before the job's variables arrive, as the launcher that ran this script used to. This script has
# no privileged setup to wait for. Unset, so the runtime doesn't repeat the check with the job's variables.
# A failing or hanging agent (the runtime's 120 s limit, where coreutils `timeout` exists) stops the script.
# An image with a container engine starts as root for its setup; there the runtime checks the agent after
# dropping privileges, so this script leaves the name in place.
if [ -n "${NHX_GYM_HOST_CONTAINER_ENGINE:-}" ]; then
    :
elif [ -n "${NHX_GYM_AGENT_EXECUTABLE:-}" ]; then
    PATH="${NHX_GYM_AGENT_EXECUTABLE%/*}:$PATH"
    export PATH
    if command -v timeout >/dev/null 2>&1; then
        timeout 120 "$NHX_GYM_AGENT_EXECUTABLE" --version >&2
    else
        "$NHX_GYM_AGENT_EXECUTABLE" --version >&2
    fi
    unset NHX_GYM_AGENT_EXECUTABLE
fi

# This launcher has no privileged setup, so the job's variables, which the orchestrator packs into
# NHX_GYM_JOB_ENV, are unpacked first, as if they had arrived in the environment: before the launcher's
# own parameters and exports below, which keep their precedence over them, and before Ray starts. Names
# a shell can't export stay packed for the runtime, and so does anything this can't decode, which the
# runtime then reports as a bootstrap failure.
if [ -n "${NHX_GYM_JOB_ENV:-}" ]; then
    if job_env_exports=$(python3 -c '
import base64, json, os, re, shlex
job = json.loads(base64.b64decode(os.environ["NHX_GYM_JOB_ENV"], validate=True))
rest = {}
for name, value in job.items():
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        print(f"export {name}={shlex.quote(value)}")
    else:
        rest[name] = value
packed = base64.b64encode(json.dumps(rest, sort_keys=True).encode()).decode() if rest else ""
print(f"NHX_GYM_JOB_ENV={packed}")
'); then
        eval "$job_env_exports"
    fi
fi
if [ -z "${NHX_GYM_JOB_ENV:-}" ]; then
    unset NHX_GYM_JOB_ENV
fi

venv=${1:-/opt/ray_venvs/sandboxed_gym.ray.gym_actor.SandboxedGymActor}
root=${2:-/opt/nemo-rl}
gym_rw=${3:-/tmp/gym-src/Gym}
runtime=${4:-}

if [ -z "$runtime" ]; then
    runtime=$root/packages/sandboxed_gym/src/sandboxed_gym/runtime/gym_host_runtime.py
fi

gym_tree=${SANDBOXED_GYM_TREE:-$root/3rdparty/Gym-workspace/Gym}

# Copy Gym to a writable path only because /opt/gym_venvs is often not writable by
# uid 1000 under OpenSandbox. Remove this once that directory is writable.
# Staged through a sibling and renamed, rather than copied straight to $gym_rw. `cp -a src dst`
# copies *into* dst when dst already exists, so a leftover or half-written $gym_rw would become
# $gym_rw/Gym/nemo_gym -- which leaves the guard below still true, so every restart adds another
# layer and PYTHONPATH never resolves nemo_gym. Removing first also repairs an interrupted copy.
if [ ! -d "$gym_rw/nemo_gym" ]; then
    mkdir -p "$(dirname "$gym_rw")"
    rm -rf "$gym_rw" "$gym_rw.tmp"
    cp -a "$gym_tree" "$gym_rw.tmp"
    mv "$gym_rw.tmp" "$gym_rw"
fi

# Again, in case the job's variables named it.
unset EXECD_ACCESS_TOKEN

# Pin the single-node in-sandbox Ray to loopback; the routable node IP it picks is dropped by the deny-default sandbox egress.
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0

# One node that is both head and worker. `ray start --head` runs GCS and the raylet
if [ -x "$venv/bin/ray" ]; then
    ray_bin=$venv/bin/ray
else
    ray_bin=ray
fi
ray_head_address=127.0.0.1:6379
"$ray_bin" start --head \
    --disable-usage-stats \
    --include-dashboard=false \
    --node-ip-address=127.0.0.1 \
    --port=6379

# Gym connects only when this key is in the global config. The orchestrator strips it on the way
# in, so publish the loopback head here, after this sandbox's own cluster is up.
export RAY_HEAD_NODE_ADDRESS="$ray_head_address"
NHX_GYM_GLOBAL_CONFIG=$(python3 -c '
import json, os
raw = os.environ.get("NHX_GYM_GLOBAL_CONFIG", "").strip()
cfg = json.loads(raw) if raw else {}
if not isinstance(cfg, dict):
    raise SystemExit("NHX_GYM_GLOBAL_CONFIG must be a JSON object")
cfg["ray_head_node_address"] = os.environ["RAY_HEAD_NODE_ADDRESS"]
print(json.dumps(cfg))
')
export NHX_GYM_GLOBAL_CONFIG

# The runtime discovers built-in servers from this variable. The third argument
# stages the checkout but does not otherwise reach that process.
export SANDBOXED_GYM_SRC_DIR="$gym_rw"
export PYTHONPATH="$gym_rw:$root${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
echo "gym-host: starting gym_host_runtime (gym src $gym_rw)" >&2
exec "$venv/bin/python" "$runtime"

