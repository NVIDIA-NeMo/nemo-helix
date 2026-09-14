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

export PYTHONPATH="$gym_rw:$root${PYTHONPATH:+:$PYTHONPATH}"
cd "$root"
echo "gym-host: starting gym_host_runtime (gym src $gym_rw)" >&2
exec "$venv/bin/python" "$runtime"
