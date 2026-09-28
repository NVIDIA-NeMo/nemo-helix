#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build one RL CUDA extension wheel into /wheels against the image venv.
#
#   build-rl-cuda-ext-wheel.sh git REPO REV [--submodules]
#   build-rl-cuda-ext-wheel.sh sdist SPEC
#
# git checkouts are full clones. flash-mla and deep-ep stamp a local version
# from `git rev-parse`, which a shallow clone hides.
set -euo pipefail

mode="${1:?usage: build-rl-cuda-ext-wheel.sh git|sdist ...}"
out="${WHEEL_OUT:-/wheels}"
src=/src/pkg
# The image UV_PYTHON points at the CPython prefix, which does not have torch.
# --no-build-isolation has to see the venv the base stage installed torch into.
export UV_PYTHON=/opt/venv/bin/python
export VIRTUAL_ENV=/opt/venv
mkdir -p "$out"
rm -rf "$src" /tmp/sdist

if [[ "$mode" == git ]]; then
    repo="${2:?git mode requires REPO}"
    rev="${3:?git mode requires REV}"
    submodules="${4:-}"
    git clone "$repo" "$src"
    git -C "$src" checkout "$rev"
    if [[ "$submodules" == --submodules ]]; then
        git -C "$src" submodule update --init --recursive
    fi
    uv build --wheel --no-build-isolation --out-dir "$out" "$src"
elif [[ "$mode" == sdist ]]; then
    # uv 0.11.33 has no `uv pip download`. Pull the sdist URL from the PyPI JSON API.
    spec="${2:?sdist mode requires SPEC}"
    name="${spec%%==*}"
    version="${spec#*==}"
    if [[ "$name" == "$spec" || -z "$version" ]]; then
        echo "sdist spec must be name==version, got: $spec" >&2
        exit 1
    fi
    mkdir -p /tmp/sdist
    url="$(/opt/venv/bin/python -c '
import json, sys, urllib.request
name, version = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json") as resp:
    data = json.load(resp)
urls = [item["url"] for item in data["urls"] if item["packagetype"] == "sdist"]
if len(urls) != 1:
    raise SystemExit(f"expected one sdist for {name}=={version}, found {urls}")
print(urls[0])
' "$name" "$version")"
    curl --retry 3 --retry-delay 2 -fsSL -o /tmp/sdist/src.tar.gz "$url"
    tar -xzf /tmp/sdist/src.tar.gz -C /tmp/sdist
    srcdir="$(find /tmp/sdist -mindepth 1 -maxdepth 1 -type d -print -quit)"
    uv build --wheel --no-build-isolation --out-dir "$out" "$srcdir"
else
    echo "unknown mode: $mode" >&2
    exit 1
fi

rm -rf "$src" /tmp/sdist
find "$out" -maxdepth 1 -type f -name '*.whl' -print
