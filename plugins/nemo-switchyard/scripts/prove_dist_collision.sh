#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Throwaway venvs proving dist rename + vendor/native isolation.
# Do not install into the worktree .venv or /work/nmp/.venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLUGIN="$ROOT/plugins/nemo-switchyard"
COLLISION_ROOT="${SWITCHYARD_COLLISION_DIR:-${TMPDIR:-/tmp}/nemo-switchyard-collision}"
mkdir -p "$COLLISION_ROOT"
BASE="$(mktemp -d "${COLLISION_ROOT%/}/run.XXXXXX")"
MAY="$BASE/may-venv"
NATIVE="$BASE/native-venv"
# Remove only the unique run directory, never the operator-supplied COLLISION_ROOT.
trap 'rm -rf -- "$BASE"' EXIT

python3 -m venv "$MAY"
# shellcheck disable=SC1091
source "$MAY/bin/activate"
python -m pip install -U pip
python -m pip install --no-deps "$PLUGIN"
python -m pip install "$PLUGIN/vendor/switchyard"
python - << 'PY'
import importlib.util
import switchyard.lib  # noqa: F401
assert importlib.util.find_spec("switchyard_rust") is None, "May venv must not have switchyard_rust"
print("MAY_OK", switchyard.lib.__file__)
PY
python - << 'PY'
import json
import subprocess
import sys

pkgs = json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"]))
names = [p["name"] for p in pkgs]
print("MAY_PKGS", [n for n in names if "switchyard" in n.lower()])
assert names.count("nemo-switchyard-plugin") == 1
assert names.count("nemo-switchyard") == 0, "May venv must not install upstream dist nemo-switchyard"
assert names.count("switchyard-vendored") == 1
PY
deactivate

python3 -m venv "$NATIVE"
# shellcheck disable=SC1091
source "$NATIVE/bin/activate"
python -m pip install -U pip
# Plugin without its vendor extra: install the path with --no-deps then rust.
python -m pip install --no-deps "$PLUGIN"
python -m pip install "git+https://github.com/NVIDIA-NeMo/Switchyard.git@${SWITCHYARD_NATIVE_TAG:-v0.3.0-rc.2}"
python - << 'PY'
import importlib
import switchyard_rust
import nemo_switchyard
print("NATIVE_OK", switchyard_rust.__file__, nemo_switchyard.__file__)
# One switchyard tree (upstream), not May vendor.
sy = importlib.import_module("switchyard")
assert "vendor/switchyard" not in (getattr(sy, "__file__", "") or "")
PY
python - << 'PY'
import json
import subprocess
import sys

pkgs = json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"]))
names = [p["name"] for p in pkgs]
print("NATIVE_PKGS", [n for n in names if "switchyard" in n.lower()])
assert names.count("nemo-switchyard-plugin") == 1
assert names.count("nemo-switchyard") == 1, "native venv should have exactly one upstream dist named nemo-switchyard"
assert names.count("switchyard-vendored") == 0
PY
echo "COLLISION_SMOKE_OK"
