#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PLUGIN="$ROOT/plugins/nemo-switchyard"
BASE="$(mktemp -d "${TMPDIR:-/tmp}/nemo-switchyard-collision.XXXXXX")"
trap 'rm -rf -- "$BASE"' EXIT

uv venv "$BASE/vendor"
uv pip install --python "$BASE/vendor/bin/python" --no-deps "$PLUGIN" "$PLUGIN/vendor/switchyard"
"$BASE/vendor/bin/python" - <<'PY'
import importlib.util
import switchyard.lib

assert importlib.util.find_spec("switchyard_rust") is None
print("VENDOR_OK", switchyard.lib.__file__)
PY

uv venv "$BASE/native"
uv pip install --python "$BASE/native/bin/python" --no-deps "$PLUGIN" "nemo-switchyard==0.3.0"
"$BASE/native/bin/python" - <<'PY'
import importlib.metadata

import nemo_switchyard
import switchyard_rust

assert importlib.metadata.version("nemo-switchyard-plugin") == "0.1.0"
assert importlib.metadata.version("nemo-switchyard") == "0.3.0"
print("NATIVE_OK", nemo_switchyard.__file__, switchyard_rust.__file__)
PY
