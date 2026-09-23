#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Regenerate the partial dependency constraints used by CI's wheel install-smoke-test
# (.github/wheel-constraints/*.txt).
#
# Why partial (constraints, not a full lock): the smoke test installs the freshly
# built nemo-helix / nemo-helix-plugin wheels. Without any pin it resolves
# the whole tree fresh from PyPI (non-reproducible, supply-chain risk). A FULL
# lock does not work here — the vendored-SDK wheel needs newer deps than
# uv.lock. So we pin each wheel's DIRECT external deps (the versions we've
# vetted), plus any pre-release transitive (uv resolves a pre-release only for a
# package it is told about by name), and let the remaining deep transitives
# resolve normally, so they cannot self-conflict.
#
# Usage:
#   script/compile-wheel-constraints.sh <dir-with-both-wheels>
#
# Wheels come from `uv build --package nemo-helix[-plugin]` (nemo-helix
# needs the Studio/node toolchain) or a CI "<pkg>-wheel-py3.12" artifact
# (`gh run download <run-id> -n nemo-helix-wheel-py3.12 -D <dir>`).
set -euo pipefail

WHEEL_DIR="${1:?usage: script/compile-wheel-constraints.sh <dir-with-built-wheels>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/.github/wheel-constraints"

# uv accepts a pre-release version only for a package named in a first-party requirement or
# constraint, and a pre-release dependency can reach the wheel transitively. Feed the
# pre-releases uv.lock already resolved in as constraints, and pin them in the output too, so
# neither the snapshot below nor CI's install has to fall back to a global --prerelease=allow
# (which would also drag in pre-releases of unrelated transitives).
prerelease_pins() {
  python3 - "${REPO_ROOT}/uv.lock" <<'PY'
import re, sys

blocks = open(sys.argv[1]).read().split("\n[[package]]\n")
prerelease = re.compile(r"(?:[ab]|rc|\.dev)\d")
for block in blocks:
    # Registry packages only: workspace members are not installable from an index.
    match = re.match(r'name = "([^"]+)"\nversion = "([^"]+)"\nsource = \{ registry', block)
    if match and prerelease.search(match.group(2)):
        print("{}=={}".format(*match.groups()))
PY
}

# Pin each of a wheel's direct external deps to the version that resolves for it.
emit_constraints() {
  local wheel="$1" spec="$2" label="$3" out="$4" venv meta
  venv="$(mktemp -d)"
  meta="$(mktemp -d)"
  uv venv "${venv}" --python 3.12 --quiet
  prerelease_pins >"${meta}/prerelease.txt"
  # Resolve+install once so we snapshot a consistent set.
  # --no-config: run from the checkout this would otherwise apply the repo's
  # [tool.uv] override-dependencies and snapshot versions no user can resolve.
  uv pip install --no-config --python "${venv}/bin/python" \
    --constraint "${meta}/prerelease.txt" "${spec}" >/dev/null
  # Direct external deps = the wheel's own Requires-Dist, minus self-referential extras.
  python3 - "$wheel" >"${meta}/names.txt" <<'PY'
import sys, zipfile, re
names=set()
with zipfile.ZipFile(sys.argv[1]) as z:
    md=next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
    for line in z.read(md).decode().splitlines():
        if line.startswith("Requires-Dist:"):
            dep=line.split(":",1)[1].strip()
            name=re.split(r"[<>=!~;\[ ]", dep, 1)[0].strip()
            if name and not name.startswith("nemo-helix"):
                names.add(name)
print("\n".join(sorted(names)))
PY
  # Pre-release transitives need a pin too; names not installed for this wheel drop out below.
  cut -d= -f1 <"${meta}/prerelease.txt" >>"${meta}/names.txt"
  {
    printf '# Partial dependency constraints for the CI wheel install-smoke-test.\n'
    printf '# Pins the DIRECT external deps of the built %s wheel (plus any pre-release\n' "${label}"
    printf '# transitive, which uv resolves only when named) to vetted versions for\n'
    printf '# reproducibility; deep transitives resolve normally and cannot self-conflict.\n'
    printf '# Regenerate with: script/compile-wheel-constraints.sh <dir-with-built-wheels>\n#\n'
    while read -r name; do
      [[ -n "${name}" ]] || continue
      ver="$("${venv}/bin/python" -c "import importlib.metadata as m; print(m.version('${name}'))" 2>/dev/null || true)"
      [[ -n "${ver}" ]] && printf '%s==%s\n' "${name}" "${ver}"
    done < <(sort -u "${meta}/names.txt") | sort
  } >"${out}"
  rm -rf "${venv}" "${meta}"
  echo "wrote ${out} ($(grep -cE '^[a-z0-9].*==' "${out}") pins)"
}

np_wheel="$(find "${WHEEL_DIR}" -name 'nemo_helix-*.whl' | head -1)"
pl_wheel="$(find "${WHEEL_DIR}" -name 'nemo_helix_plugin-*.whl' | head -1)"
[[ -n "${np_wheel}" ]] || { echo "no nemo_helix-*.whl in ${WHEEL_DIR}" >&2; exit 1; }
[[ -n "${pl_wheel}" ]] || { echo "no nemo_helix_plugin-*.whl in ${WHEEL_DIR}" >&2; exit 1; }

emit_constraints "${np_wheel}" "${np_wheel}[services]" "nemo-helix[services]" "${OUT_DIR}/nemo-helix-services.txt"
emit_constraints "${pl_wheel}" "${pl_wheel}" "nemo-helix-plugin" "${OUT_DIR}/nemo-helix-plugin.txt"
