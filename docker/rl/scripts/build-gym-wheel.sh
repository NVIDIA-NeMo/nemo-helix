#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build an installable nemo-gym wheel for the Gym per-server venvs.
#
# Gym rewrites the install line for any server directory that is NOT inside a Gym checkout --
# which a platform environment FileSet staged at /job/environment never is. It strips the editable
# line and substitutes `nemo-gym==<the parent venv's version>`, resolved from a package INDEX
# (nemo_gym/cli/setup_command.py, _get_nemo_gym_version_spec). This image runs the soluwalana fork,
# whose version is published nowhere, so that resolution has no candidate and every native-v1 job
# dies at server spin-up with "No solution found when resolving: nemo-gym" (nvbug 6716627).
#
# The pin is exact, so a wheel built at any other version is invisible to it and the job fails
# exactly as before -- hence the version check at the end.
set -euo pipefail

GYM_SRC=${1:?usage: build-gym-wheel.sh <gym-source-dir> <out-dir>}
OUT_DIR=${2:?usage: build-gym-wheel.sh <gym-source-dir> <out-dir>}

# Upstream declares almost no package-data, relying on setuptools-scm's VCS file-finder to sweep
# the configs and READMEs into the wheel. That finder needs git, and there is none here: the RL
# source arrives by `ADD <repo>#<ref>`, which keeps no submodule .git (a submodule's .git is a
# gitdir pointer file, not a directory). Left alone the wheel builds and installs happily while
# quietly missing every YAML, including the sandbox provider configs. Declare the data explicitly
# rather than depend on VCS state this build cannot have.
# Anchored on the leading "resources/*.py" entry only: upstream appends further entries to this
# list (e.g. token_id_capture golden vectors), and those are carried through by the capture group.
PKG_DATA_RE='^nemo_gym = \["resources/\*\.py"\(.*\)\]$'
if ! grep -q "${PKG_DATA_RE}" "${GYM_SRC}/pyproject.toml"; then
    echo "build-gym-wheel: Gym's [tool.setuptools.package-data] entry moved;" \
         "re-check this substitution against ${GYM_SRC}/pyproject.toml" >&2
    exit 1
fi
# Fully single-quoted, and the brackets/dots/stars escaped: unescaped, `["resources/*.py"]` is a
# BRE character class, which matches nothing here and would edit the file silently not at all.
sed -i 's|^nemo_gym = \["resources/\*\.py"\(.*\)\]$|nemo_gym = ["resources/*.py"\1, "**/*.yaml", "**/*.yml", "**/*.md"]\n"*" = ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.md", "**/*.txt"]|' \
    "${GYM_SRC}/pyproject.toml"
if ! grep -qF '"**/*.yaml"' "${GYM_SRC}/pyproject.toml"; then
    echo "build-gym-wheel: package-data substitution did not apply to ${GYM_SRC}/pyproject.toml" >&2
    exit 1
fi

# --no-config: uv would otherwise discover the platform workspace's `required-version` pin
# (docker/rl/pyproject.workspace.toml, uv <0.10) and refuse to run as this image's uv 0.11.
build_log=$(mktemp)
if ! uv build --no-config --wheel --out-dir "${OUT_DIR}" "${GYM_SRC}" >"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    rm -f "${build_log}"
    exit 1
fi
rm -f "${build_log}"

# Fail here rather than at spin-up on a GPU node: a wheel at the wrong version, or one missing the
# package data above, installs cleanly and only misbehaves later.
shopt -s nullglob
wheels=("${OUT_DIR}"/nemo_gym-*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "build-gym-wheel: expected exactly one wheel, got: ${wheels[*]:-none}" >&2
    exit 1
fi
# package_info.py assembles __version__ from MAJOR/MINOR/PATCH/PRE_RELEASE and imports nothing,
# so it can be read without installing the package it describes.
expected="nemo_gym-$(python -c \
    'import runpy, sys; print(runpy.run_path(sys.argv[1])["__version__"])' \
    "${GYM_SRC}/nemo_gym/package_info.py")-py3-none-any.whl"
if [[ "$(basename "${wheels[0]}")" != "${expected}" ]]; then
    echo "build-gym-wheel: built $(basename "${wheels[0]}"), expected ${expected}" >&2
    exit 1
fi
probe="nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"
# Listed into a variable rather than piped into grep: `grep -q` exits at the first match, unzip
# takes SIGPIPE, and `set -o pipefail` then reports the pipeline as failed -- so a file that IS
# present reads as missing.
entries="$(unzip -Z1 "${wheels[0]}")"
if ! grep -qxF "${probe}" <<<"${entries}"; then
    echo "build-gym-wheel: ${probe} missing from the wheel; package-data did not take" >&2
    exit 1
fi
echo "build-gym-wheel: built $(basename "${wheels[0]}")"
