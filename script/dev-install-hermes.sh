#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dev-only: install Hermes Agent and its Fabric adapter in an isolated environment.
# Run from a NeMo Platform source checkout after `make bootstrap-python`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATFORM_PY="${REPO_ROOT}/.venv/bin/python"
HERMES_VENV="${REPO_ROOT}/.venv-hermes"
HERMES_PY="${HERMES_VENV}/bin/python"
HERMES_COMMIT="f80f453ae0679347e38abc917c7f94f717bf96c5" # Hermes Agent 0.20.1
HERMES_CHECKOUT="${HERMES_VENV}/src/hermes-agent"

if [[ ! -x "${PLATFORM_PY}" ]]; then
  echo "Project venv not found at ${PLATFORM_PY}. Run 'make bootstrap-python' first." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Run 'make bootstrap-python' to verify the project toolchain." >&2
  exit 1
fi

fabric_version="$("${PLATFORM_PY}" -c 'import importlib.metadata as m; print(m.version("nemo-fabric"))' 2>/dev/null || true)"
if [[ -z "${fabric_version}" ]]; then
  echo "nemo-fabric is not installed in ${PLATFORM_PY}. Run 'make bootstrap-python' first." >&2
  exit 1
fi

if [[ ! -x "${HERMES_PY}" ]]; then
  uv --no-config venv --python 3.12 "${HERMES_VENV}"
fi

if [[ -e "${HERMES_CHECKOUT}" && ! -d "${HERMES_CHECKOUT}/.git" ]]; then
  echo "Expected a Git checkout at ${HERMES_CHECKOUT}." >&2
  exit 1
fi
if [[ ! -d "${HERMES_CHECKOUT}/.git" ]]; then
  mkdir -p "$(dirname "${HERMES_CHECKOUT}")"
  git init --quiet "${HERMES_CHECKOUT}"
  git -C "${HERMES_CHECKOUT}" remote add origin https://github.com/NousResearch/hermes-agent.git
elif ! git -C "${HERMES_CHECKOUT}" diff --quiet || ! git -C "${HERMES_CHECKOUT}" diff --cached --quiet; then
  echo "Hermes Agent checkout has tracked changes: ${HERMES_CHECKOUT}." >&2
  exit 1
fi

hermes_head="$(git -C "${HERMES_CHECKOUT}" rev-parse --verify HEAD 2>/dev/null || true)"
if [[ "${hermes_head}" != "${HERMES_COMMIT}" ]]; then
  git -C "${HERMES_CHECKOUT}" fetch --depth 1 origin "${HERMES_COMMIT}"
  git -C "${HERMES_CHECKOUT}" checkout --quiet --detach FETCH_HEAD
fi

uv --no-config pip install --prerelease=allow --python "${HERMES_PY}" \
  "nemo-fabric[relay]==${fabric_version}" \
  "nemo-fabric-adapters-hermes==${fabric_version}" \
  --editable "${HERMES_CHECKOUT}"
uv --no-config pip check --python "${HERMES_PY}"

echo "Hermes Agent 0.20.1 installed in ${HERMES_VENV}."
echo "Set this before starting NeMo Platform:"
echo "export ADAPTER_PYTHON=\"${HERMES_PY}\""
