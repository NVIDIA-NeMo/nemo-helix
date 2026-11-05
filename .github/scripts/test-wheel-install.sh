#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exercise a built wheel outside the checkout. Each invocation gets its own
# environment, tool registry, data directory, and service logs, even in one job.
set -euo pipefail
package="${1:?package required}"
method="${2:?install method required}"
: "${WHEEL:?absolute wheel path required}" "${PYTHON_VERSION:?Python version required}"
: "${GITHUB_WORKSPACE:?}" "${RUNNER_TEMP:?}"
if [[ ! -f "${WHEEL}" || "${WHEEL}" != /* ]]; then
  echo "::error::Expected an existing absolute wheel path: ${WHEEL}" >&2
  exit 1
fi
case "${package}:${method}" in
  nemo-helix:venv|nemo-helix:tool|nemo-helix-plugin:venv) ;;
  *) echo "::error::Unsupported install combination: ${package}:${method}" >&2; exit 1 ;;
esac

check_dir="$(mktemp -d "${RUNNER_TEMP}/wheelcheck-${package}-${method}.XXXXXX")"
export RUNNER_TEMP="${check_dir}"
export UV_TOOL_DIR="${check_dir}/tools"
export UV_TOOL_BIN_DIR="${check_dir}/tool-bin"
export NHX_DATA_DIR="${check_dir}/nemo-data"
export NHX_AUTH_ENABLED=false
export _TYPER_FORCE_DISABLE_TERMINAL=1
unset PYTHONPATH VIRTUAL_ENV
cd "${check_dir}"

# Partial-pin direct external dependencies, without applying the checkout's uv
# overrides: the resolver must see the same dependency set as an actual user.
if [[ "${package}" == nemo-helix ]]; then
  constraints="${GITHUB_WORKSPACE}/tools/wheel-constraints/nemo-helix-services.txt"
  requirement="${WHEEL}[services]"
else
  constraints="${GITHUB_WORKSPACE}/tools/wheel-constraints/nemo-helix-plugin.txt"
  requirement="${WHEEL}"
fi
if [[ "${method}" == tool ]]; then
  uv tool install --no-config --python "${PYTHON_VERSION}" \
    --constraints "${constraints}" "${requirement}"
  bin_dir="${UV_TOOL_BIN_DIR}"
  python_bin="${UV_TOOL_DIR}/${package}/bin/python"
else
  uv venv .venv --python "${PYTHON_VERSION}"
  uv pip install --no-config --python .venv/bin/python \
    --constraint "${constraints}" "${requirement}"
  bin_dir="${check_dir}/.venv/bin"
  python_bin="${bin_dir}/python"
fi

if [[ "${package}" == nemo-helix ]]; then
  # Fail directly on plugin import errors: degraded discovery and auth-disabled
  # startup can otherwise hide missing workspace-only dependencies. -I also
  # prevents checkout/PYTHONPATH imports from satisfying a broken wheel.
  "${python_bin}" -I "${GITHUB_WORKSPACE}/.github/scripts/check-wheel-imports.py"
  export PATH="${bin_dir}:${PATH}"
  bash "${GITHUB_WORKSPACE}/script/test-nemo-cli.sh"
else
  "${bin_dir}/python" -c "
import nemo_helix_plugin
import nemo_helix_plugin.cli
import nemo_helix_plugin.commands
print('nemo_helix_plugin', getattr(nemo_helix_plugin, '__version__', '<no __version__>'))
"
fi
