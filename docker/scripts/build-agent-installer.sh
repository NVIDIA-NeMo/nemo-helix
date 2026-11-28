#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the agent installer for one platform into <output>/<os>-<arch>.tar.gz, and collect its sources into
# /source-distributions/nhx-agent-installer. Run by Dockerfile.nhx-agent-installer in a stage of that platform.
#
# Usage: build-agent-installer.sh <platform> <output-dir>
#   platform: linux/amd64 or linux/arm64.
# Reads PYTHON_VERSION, PYTHON_BUILD_STANDALONE_RELEASE, HARBOR_VERSION, NEMO_FABRIC_VERSION,
# OPENTELEMETRY_PROTO_VERSION, PROTOBUF_VERSION and NHX_COLLECT_SOURCES from the environment.

set -euo pipefail

platform="$1"
output="$2"
case "${platform}" in
    linux/amd64) machine=x86_64 ;;
    linux/arm64) machine=aarch64 ;;
    *) echo "no agent installer for ${platform}" >&2; exit 1 ;;
esac
installer=/opt/nhx-agent-installer
sources=/source-distributions/nhx-agent-installer

uv python install "${PYTHON_VERSION}" --install-dir "${installer}/python"
python_dir="${installer}/python/cpython-${PYTHON_VERSION}-linux-${machine}-gnu"
if [[ ! -x "${python_dir}/bin/python3" ]]; then
    echo "uv installed $(ls "${installer}/python"), not a ${machine} CPython: this stage isn't running as ${platform}" >&2
    exit 1
fi
build="$(cat "${python_dir}/BUILD")"
if [[ "${build}" != "${PYTHON_BUILD_STANDALONE_RELEASE}" ]]; then
    echo "uv installed python-build-standalone ${build}, not ${PYTHON_BUILD_STANDALONE_RELEASE}: re-pin docker/sources/nhx-agent-installer.json" >&2
    exit 1
fi
uv venv --relocatable --python "${python_dir}/bin/python3" "${installer}/venv"
printf 'harbor==%s\n' "${HARBOR_VERSION}" > /tmp/overrides.txt
uv pip install --python "${installer}/venv/bin/python" --override /tmp/overrides.txt \
    "harbor==${HARBOR_VERSION}" "nemo-fabric[harbor]==${NEMO_FABRIC_VERSION}" \
    "opentelemetry-proto==${OPENTELEMETRY_PROTO_VERSION}" "protobuf==${PROTOBUF_VERSION}"
uv pip install --python "${installer}/venv/bin/python" --no-deps /src/nhx_evals_sdk
# The installer's entry point, and the agents it installs, import without the SDK's other dependencies.
"${installer}/venv/bin/python" -c 'import nhx_evals_sdk.agent_eval.runtimes.harbor.agent_layer_install, \
nhx_evals_sdk.agent_eval.runtimes.harbor.fabric_installed_agent, harbor.agents.installed.codex, \
harbor.agents.installed.claude_code'

python3 /usr/local/bin/collect-pinned-sources.py \
    --manifest /tmp/sources/nhx-agent-installer.json \
    --manifest /tmp/sources/kaniko-busybox.json \
    --output "${sources}"
uv run --no-project --python "${installer}/venv/bin/python" \
    /usr/local/bin/collect-python-sdists.py --python "${installer}/venv/bin/python" --output "${sources}"
uv run --no-project --python "${installer}/venv/bin/python" \
    /usr/local/bin/collect-cpython-source.py --python "${installer}/venv/bin/python" --output "${sources}"
mkdir -p "${sources}" "${output}"
tar -C /opt -czf "${output}/${platform//\//-}.tar.gz" nhx-agent-installer
