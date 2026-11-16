#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional Gym prefetch, then re-pin packages the upstream locks leave behind.
#
# The Dockerfile passes:
#   NEMO_GYM_PREFETCH_CONFIGS, TARGETARCH, NEMO_GYM_CUDA, NEMO_GYM_VLLM_VERSION
#   PACKAGE_FLOORS   JSON object mapping package name to its version floor spec.
# Override files are bind-mounted at /tmp.
set -euo pipefail

umask 022
export UV_LINK_MODE=symlink

# Version floors. Each lookup fails the build if the key is missing from PACKAGE_FLOORS.
MLFLOW_SKINNY_FLOOR=$(jq -er '.["mlflow-skinny"]' <<<"${PACKAGE_FLOORS}")
SQLPARSE_FLOOR=$(jq -er '.sqlparse' <<<"${PACKAGE_FLOORS}")
VIRTUALENV_FLOOR=$(jq -er '.virtualenv' <<<"${PACKAGE_FLOORS}")
HYDRA_CORE_FLOOR=$(jq -er '.["hydra-core"]' <<<"${PACKAGE_FLOORS}")
GITPYTHON_FLOOR=$(jq -er '.gitpython' <<<"${PACKAGE_FLOORS}")
WANDB_FLOOR=$(jq -er '.wandb' <<<"${PACKAGE_FLOORS}")

stamp=$(chmod-newer-than.sh mark)
mapfile -t OTEL_PINS < <(grep -Ev '^[[:space:]]*(#|$)' /tmp/opentelemetry-overrides.txt)
OTEL_UPGRADE=()
for pin in "${OTEL_PINS[@]}"; do
    OTEL_UPGRADE+=(--reinstall-package "${pin%%==*}")
done
mkdir -p /opt/gym_venvs
if [ -n "${NEMO_GYM_PREFETCH_CONFIGS:-}" ]; then
    cd /opt/nemo-rl
    case "${TARGETARCH:-amd64}" in arm64) WHEEL_ARCH=aarch64 ;; *) WHEEL_ARCH=x86_64 ;; esac
    GYM_VLLM_OVERRIDE=/opt/nemo-rl/.gym-vllm-override.txt
    printf 'vllm @ https://github.com/vllm-project/vllm/releases/download/v%s/vllm-%s-cp38-abi3-manylinux_2_28_%s.whl\n' \
        "${NEMO_GYM_VLLM_VERSION}" "${NEMO_GYM_VLLM_VERSION}" "${WHEEL_ARCH}" > "${GYM_VLLM_OVERRIDE}"
    UV_TORCH_BACKEND="${NEMO_GYM_CUDA}" \
    UV_OVERRIDE="${GYM_VLLM_OVERRIDE}" \
    uv run python examples/nemo_gym/prefetch_venvs.py ${NEMO_GYM_PREFETCH_CONFIGS}
    rm -f "${GYM_VLLM_OVERRIDE}"
    rm -rf /tmp/ray /root/.ray /tmp/ray_tmp* 2>/dev/null || true
    find /opt/nemo-rl -name .git -prune -exec rm -rf {} +
fi

for py in /opt/nemo_rl_venv/bin/python /opt/ray_venvs/*/bin/python /opt/gym_venvs/*/bin/python; do
    [ -x "${py}" ] || continue
    if "${py}" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("mlflow") else 1)'; then
        uv pip install --python "${py}" --upgrade-package mlflow-skinny --upgrade-package sqlparse \
            "${MLFLOW_SKINNY_FLOOR}" "${SQLPARSE_FLOOR}"
    elif "${py}" -c 'import importlib.metadata as m; m.version("sqlparse")' >/dev/null 2>&1; then
        uv pip install --python "${py}" --no-deps --upgrade-package sqlparse "${SQLPARSE_FLOOR}"
    fi

    # Patch the upstream RL/Gym locks without relocking their CUDA dependency stacks.
    if "${py}" -c 'import importlib.metadata as m; m.version("virtualenv")' >/dev/null 2>&1; then
        uv pip install --directory /tmp --python "${py}" --upgrade-package virtualenv "${VIRTUALENV_FLOOR}"
    fi
    if "${py}" -c 'import importlib.metadata as m; m.version("hydra-core")' >/dev/null 2>&1; then
        uv pip install --python "${py}" --no-deps --upgrade-package hydra-core "${HYDRA_CORE_FLOOR}"
    fi
    if "${py}" -c 'import importlib.metadata as m; m.version("gitpython")' >/dev/null 2>&1; then
        uv pip install --python "${py}" --no-deps --upgrade-package gitpython "${GITPYTHON_FLOOR}"
    fi
    if "${py}" -c 'import importlib.metadata as m; m.version("transformers")' >/dev/null 2>&1; then
        uv pip install --directory /tmp --python "${py}" \
            --overrides /tmp/hf-stack-overrides.txt \
            --upgrade-package transformers --upgrade-package huggingface-hub --upgrade-package tokenizers \
            -r /tmp/hf-stack-overrides.txt
    fi
    if "${py}" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("wandb") or importlib.util.find_spec("opentelemetry") else 1)'; then
        # Outside the project, or uv keeps opentelemetry-api 1.45.
        args=(--directory /tmp --python "${py}" "${OTEL_UPGRADE[@]}" "${OTEL_PINS[@]}")
        if "${py}" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("wandb") else 1)'; then
            uv pip install "${args[@]}" --upgrade-package wandb "${WANDB_FLOOR}"
        else
            uv pip install "${args[@]}"
        fi
    fi
    for pkg in mlflow pynvml mooncake-transfer-engine-cuda13; do
        uv pip uninstall --python "${py}" "${pkg}" || true
    done
done

find /opt/nemo_rl_venv /opt/ray_venvs /opt/gym_venvs -type d \
    \( -path '*/site-packages/mooncake' \
       -o -path '*/site-packages/mooncake_transfer_engine_cuda13-*.dist-info' \
       -o -path '*/site-packages/mooncake_transfer_engine_cuda13.libs' \) \
    -prune -exec rm -rf {} + || true
prune-unreferenced-uv-archives.sh
chmod-newer-than.sh "${stamp}" /opt/nemo_rl_venv /opt/ray_venvs /opt/gym_venvs
