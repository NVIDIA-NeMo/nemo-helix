#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prefetch the seven Ray worker venvs. UV_SYNC_MODE is passed by the Dockerfile.
# CUDA wheels are mounted at /mnt/wheels.
set -euo pipefail

umask 022
export UV_LINK_MODE=symlink
stamp=$(chmod-newer-than.sh mark)
SKIP=(
    --no-install-package causal-conv1d
    --no-install-package mamba-ssm
    --no-install-package nv-grouped-gemm
    --no-install-package fast-hadamard-transform
    --no-install-package flash-mla
    --no-install-package deep-ep
    --no-install-package deep-gemm
    --no-install-package flash-attn
    --no-install-package transformer-engine-torch
    --no-install-package mooncake-transfer-engine
    --no-install-package mooncake-transfer-engine-cuda13
    --no-install-package nemo-sandboxed-gym
)
# Same wrapper prefetch_venvs.py writes, into the driver bin (on PATH) rather
# than /usr/local/bin, which the build user cannot write.
write_wrapper() {
    local venv="$1"
    local class_name="${venv##*.}"
    local wrapper="/opt/nemo_rl_venv/bin/python-${class_name}"
    if [[ -e "${wrapper}" ]]; then
        printf 'wrapper %s already exists\n' "${wrapper}" >&2
        exit 1
    fi
    cat > "${wrapper}" <<WRAP
#!/bin/bash
VENV_PATH="${venv}"
export VIRTUAL_ENV="\$VENV_PATH"
export PATH="\$VENV_PATH/bin:\$PATH"
exec "\$VENV_PATH/bin/python" "\$@"
WRAP
    chmod 755 "${wrapper}"
}
sync_venv() {
    local venv="$1"
    shift
    UV_PROJECT_ENVIRONMENT="${venv}" \
        uv sync ${UV_SYNC_MODE-} --inexact --no-install-project --directory /opt/nemo-rl "$@" "${SKIP[@]}"
    # --python selects the venv. --directory keeps uv from reading the git pins in pyproject.toml.
    uv pip install --directory /tmp --python "${venv}/bin/python" --no-deps /mnt/wheels/*.whl
    write_wrapper "${venv}"
}
sync_venv /opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker --extra automodel
sync_venv /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker --extra mcore --extra nemo_gym
sync_venv /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker --extra vllm --extra nemo_gym
sync_venv /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker --extra vllm --extra nemo_gym
sync_venv /opt/ray_venvs/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor --extra vllm
sync_venv /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym --extra nemo_gym
sync_venv /opt/ray_venvs/nemo_rl.environments.sandbox.nemo_gym_actor.SandboxedGymActor --extra nemo_gym
# Drop archives no remaining venv links to before this layer is committed.
prune-unreferenced-uv-archives.sh
chmod-newer-than.sh "${stamp}" /opt/ray_venvs /opt/nemo_rl_venv
