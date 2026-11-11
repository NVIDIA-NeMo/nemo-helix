#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prefetch the seven Ray worker venvs.
# The Dockerfile passes UV_SYNC_MODE and NO_INSTALL_PACKAGES (a JSON array of package names).
# CUDA wheels are mounted at /mnt/wheels.
set -euo pipefail

umask 022
export UV_LINK_MODE=symlink
stamp=$(chmod-newer-than.sh mark)

# Turn the JSON array into one --no-install-package flag per package.
skip_args=()
while IFS= read -r package; do
    skip_args+=(--no-install-package "${package}")
done < <(jq -r '.[]' <<<"${NO_INSTALL_PACKAGES}")

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
        uv sync ${UV_SYNC_MODE-} --inexact --no-install-project --directory /opt/nemo-rl "$@" "${skip_args[@]}"
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
