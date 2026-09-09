# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sandboxed Gym host runtime for Evaluator agent-evaluation jobs.
#
ARG NMP_PYTHON_BASE=nmp-python-base
ARG NMP_PYTHON_IMAGE=python:3.13.15-slim-trixie

FROM ${NMP_PYTHON_BASE} AS builder

# Copy the host's standalone dependency manifest and lockfile.
COPY docker/locks/nmp-gym-host /app/docker/locks/nmp-gym-host

# Copy the complete package inputs at the path used by the lock's local source.
COPY packages/sandboxed_gym/pyproject.toml /app/packages/sandboxed_gym/pyproject.toml
COPY packages/sandboxed_gym/README.md /app/packages/sandboxed_gym/README.md
COPY packages/sandboxed_gym/src /app/packages/sandboxed_gym/src

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/gym-host-venv \
    uv sync \
        --project /app/docker/locks/nmp-gym-host \
        --locked \
        --no-dev \
        --no-install-project \
        --no-editable

# Build the per-server venvs now rather than on first run. Gym creates one venv per agent, model and
# resources server the config selects, which is ~3 GB and several minutes -- paid once here instead
# of on every container start.
ENV PATH="/opt/gym-host-venv/bin:$PATH" \
    NEMO_GYM_VENV_DIR=/opt/gym_venvs \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv
RUN mkdir -p "${NEMO_GYM_VENV_DIR}" "${UV_CACHE_DIR}"
RUN gym env prefetch "+config_paths=[responses_api_agents/simple_agent/configs/simple_agent.yaml,responses_api_models/inference_provider/configs/inference_provider.yaml,resources_servers/mcqa/configs/mcqa.yaml]" \
      "++policy_base_url=http://127.0.0.1:9/v1" "++policy_api_key=unused" "++policy_model_name=unused" \
      "++simple_agent.responses_api_agents.simple_agent.resources_server.name=mcqa" \
      "++uv_venv_dir=${NEMO_GYM_VENV_DIR}" "++uv_cache_dir=${UV_CACHE_DIR}" \
    || echo "prefetch failed; venvs will be built at run time"

FROM ${NMP_PYTHON_IMAGE} AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv because Gym uses it to launch component environments.
COPY --from=builder /bin/uv /bin/uv
# Copy the locked host Python environment; this supplies Gym and sandboxed_gym.
COPY --from=builder /opt/gym-host-venv /opt/gym-host-venv
# Copy Gym's prebuilt per-component environments to avoid runtime installation.
COPY --from=builder /opt/gym_venvs /opt/gym_venvs
# Copy the matching uv cache used by those component environments.
COPY --from=builder /home/ubuntu/.cache/uv /home/ubuntu/.cache/uv

ENV PATH="/opt/gym-host-venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/gym-host-venv \
    PYTHONUNBUFFERED=1 \
    NEMO_GYM_VENV_DIR=/opt/gym_venvs \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv \
    NMP_RUNTIME_HTTP_PORT=8080

# Create the mount points used by the OpenSandbox provider.
RUN mkdir -p /job/environment /job/work
WORKDIR /job/work

EXPOSE 8080
CMD ["python", "-m", "sandboxed_gym.runtime.gym_host_runtime"]

FROM runtime AS smoke-test
RUN gym --help >/dev/null \
    && python -c \
        'import importlib.metadata as m; import nemo_gym, pydantic, yaml; from sandboxed_gym.runtime import gym_host_runtime; print({"nemo-gym": m.version("nemo-gym"), "nemo-sandboxed-gym": m.version("nemo-sandboxed-gym"), "pydantic": pydantic.__version__, "pyyaml": yaml.__version__, "runtime": gym_host_runtime.__name__})'

FROM runtime
