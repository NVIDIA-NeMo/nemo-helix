# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sandboxed Gym host runtime for Evaluator agent-evaluation jobs.
#
ARG NHX_PYTHON_BASE=nhx-python-base
ARG NHX_PYTHON_IMAGE=python:3.13.16-slim-trixie

FROM ${NHX_PYTHON_BASE} AS builder

# Copy the host's standalone dependency manifest and lockfile.
COPY docker/locks/nhx-gym-host /app/docker/locks/nhx-gym-host

# Copy local sources used by the standalone lock, including build-system requirements.
COPY packages/nhx_build_tools /app/packages/nhx_build_tools

# Copy the complete package inputs at the path used by the lock's local source.
COPY packages/sandboxed_gym/pyproject.toml /app/packages/sandboxed_gym/pyproject.toml
COPY packages/sandboxed_gym/README.md /app/packages/sandboxed_gym/README.md
COPY packages/sandboxed_gym/src /app/packages/sandboxed_gym/src

RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/gym-host-venv \
    uv sync \
        --project /app/docker/locks/nhx-gym-host \
        --locked \
        --no-dev \
        --no-install-project \
        --no-editable

# CVE remediation for the gym-host base venv.
# cryptography 50.x is not lockable yet because nemo-gym pulls mlflow>=3.15.1,
# whose current metadata caps cryptography at <50. Pin the post-sync override so
# the scanner-visible artifact is deterministic until that upstream cap moves.
# Same pattern as docker/Dockerfile.nhx-gym-tasks (GHSA-g6cj-pr64-35w5).
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy \
    uv pip install --python /opt/gym-host-venv/bin/python --no-deps "cryptography==50.0.0"

# Build the per-server venvs now rather than on first run. Gym creates one venv per agent, model and
# resources server the config selects, which is ~3 GB and several minutes -- paid once here instead
# of on every container start.
ENV PATH="/opt/gym-host-venv/bin:$PATH" \
    NEMO_GYM_VENV_DIR=/opt/gym_venvs \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv
RUN mkdir -p "${NEMO_GYM_VENV_DIR}" "${UV_CACHE_DIR}"
RUN if UV_LINK_MODE=copy gym env prefetch "+config_paths=[responses_api_agents/simple_agent/configs/simple_agent.yaml,responses_api_models/inference_provider/configs/inference_provider.yaml,resources_servers/mcqa/configs/mcqa.yaml]" \
      "++policy_base_url=http://127.0.0.1:9/v1" "++policy_api_key=unused" "++policy_model_name=unused" \
      "++simple_agent.responses_api_agents.simple_agent.resources_server.name=mcqa" \
      "++uv_venv_dir=${NEMO_GYM_VENV_DIR}" "++uv_cache_dir=${UV_CACHE_DIR}"; then \
      uv cache clean mlflow ray; \
    else \
      echo "prefetch failed; venvs will be built at run time"; \
    fi

# Full MLflow carries gateway server CVEs, while this image only needs Gym's client-side imports.
# Prefetched Gym venvs live at ${NEMO_GYM_VENV_DIR}/<kind>/<server>/.venv, not one directory deep.
RUN find /opt/gym-host-venv "${NEMO_GYM_VENV_DIR}" \
      \( -type f -o -type l \) -path '*/bin/python' \
    | while IFS= read -r py; do \
        uv pip uninstall --python "${py}" mlflow || true; \
      done \
    && find "${UV_CACHE_DIR}" -path '*/archive-v0/*/mlflow-*.dist-info' -prune \
      -exec sh -c 'for path; do rm -rf "$(dirname "${path}")"; done' sh {} + \
    && test -z "$(find /opt/gym-host-venv "${NEMO_GYM_VENV_DIR}" "${UV_CACHE_DIR}" \
      -path '*/mlflow-*.dist-info/METADATA' -print -quit)"

# Ray's bundled Java jars carry httpcore5 CVEs and are only needed for JVM workers.
RUN find /opt/gym-host-venv "${NEMO_GYM_VENV_DIR}" "${UV_CACHE_DIR}" \
      -path '*/ray/jars' -prune -exec rm -rf {} + \
    && test -z "$(find /opt/gym-host-venv "${NEMO_GYM_VENV_DIR}" "${UV_CACHE_DIR}" \
      -path '*/ray/jars/*' -print -quit)"

ARG NHX_COLLECT_SOURCES=0
ENV NHX_COLLECT_SOURCES=${NHX_COLLECT_SOURCES}
# Update each installed Python environment and rebuild embedded Go TLS clients.
RUN --mount=from=golang:1.27.3,source=/usr/local/go,target=/usr/local/go,readonly \
    --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=bind,source=docker/scripts/update-python-trust.py,target=/tmp/update-python-trust.py,readonly \
    --mount=type=bind,source=docker/scripts/collect-go-module-sources.py,target=/tmp/collect-go-module-sources.py,readonly \
    GOTOOLCHAIN=local GOROOT=/usr/local/go python3 /tmp/update-python-trust.py

# Sources of the host venv, each prefetched component venv, CPython, and uv and its crates, collected
# after the update above so they match what ships; the update collects the rebuilt wandb-cores' Go
# modules into rebuilt-wandb. The uv cache's distributions are collected too: it also holds the build
# backends uv fetched to build sdists, such as setuptools, at whatever version was current.
# docker/sources/nhx-gym-host.json pins ray, which has no sdist, the packages ray bundles, and the
# source of distlib's Windows launchers.
COPY docker/sources/ /tmp/nhx-sources/
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv uv run --no-project --python /opt/gym-host-venv/bin/python \
        /usr/local/bin/collect-python-sdists.py \
        --python /opt/gym-host-venv/bin/python \
        --output /source-distributions/nhx-gym-host \
        --lock-file /app/docker/locks/nhx-gym-host/uv.lock \
        --label gym-host-venv && \
    find "${NEMO_GYM_VENV_DIR}" -mindepth 3 -maxdepth 3 -type d -name .venv | sort | while IFS= read -r venv; do \
        label="$(dirname "${venv#"${NEMO_GYM_VENV_DIR}"/}" | tr / -)"; \
        UV_CACHE_DIR=/root/.cache/uv uv run --no-project --python "${venv}/bin/python" \
            /usr/local/bin/collect-python-sdists.py \
            --python "${venv}/bin/python" \
            --output /source-distributions/nhx-gym-host \
            --label "${label}" || exit 1; \
    done && \
    python3 /usr/local/bin/collect-python-sdists.py \
        --uv-cache "${UV_CACHE_DIR}" \
        --output /source-distributions/nhx-gym-host \
        --label uv-cache && \
    UV_CACHE_DIR=/root/.cache/uv uv run --no-project --python /opt/gym-host-venv/bin/python \
        /usr/local/bin/collect-cpython-source.py \
        --python /opt/gym-host-venv/bin/python \
        --output /source-distributions/nhx-gym-host && \
    python3 /usr/local/bin/collect-pinned-sources.py \
        --manifest /tmp/nhx-sources/uv.json \
        --manifest /tmp/nhx-sources/nhx-gym-host.json \
        --installed-in /opt/gym-host-venv/bin/python \
        $(find "${NEMO_GYM_VENV_DIR}" -mindepth 3 -maxdepth 3 -type d -name .venv -printf '--installed-in %p/bin/python ') \
        --output /source-distributions/nhx-gym-host && \
    python3 /usr/local/bin/collect-cargo-crate-sources.py \
        --binary /bin/uv \
        --cargo-lock /source-distributions/nhx-gym-host/pinned/uv-0.13.4.tar.gz \
        --output /source-distributions/nhx-gym-host

FROM ${NHX_PYTHON_IMAGE} AS runtime-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* && \
    dpkg --compare-versions "$(dpkg-query -W -f='${Version}' ca-certificates)" ge "20250419+deb13u1"

# The runtime's own Debian packages, not the builder's build tools.
FROM runtime-base AS apt-sources
ARG NHX_COLLECT_SOURCES=0
ENV NHX_COLLECT_SOURCES=${NHX_COLLECT_SOURCES}
COPY docker/scripts/collect-apt-sources.sh /usr/local/bin/
RUN bash /usr/local/bin/collect-apt-sources.sh /source-distributions/nhx-gym-host/apt --installed

FROM runtime-base AS runtime

# Copy uv because Gym uses it to launch component environments.
COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
RUN --mount=type=bind,source=docker/scripts/update-python-trust.py,target=/tmp/update-python-trust.py,readonly \
    python3 /tmp/update-python-trust.py --root /usr/local --bundles-only
# Copy the locked host Python environment; this supplies Gym and sandboxed_gym.
COPY --from=builder /usr/share/nemo-helix/source-distributions/rebuilt-wandb/ /usr/share/nemo-helix/source-distributions/rebuilt-wandb/
COPY --from=builder /opt/gym-host-venv /opt/gym-host-venv
# Copy Gym's prebuilt per-component environments to avoid runtime installation.
COPY --from=builder /opt/gym_venvs /opt/gym_venvs
# Copy the matching uv cache used by those component environments.
COPY --from=builder /home/ubuntu/.cache/uv /home/ubuntu/.cache/uv
COPY --from=builder /source-distributions/nhx-gym-host/ /usr/share/nemo-helix/source-distributions/nhx-gym-host/
COPY --from=apt-sources /source-distributions/nhx-gym-host/ /usr/share/nemo-helix/source-distributions/nhx-gym-host/

ENV PATH="/opt/gym-host-venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/gym-host-venv \
    PYTHONUNBUFFERED=1 \
    NEMO_GYM_VENV_DIR=/opt/gym_venvs \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv \
    NHX_RUNTIME_HTTP_PORT=8080

# Create the mount points used by the OpenSandbox provider.
RUN mkdir -p /job/environment /job/work
WORKDIR /job/work

EXPOSE 8080
CMD ["python", "-m", "sandboxed_gym.runtime.gym_host_runtime"]

FROM runtime AS smoke-test
RUN gym --help >/dev/null \
    && test -z "$(find /opt/gym-host-venv /opt/gym_venvs /home/ubuntu/.cache/uv \
        -path '*/mlflow-*.dist-info/METADATA' -print -quit)" \
    && test -z "$(find /opt/gym-host-venv /opt/gym_venvs /home/ubuntu/.cache/uv \
        -path '*/ray/jars/*' -print -quit)" \
    && python -c \
        'import importlib.metadata as m; import nemo_gym, pydantic, yaml; from sandboxed_gym.runtime import gym_host_runtime; print({"nemo-gym": m.version("nemo-gym"), "nemo-sandboxed-gym": m.version("nemo-sandboxed-gym"), "pydantic": pydantic.__version__, "pyyaml": yaml.__version__, "runtime": gym_host_runtime.__name__})'

FROM runtime
