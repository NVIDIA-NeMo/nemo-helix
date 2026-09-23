# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Managed by `nemo agents package` — safe to delete if you take ownership.
ARG BASE_IMAGE_URL=nvcr.io/nvidia/base/ubuntu
ARG BASE_IMAGE_TAG=noble-20260217
ARG PYTHON_VERSION=3.13
ARG NAT_VERSION=1.6.0
FROM ${BASE_IMAGE_URL}:${BASE_IMAGE_TAG}
ARG PYTHON_VERSION
ARG NAT_VERSION

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1

# Keep the uv-managed Python in a world-readable location and use copy
# link-mode so the venv is self-contained (no cross-directory symlinks),
# letting the non-root runtime user exec it without needing /root access.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_LINK_MODE=copy

RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ gcc ca-certificates curl && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /workspace

COPY ./ /workspace

RUN --mount=type=cache,id=uv_cache,target=/root/.cache/uv,sharing=locked \
    uv venv --python ${PYTHON_VERSION} /workspace/.venv && \
    . /workspace/.venv/bin/activate && \
    test -n "${NAT_VERSION}" || { echo "NAT_VERSION build-arg is required" >&2; exit 1; } && \
    uv pip install --prerelease=allow "nvidia-nat[most]==${NAT_VERSION}" && \
    chmod -R a+rX /opt/uv /workspace/.venv

LABEL org.opencontainers.image.title="hello_world" \
      org.opencontainers.image.version="26.06.01" \
      org.opencontainers.image.authors="NeMo Agents Team" \
      org.opencontainers.image.created="2026-06-01T10:11:07-07:00" \
      org.opencontainers.image.description="react_agent agent" \
      org.opencontainers.image.revision="40924998df45aef404781ca39c9bdd9000016bb7" \
      org.opencontainers.image.source="git@github.com:NVIDIA-NeMo/nemo-helix.git" \
      com.nemo.agent.id="9ee8d3b15822" \
      com.nemo.agent.framework="nemo_agent_toolkit" \
      com.nemo.agent.nat-version="1.6.0" \
      com.nemo.agent.contract-version="0.1.0"

ENV NAT_CONFIG_FILE=/workspace/hello_world.yaml

ENV PATH="/workspace/.venv/bin:$PATH"

RUN if getent passwd 1000 >/dev/null; then userdel -rf "$(getent passwd 1000 | cut -d: -f1)" 2>/dev/null || true; fi && \
    if getent group  1000 >/dev/null; then groupdel -f "$(getent group  1000 | cut -d: -f1)" 2>/dev/null || true; fi && \
    groupadd -g 1000 agent && useradd -u 1000 -g agent -m agent && \
    chown -R agent:agent /workspace
USER agent

ENTRYPOINT ["sh", "-c", "exec nat serve --config_file=$NAT_CONFIG_FILE --host 0.0.0.0"]
