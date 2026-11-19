# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The helix-operator pod. Build from the repo root:
#   docker build -f plugins/nemo-helix-operator/Dockerfile --build-arg tag=<nhx-api tag> .
ARG tag=70332d17b8562e2b1070da40d733210b19e43b70

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
COPY plugins/nemo-helix-operator /src/plugins/nemo-helix-operator
# --no-sources: the plugin's workspace sources don't exist outside the repo, and --no-deps needs none of them.
RUN uv pip install --no-deps --no-sources --target /out /src/plugins/nemo-helix-operator

# nhx-api has no pip, but already carries nemo-builder-plugin and every other dependency.
FROM ghcr.io/nvidia-nemo/nemo-helix/nhx-api:${tag}
COPY --chown=1000:1000 --from=0 /out/ /app/.venv/lib/python3.13/site-packages/
# Deploy runs `helm upgrade`; --reset-then-reuse-values needs Helm 3.14 or later.
COPY --from=docker.io/alpine/helm:3.16.1 /usr/bin/helm /usr/local/bin/helm
# Test runs import the uploaded checkout's e2e conftest, which needs these. Versions are uv.lock's; the checkout's
# own nhx-testing and nemo-nb come from its source tree.
RUN uv pip install --python /app/.venv/bin/python \
    pytest==9.1.1 pytest-asyncio==1.4.0 pytest-cov==7.1.0 pytest-env==1.7.0 pytest-rerunfailures==16.4 \
    pytest-timeout==2.4.0 pytest-xdist==3.8.0 pytest-httpserver==1.1.5 testcontainers==4.15.0 sphinx==9.1.0 \
    httpx2==2.13.0
