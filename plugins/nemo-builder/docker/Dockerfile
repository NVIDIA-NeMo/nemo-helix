# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The image the build steps run in; it also carries cosign, to check what the push step signs (README,
# step 8). The sandbox runs Dockerfile.kaniko's.
FROM gcr.io/go-containerregistry/crane:v0.22.1@sha256:1f968817b95790bed063f71175aa6b8ff879fa17064020415f3e18bb6e6a36e1 AS crane
FROM ghcr.io/sigstore/cosign/cosign:v2.5.3 AS cosign

FROM python:3.12-slim AS runtime

COPY --from=crane /ko-app/crane /usr/local/bin/crane
COPY --from=cosign /ko-app/cosign /usr/local/bin/cosign

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

WORKDIR /src
# The plugin's path dependencies resolve through these.
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY plugins/nemo-builder ./plugins/nemo-builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache ./plugins/nemo-builder

# A missing dependency fails the build rather than a job.
RUN nhx-build --help 2>&1 | grep -q 'fetch|supervise|push' \
 && python -c "import nemo_builder_plugin.run.fetch, nemo_builder_plugin.run.supervise, nemo_builder_plugin.run.push" \
 && crane version && cosign version 2>&1 | head -3

WORKDIR /
ENTRYPOINT []
