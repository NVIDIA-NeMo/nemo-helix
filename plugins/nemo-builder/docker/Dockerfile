# syntax=docker/dockerfile:1
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The image the three TRUSTED steps run as, and the credential broker. Not the sandbox, which runs
# the kaniko image Dockerfile.kaniko builds.
#
# One image for all of them, even though each holds different things. The isolation between
# them is the ServiceAccount and the mounts each is given, not the filesystem: separate images
# would imply the binary is what separates `fetch` from `push`, or `push` from the broker, and it
# is not.
#
# Built with plain `docker build`, not as a target in docker-bake.hcl like the platform's other
# images.
FROM gcr.io/go-containerregistry/crane:debug AS crane
FROM ghcr.io/sigstore/cosign/cosign:v2.5.3 AS cosign

FROM python:3.12-slim AS runtime

# `push` shells out to crane; the broker to crane and cosign. They are copied from their upstream
# images rather than curl'd, so the version is pinned by the FROM lines above and is visible in a
# diff.
COPY --from=crane /ko-app/crane /usr/local/bin/crane
COPY --from=cosign /ko-app/cosign /usr/local/bin/cosign

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

WORKDIR /src
# The workspace paths the plugin's path-dependencies resolve through. Narrower than the repo:
# see Dockerfile.dockerignore.
COPY pyproject.toml uv.lock ./
COPY sdk ./sdk
COPY packages ./packages
COPY plugins/nemo-builder ./plugins/nemo-builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache ./plugins/nemo-builder

# Proves the console script resolves and each step module imports, at BUILD time rather than
# twenty minutes into a job. A missing transitive dependency is otherwise discovered by a pod.
RUN nhx-build --help 2>&1 | grep -q 'fetch|supervise|push' \
 && python -c "import nemo_builder_plugin.run.fetch, nemo_builder_plugin.run.supervise, nemo_builder_plugin.run.push" \
 && crane version && cosign version 2>&1 | head -3

WORKDIR /
ENTRYPOINT []
