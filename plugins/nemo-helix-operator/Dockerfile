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
