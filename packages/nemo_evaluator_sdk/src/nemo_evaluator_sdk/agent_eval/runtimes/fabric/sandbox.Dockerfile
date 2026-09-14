# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Image for FabricContainerRuntime: NeMo Fabric plus the harness adapters, installed from published
# wheels. Fabric's adapter descriptors ship inside those wheels (under `share/nemo-fabric/adapters`),
# so a wheel-only install resolves any bundled harness with no source checkout and no build context
# beyond the pins in requirements.txt (written by image.py).
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
COPY requirements.txt /tmp/requirements.txt
# Installed into the image's own interpreter rather than a venv: Fabric spawns its Python adapter host
# as the absolute `/usr/local/bin/python`, so packages in a venv are invisible to the adapter.
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt
RUN python -c "from nemo_fabric import Fabric, FabricConfig"
# Run agent-generated code as a non-root user: this sandbox runs untrusted, agent-produced content, so
# dropping root narrows the blast radius of a container escape. Pre-create and own the fixed /in
# (seeded inputs) and /out (workspace + results) trees, since a non-root process cannot mkdir under /
# at exec time and the runtime creates /out/{workspace,relay,artifacts,logs} then.
RUN useradd --create-home --uid 1000 sandbox \
 && mkdir -p /in /out \
 && chown -R sandbox:sandbox /in /out
WORKDIR /out
USER sandbox
