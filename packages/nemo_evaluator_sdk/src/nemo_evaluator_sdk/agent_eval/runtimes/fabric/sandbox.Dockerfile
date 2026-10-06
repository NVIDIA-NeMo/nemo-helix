# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Image for FabricAgentRuntime's sandbox mode: NeMo Fabric plus the harness adapters. The Python adapters
# install from published wheels; Fabric's adapter descriptors ship inside those wheels (under
# `share/nemo-fabric/adapters`), so a wheel-only install resolves any bundled harness. The Pi adapter is an
# npm package, so Node and the adapter's pinned npm dependencies are baked in too, and its descriptor is
# rewritten into the same share tree. No source checkout and no build context beyond the pins in
# requirements.txt and pi-package.json (written by image.py).
ARG PYTHON_VERSION=3.12

# Node + the Fabric Pi adapter + its pinned Pi SDK peers, installed in a node stage and copied into the
# runtime image so the adapter's process runner executes against the image's own node.
FROM node:24-bookworm-slim AS pi
WORKDIR /opt/pi-adapter
COPY pi-package.json package.json
RUN npm install --no-audit --no-fund --no-update-notifier

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
COPY requirements.txt /tmp/requirements.txt
# Installed into the image's own interpreter rather than a venv: Fabric spawns its Python adapter host
# as the absolute `/usr/local/bin/python`, so packages in a venv are invisible to the adapter.
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt
RUN python -c "from nemo_fabric import Fabric, FabricConfig"
# Node for the Pi adapter's process runner, the adapter install itself, and npm for in-sandbox use.
# node:24-bookworm-slim is the same Debian release as the runtime image, so the binary runs unmodified.
COPY --from=pi /usr/local/bin/node /usr/local/bin/node
COPY --from=pi /opt/pi-adapter /opt/pi-adapter
COPY --from=pi /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
# Fabric discovers the Pi adapter the same way it discovers the wheel adapters: a descriptor under
# `<sysconfig data>/share/nemo-fabric/adapters/`. The npm package ships its descriptor with a runner
# script relative to the package root, so rewrite it to an absolute path into the install dir before
# installing it into the share tree.
RUN node -e 'const fs=require("node:fs"),path=require("node:path");const p="/opt/pi-adapter/node_modules/nemo-fabric-adapters-pi/pi.fabric-adapter.json";const d=JSON.parse(fs.readFileSync(p,"utf8"));d.runner.script=path.join(path.dirname(p),d.runner.script);const out="/usr/local/share/nemo-fabric/adapters/pi/pi.fabric-adapter.json";fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(d,null,2)+"\n");'
# Run agent-generated code as a non-root user: this sandbox runs untrusted, agent-produced content, so
# dropping root narrows the blast radius of a container escape. Pre-create and own the fixed /in
# (seeded inputs) and /out (workspace + results) trees, since a non-root process cannot mkdir under /
# at exec time and the runtime creates /out/{workspace,relay,artifacts,logs} then.
RUN useradd --create-home --uid 1000 sandbox \
 && mkdir -p /in /out \
 && chown -R sandbox:sandbox /in /out
WORKDIR /out
USER sandbox
